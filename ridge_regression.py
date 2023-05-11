from typing import Union, List
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import pickle
import os
import glob

sys.path.append('~/partIII-amino-acid-prediction/')

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as geom_DataLoader
from Bio.Data.IUPACData import protein_letters_1to3

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from gvp_antonia.main import ModelWrapper
from protein_engineering.pipeline import AADataset
from protein_engineering.utils.de_dataset import DirectedEvolutionDataset
from protein_engineering.protein_gym import ProteinGymDataset
from protein_engineering.utils.regression_metrics import get_spearman_fractions, wt_improvement_metric, topk_median

MAPPER = './data/mapping.csv'
MODEL_PATH = './data/eqgat_debug.ckpt'
EMBEDDINGS_PATH = './data/dummy_embeddings'
DATA_DIR = '/Users/antoniaboca/partIII-amino-acid-prediction/data'
AA_INDEX = './protein_engineering/utils/aa_index_pca19.npy'
TRANCEPTION = './data/ProteinGym_Tranception_scores/substitutions'

RANDOM_SEEDS = [
      42,	6731,	7390,	2591,	7886,
    9821,	2023,	5376,	2199,	898,
    4221,	6956,	9112,	6626,	423,
    5657,	3248,	5527,	4653,	1422,
]

_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

_1toInt = lambda aa: _amino_acids(protein_letters_1to3[aa.upper()].upper())

def get_embeddings(trainer : pl.Trainer, loader : geom_DataLoader, dataset_name : str, device, name : str):
    trainer.eval()
    print('Retrieving model logits...')
    embed_path = Path(f'{EMBEDDINGS_PATH}/{name}/{dataset_name}.pkl')
    if embed_path.exists():
        with open(embed_path, 'rb') as handle:
            positions = pickle.load(handle)
            return positions
    else:
        embed_path = Path(f'{EMBEDDINGS_PATH}/{name}')
        os.makedirs(embed_path, exist_ok=True)

    positions = {}
    for batch in tqdm(loader):
        with torch.no_grad():
            batch = batch.to(device)
            logits = trainer.model(batch)
            if logits.dim() > 2:
                # THIS IS A HOMO MULTIMER, TAKE THE AVERAGE SCORE
                assert logits.dim() == 3 and logits.shape[-1] == 20
                logits = torch.mean(logits, dim=1)
            
            for bidx in range(len(batch)):
                pos = int(batch[bidx].masked_res_id)
                positions[pos] = logits[bidx].cpu().numpy()

    embed_path = Path(f'{EMBEDDINGS_PATH}/{name}/{dataset_name}.pkl')
    with open(embed_path, 'wb') as handle:
        pickle.dump(positions, handle)

    return positions

# STEP 1: load all mutations
# STEP 2: create embeddings
# STEP 3: sample training_data, testing_data
# STEP 4: create features based on embeddings
# STEP 5: run actual ridge regression, save coefficient 
# STEP 6: Repeat steps 3-6 any number of times

def get_features(
        sequences : pd.Series, 
        scores : dict, 
        variants : pd.Series,  
        embeddings_type : str = 'one_hot', 
        add_score : bool = True
):
    seqs = sequences.to_numpy()
    if embeddings_type == 'one_hot':
        features = np.stack([np.concatenate([np.eye(21)[_1toInt(letter)] for letter in sequence]) for sequence in seqs])
    
    elif embeddings_type == 'aa_index':
        aa_index = np.load(AA_INDEX)
        features = np.stack([np.concatenate([aa_index[_1toInt(letter)] for letter in sequence]) for sequence in seqs])
    else:
        raise NotImplementedError('Unknown amino-acid embedding for feature creation.')

    if not add_score:
        return features, []

    confidences = np.array(variants.apply(lambda x:int(scores[int(x[1:-1])][_1toInt(x[-1])]) if x != '' else 0)).reshape(-1, 1)
    # total_feats = np.concatenate([features, confidences], axis=1)
    return features, confidences
    
def ridge_regression(
    wildtype : DirectedEvolutionDataset,
    mapper : pd.DataFrame,
    training_sizes : List[int],
    work_dir : str,
    iterations : int = 20,
    random_seeds : List[int] = RANDOM_SEEDS,
    model : str = 'eqgat',
    model_path : str = MODEL_PATH,
    n_layers : int = 5,
    data_dir : str = DATA_DIR,
    embeddings_type : str = 'one_hot',
    add_score : bool = True,
    **loader_kwargs
):
    assert model in ['eqgat', 'gvp', 'tranception'], 'Unrecognised model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AADataset(
        wildtype, 
        mapper, 
        structure_dir=os.path.join(data_dir, 'ProteinGym_assemblies_clean')
    )

    if dataset.skip:
        print(f'Skipping {dataset.name}.')
        return

    dataset_dir = os.path.join(work_dir, dataset.name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if model in ['eqgat', 'gvp']:
        loader = geom_DataLoader(dataset, **loader_kwargs)
        example = next(iter(loader))
        trainer = ModelWrapper(model, 1e-3, example, 0.0, n_layers=n_layers)
        trainer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
        trainer.to(device)
        embeddings = get_embeddings(trainer, loader, dataset.name, device, model)
    else:
        df = pd.read_csv(os.path.join(TRANCEPTION, f'{dataset.name}.csv'))
        mask = (df['mutant'].str.contains(',') | df['mutant'].str.contains(':'))
        df = df[~mask]
        scores = df['Tranception_L_retrieval']
        df = df.rename(columns={'Tranception_L_retrieval': 't_score', 'mutant': 'variant'})

    # NOTE: we only work with single-mutants. ProteinMPNN does not know how to handle other types of mutants    
    mask = ~(wildtype.data['variant'].str.contains(','))
    single_mutant = wildtype.data[mask].reset_index(drop=True)
    if not add_score:
        model = 'basic'

    if model == 'tranception':
        single_mutant = single_mutant.merge(df[['variant', 't_score']], on='variant', how='inner')

    full_training = int(0.8 * len(single_mutant))
    try:
        for N in training_sizes + [full_training]:
            if N > full_training:
                continue
            
            print(f'Ridge regression for {dataset.name} with training size {N}...')
            save_dir = os.path.join(dataset_dir, f'training_{N}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for iteration in range(iterations):
                test_sample = single_mutant.sample(frac=0.2, random_state=RANDOM_SEEDS[iteration])
                if model in ['eqgat', 'gvp', 'basic']:
                    X_test_A, X_test_B = get_features(test_sample['sequence'], embeddings, test_sample['variant'], embeddings_type=embeddings_type, add_score=add_score)
                else: # model is Tranception
                    X_test_A, _ = get_features(test_sample['sequence'], None, test_sample['variant'], embeddings_type=embeddings_type, add_score=False)
                    X_test_B = test_sample['t_score'].to_numpy().reshape(-1, 1)
                
                y_test = test_sample['fitness'].to_numpy()
                y_wt =  wildtype.data[wildtype.data['is_wildtype'] == True].iloc[0]['fitness']
                
                training_data = single_mutant.drop(test_sample.index)
                train_sample = training_data.sample(n=N, random_state=RANDOM_SEEDS[iteration])

                if model in ['eqgat', 'gvp', 'basic']:
                    X_train_A, X_train_B = get_features(train_sample['sequence'], embeddings, train_sample['variant'], embeddings_type=embeddings_type, add_score=add_score)
                else: # model is Tranception
                    X_train_A, _ = get_features(train_sample['sequence'], None, train_sample['variant'], embeddings_type=embeddings_type, add_score=False)
                    X_train_B = train_sample['t_score'].to_numpy().reshape(-1, 1)
                y_train = train_sample['fitness'].to_numpy()

                
                cv_scores = []
                alphas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
                for alpha in alphas:
                    if add_score:
                        scaler_a = StandardScaler() 
                        scaler_b = StandardScaler()  

                        preprocessor = ColumnTransformer(
                        transformers=[
                            ('scaler_a', scaler_a, slice(0, X_train_A.shape[-1])),
                            ('scaler_b', scaler_b, slice(X_train_A.shape[-1], X_train_A.shape[-1] + 1))
                        ])

                        ridge = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('regressor', Ridge(alpha=alpha))
                        ])
                    else:
                        ridge = Ridge(alpha=alpha)

                    if add_score:
                        X_train = np.concatenate([X_train_A, X_train_B], axis=1)
                    else:
                        X_train = X_train_A

                    scores = cross_val_score(ridge, X_train, y_train)
                    cv_scores.append(scores.mean())
                    best_alpha = alphas[np.argmax(cv_scores)]

                # AFTER CROSS VALIDATION EVALUATE
                if add_score:
                    scaler_a = StandardScaler() 
                    scaler_b = StandardScaler()  

                    preprocessor = ColumnTransformer(
                    transformers=[
                        ('scaler_a', scaler_a, slice(0, X_train_A.shape[-1])),
                        ('scaler_b', scaler_b, slice(X_train_A.shape[-1], X_train_A.shape[-1] + 1))
                    ])

                    ridge = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', Ridge(alpha=best_alpha))
                    ])
                else:
                    ridge = Ridge(alpha=alpha)
                    
                ridge.fit(X_train, y_train)
                if add_score:
                    X_test = preprocessor.transform(np.concatenate([X_test_A, X_test_B], axis=1))
                else:
                    X_test = X_test_A

                y_pred = ridge.predict(X_test)

                test_sample[f'{model}_ridge_prediction'] = y_pred
                if N == full_training:
                    results_file = os.path.join(save_dir, f'iteration_full.csv')
                else:    
                    results_file = os.path.join(save_dir, f'iteration_{iteration}.csv')
                test_sample.to_csv(Path(results_file), index=False)

                if N == full_training:
                    break
    except Exception as e:
        print(f'Unable to do ridge regression on {dataset.name}. Error message:\n{e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--mapper', type=str, default=MAPPER)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--model', type=str, default='eqgat')
    parser.add_argument('--out_dir', type=str, default=DATA_DIR)
    parser.add_argument('--embeddings_type', type=str, default='one_hot')
    parser.add_argument('--add_score', action='store_true')
    args = parser.parse_args()

    mapper = pd.read_csv(args.mapper)
    
    if args.add_score:
        model_dir = os.path.join(args.out_dir, args.model)
        work_dir = os.path.join(model_dir, args.embeddings_type)
    else:
        work_dir = os.path.join(args.out_dir, args.embeddings_type)
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    sizes = list(range(24, 240, 24))
    # sizes = [24]

    if args.dataset == 'all':
        csv_files = glob.glob(args.out_dir + '/ProteinGym_substitutions/*.csv')
        for file in csv_files:
            try:
                print(file)
                dataset = ProteinGymDataset(Path(file))
                ridge_regression(dataset, mapper, sizes, work_dir, 
                                model=args.model, model_path=args.model_path, 
                                data_dir=args.out_dir, embeddings_type=args.embeddings_type, 
                                add_score=args.add_score)
            except Exception as e:
                print(f'Failed on {file}. Error message: \n{e}')
    else:
        dataset = ProteinGymDataset(Path(args.dataset))
        ridge_regression(dataset, mapper, sizes, work_dir, model=args.model,
                         model_path=args.model_path, data_dir=args.out_dir, embeddings_type=args.embeddings_type,
                         add_score=args.add_score)
                    



