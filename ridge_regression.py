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

def get_embeddings(trainer : pl.Trainer, loader : geom_DataLoader, dataset_name : str, device):
    trainer.eval()
    print('Retrieving model logits...')
    embed_path = Path(f'{EMBEDDINGS_PATH}/{dataset_name}.pkl')
    if embed_path.exists():
        with open(embed_path, 'rb') as handle:
            positions = pickle.load(handle)
            return positions

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

    with open(embed_path, 'wb') as handle:
        pickle.dump(positions, handle)

    return positions

# STEP 1: load all mutations
# STEP 2: create embeddings
# STEP 3: sample training_data, testing_data
# STEP 4: create features based on embeddings
# STEP 5: run actual ridge regression, save coefficient 
# STEP 6: Repeat steps 3-6 any number of times

def get_features(sequences : pd.Series, embeddings : dict):
    seqs = sequences.to_numpy()
    features =  np.array([[embeddings[idx + 1][_1toInt(letter)] for idx, letter in enumerate(sequence)] for sequence in seqs])
    return features
    
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
    **loader_kwargs
):
    assert model in ['eqgat', 'gvp'], 'Unrecognised model'
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

    loader = geom_DataLoader(dataset, **loader_kwargs)
    example = next(iter(loader))
    trainer = ModelWrapper(model, 1e-3, example, 0.0, n_layers=n_layers)
    trainer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    trainer.to(device)
    embeddings = get_embeddings(trainer, loader, dataset.name, device)

    # test_sample = wildtype.data.sample(frac=0.2)
    # training_data_full = wildtype.data.drop(test_sample.index)
    # mask = ~(training_data_full['variant'].str.contains(','))

    # NOTE: we only work with single-mutants. ProteinMPNN does not know how to handle other types of mutants    
    mask = ~(wildtype.data['variant'].str.contains(','))
    single_mutant = wildtype.data[mask].reset_index(drop=True)

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
                test_sample = single_mutant.sample(frac=0.2)
                X_test = get_features(test_sample['sequence'], embeddings)
                y_test = test_sample['fitness'].to_numpy()
                y_wt =  wildtype.data[wildtype.data['is_wildtype'] == True].iloc[0]['fitness']
                
                training_data = single_mutant.drop(test_sample.index)
                train_sample = training_data.sample(n=N)
                X_train = get_features(train_sample['sequence'], embeddings)
                y_train = train_sample['fitness'].to_numpy()
                
                ridge = Ridge()
                cv_scores = []
                alphas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
                for alpha in alphas:
                    ridge.set_params(alpha=alpha)
                    scores = cross_val_score(ridge, X_train, y_train)
                    cv_scores.append(scores.mean())

                    best_alpha = alphas[np.argmax(cv_scores)]
                
                ridge.set_params(alpha=best_alpha)
                ridge.fit(X_train, y_train)
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
    args = parser.parse_args()

    mapper = pd.read_csv(args.mapper)
    
    work_dir = os.path.join(args.out_dir, args.model)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    sizes = list(range(24, 240, 24))
    # sizes = [24]

    if args.dataset == 'all':
        csv_files = glob.glob(args.out_dir + '/ProteinGym_substitutions/*.csv')
        for file in csv_files:
            print(file)
            dataset = ProteinGymDataset(Path(file))
            ridge_regression(dataset, mapper, sizes, work_dir, model=args.model, model_path=args.model_path, data_dir=args.out_dir)
    else:
        dataset = ProteinGymDataset(Path(args.dataset))
        ridge_regression(dataset, mapper, sizes, work_dir, model=args.model, iterations=20, model_path=args.model_path, data_dir=args.out_dir)



