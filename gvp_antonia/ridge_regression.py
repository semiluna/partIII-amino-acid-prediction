from typing import Union, List
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl
from torch_geometric.lodaer import DataLoader as geom_DataLoader
from Bio.Data.IUPACData import protein_letters_1to3

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from gvp_antonia.main import ModelWrapper
from protein_engineering.pipeline import AADataset
from protein_engineering.utils.de_dataset import DirectedEvolutionDataset
from protein_engineering.utils.regression_metrics import get_spearman_fractions, wt_improvement_metric, topk_median

MODEL_PATH = '/Users/antoniaboca/partIII-amino-acid-prediction/data/eqgat_debug.ckpt'

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

_1toInt = lambda aa: _amino_acids(protein_letters_1to3[aa.capitalize()].capitalize())

def get_embeddings(trainer : pl.Trainer, loader : geom_DataLoader):
    trainer.eval()
    print('Retrieving model logits...')

    positions = {}
    for batch in tqdm(loader):
        with torch.no_grad():
            logits = trainer.model(batch)
            if logits.dim() > 2:
                # THIS IS A HOMO MULTIMER, TAKE THE AVERAGE SCORE
                assert logits.dim() == 3 and logits.shape[-1] == 20
                logits = torch.mean(logits, dim=1)
            
            for bidx in range(len(batch)):
                pos = batch[bidx].masked_res_id
                positions[pos] = logits[bidx]

    return positions

# STEP 1: load all mutations
# STEP 2: create embeddings
# STEP 3: sample training_data, testing_data
# STEP 4: create features based on embeddings
# STEP 5: run actual ridge regression, save coefficient 
# STEP 6: Repeat steps 3-6 any number of times

def get_features(
    sequences : pd.Series,
    embeddings : dict,
):
    seqs = sequences.to_numpy()
    features = seqs.apply(
        lambda sequence: np.array([embeddings[_1toInt(letter)] for letter in sequence])
    )

    return features
    
def ridge_regression(
    wildtype : DirectedEvolutionDataset,
    mapper : pd.DataFrame,
    training_sizes : List[int],
    iterations : int = 20,
    random_seeds : List[int] = RANDOM_SEEDS,
    model : str = 'eqgat',
    model_path : str = MODEL_PATH,
    n_layers : int = 5,
    **loader_kwargs
):
    assert model in ['eqgat', 'gvp'], 'Unrecognised model'

    dataset = AADataset(wildtype, mapper)
    loader = geom_DataLoader(dataset, **loader_kwargs)
    example = next(iter(loader))
    trainer = ModelWrapper(model, 1e-3, example, 0.0, n_layers=n_layers)
    trainer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    
    embeddings = get_embeddings(trainer, loader)

    test_sample = wildtype.data.sample(0.2, 42)
    training_data_full = wildtype.data.drop(test_sample.index)

    mask = ~(training_data_full.str.constains(','))
    training_data = training_data_full[mask]

    X_test = get_features(test_sample['variant'], embeddings)
    y_test = test_sample['fitness'].to_numpy()
    y_wt =  wildtype.data[wildtype.data['is_wildtype'] == True].iloc[0]['fitness']

    size_results = {}
    for N in training_sizes:

        train_sample = training_data.sample(N, random_seeds[0])
        X_train = get_features(train_sample['variant'], embeddings)
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

        spm = get_spearman_fractions(y_pred, y_test)
        r2 = model.score(X_test, y_test)
        wt_imprv = wt_improvement_metric(y_pred, y_test, y_wt)
        topk_med = topk_median(y_pred, y_test)

        size_results[N] = (spm, r2, wt_imprv, topk_med, best_alpha)
        # TODO : average statistics across 20 runs

