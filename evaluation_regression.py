import os
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torchmetrics as tm

from protein_engineering.protein_gym import ProteinGymDataset
from scipy.stats import spearmanr

RESULTS_PATH = './data/results/ridge_regression/unregularised/'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--root_path')

    args = parser.parse_args()
    if not Path(RESULTS_PATH).exists():
        os.makedirs(RESULTS_PATH, exist_ok=True)
    
    model = args.model
    assert model in ['gvp', 'eqgat', 'simple', 'tranception']

    with open(os.path.join(RESULTS_PATH, f'{model}.csv'), 'w', encoding='UTF8') as handle:
        writer = csv.writer(handle)

        header = ['embedding_type', 'name', 'training_size', 'spearmanr', 
            'better_than_WT_spearmanr', 'top_10_precision', 'top_10_recall', 'top_20_precision', 'top_20_recall']
        # write the header
        writer.writerow(header)

    for embeddings_type in ['one_hot', 'aa_index']:
        mutation_dirs = []
        base_path = os.path.join(args.root_path, f'{model}_{embeddings_type}')
        for fname in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, fname)):
                mutation_dirs.append(os.path.join(base_path, fname))
        
        for mutation_dir in mutation_dirs:
            dataset_name = Path(mutation_dir).stem
            print(f'Processing {dataset_name}...')
            dataset = ProteinGymDataset(Path(f'./data/ProteinGym_substitutions/{dataset_name}.csv'))
            wt_threshold = dataset.data[dataset.data['is_wildtype'] == True].iloc[0]['fitness']

            training_sizes = list(range(24, 240, 24))

            size_dirs = []
            for fname in os.listdir(mutation_dir):
                if os.path.isdir(os.path.join(mutation_dir, fname)):
                    size_dirs.append(os.path.join(mutation_dir, fname))
            
            for work_dir in size_dirs:
                data_size = int(Path(work_dir).stem.split('_')[1])

                average_stat = 0.0
                average_better_than_WT = 0.0
                average_top_10_precision = 0.0
                average_top_20_precision = 0.0
                average_top_10_recall = 0.0
                average_top_20_recall = 0.0
                
                if data_size not in training_sizes:
                    results = pd.read_csv(os.path.join(work_dir, f'iteration_full.csv'))
                    if model == 'simple':
                        y_pred = results[f'eqgat_ridge_prediction']
                    else:
                        y_pred = results[f'{model}_ridge_prediction']

                    y_true = results['fitness']
                    average_stat = spearmanr(y_pred, y_true)[0]

                    mask = (y_true > wt_threshold)
                    average_better_than_WT = spearmanr(y_pred[mask], y_true[mask])[0]
                    
                    precision, recall, k = tm.functional.retrieval_precision_recall_curve(
                            torch.tensor(y_pred), 
                            torch.tensor((y_true > wt_threshold)).long(), 
                            max_k = min(100, len(y_pred))
                        )
                    
                    pos_k = min(9, len(k)-1)
                    average_top_10_precision = float(precision[pos_k])
                    average_top_10_recall = float(recall[pos_k])

                    pos_k = min(19, len(k)-1)
                    average_top_20_precision = float(precision[pos_k])
                    average_top_20_recall = float(recall[pos_k])

                else:
                    for iteration in range(20):
                        results = pd.read_csv(os.path.join(work_dir, f'iteration_{iteration}.csv'))
                        if model == 'simple':
                            y_pred = results[f'eqgat_ridge_prediction']
                        else:
                            y_pred = results[f'{model}_ridge_prediction']
                        y_true = results['fitness']

                        stat = spearmanr(y_pred, y_true)[0]
                        mask = (y_true > wt_threshold)

                        better_than_WT = spearmanr(y_pred[mask], y_true[mask])[0]

                        precision, recall, k = tm.functional.retrieval_precision_recall_curve(
                            torch.tensor(y_pred), 
                            torch.tensor((y_true > wt_threshold)).long(), 
                            max_k = min(100, len(y_pred))
                        )
                        
                        average_stat += stat
                        average_better_than_WT += better_than_WT

                        pos_k = min(9, len(k)-1)
                        average_top_10_precision += float(precision[pos_k])
                        average_top_10_recall += float(recall[pos_k])
                        

                        pos_k = min(19, len(k)-1)
                        average_top_20_precision += float(precision[pos_k])
                        average_top_20_recall += float(recall[pos_k]) 
                    
                    average_stat /= 20
                    average_better_than_WT /= 20
                    
                    average_top_10_precision /= 20
                    average_top_20_precision /= 20
                    
                    average_top_10_recall /= 20
                    average_top_20_recall /= 20
                    
                with open(os.path.join(RESULTS_PATH, f'{model}.csv'), 'a', encoding='UTF8') as handle:
                    writer = csv.writer(handle)
                    row = [
                        embeddings_type,
                        dataset_name,
                        data_size,
                        average_stat,
                        average_better_than_WT,
                        average_top_10_precision,
                        average_top_10_recall,
                        average_top_20_precision,
                        average_top_20_recall
                    ]
                    writer.writerow(row)


    