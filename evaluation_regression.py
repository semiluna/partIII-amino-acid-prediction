import os
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import spearmanr

RESULTS_PATH = './data/results/ridge_regression'



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

        header = ['embedding_type', 'name', 'training_size', 'spearmanr']
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

            training_sizes = list(range(24, 240, 24))

            size_dirs = []
            for fname in os.listdir(mutation_dir):
                if os.path.isdir(os.path.join(mutation_dir, fname)):
                    size_dirs.append(os.path.join(mutation_dir, fname))
            
            for work_dir in size_dirs:
                data_size = int(Path(work_dir).stem.split('_')[1])
                average_stat = 0.0
                if data_size not in training_sizes:
                    results = pd.read_csv(os.path.join(work_dir, f'iteration_full.csv'))
                    if model == 'simple':
                        y_pred = results[f'eqgat_ridge_prediction']
                    else:
                        y_pred = results[f'{model}_ridge_prediction']

                    y_true = results['fitness']
                    average_stat = spearmanr(y_pred, y_true)[0]
                else:
                    for iteration in range(20):
                        results = pd.read_csv(os.path.join(work_dir, f'iteration_{iteration}.csv'))
                        if model == 'simple':
                            y_pred = results[f'eqgat_ridge_prediction']
                        else:
                            y_pred = results[f'{model}_ridge_prediction']
                        y_true = results['fitness']

                        stat = spearmanr(y_pred, y_true)[0]
                        average_stat += stat
                    average_stat /= 20

                with open(os.path.join(RESULTS_PATH, f'{model}.csv'), 'a', encoding='UTF8') as handle:
                    writer = csv.writer(handle)
                    row = [
                        embeddings_type,
                        dataset_name,
                        data_size,
                        average_stat
                    ]
                    writer.writerow(row)


    