from evaluation_2 import plot_evaluations, generate_model_report
from protein_engineering.protein_gym import ProteinGymDataset
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import os
import pandas as pd

def full_dataset_model_report(mutation_generation_dataset): #= '/Users/antoniaboca/Downloads/EQGAT_results/EQGAT_AF_only=False-correct_only=False/'
    mutation_dirs = []
    for root, dirs, files in os.walk(mutation_generation_dataset):
            for dir in dirs:
                mutation_dirs.append(os.path.join(root, dir))

    for mutation_dir in mutation_dirs:
        print(mutation_dir)
        df_global = pd.read_csv(os.path.join(mutation_dir, 'global.csv'))
        df_local = pd.read_csv(os.path.join(mutation_dir, 'positional.csv'))
        name = df_global['name'].iloc[0]
        dataset = ProteinGymDataset(Path(f'./data/ProteinGym_substitutions/{name}.csv'))
        scores = dataset.data
        for df, mutation_type in [(df_global, 'global'), (df_local, 'positional')]:
            df = df.rename(columns={'mutation_code': 'variant'})
            merged = df.merge(scores, on='variant', how='inner')
            if len(merged) < 2:
                continue
            # if not math.isnan(spearman):
            #     avg_spearman += spearman
            dir_name = Path(mutation_generation_dataset).stem
            path = f'./data/reports/{dir_name}/{name}'
            os.makedirs(path, exist_ok=True)
            wt_thresh = scores[scores['is_wildtype'] == True].iloc[0]['fitness']
            if mutation_type == 'positional':
                merged['mutation_confidence'] = 1. / merged['rank']
            else:
                if (merged['mutation_confidence'] < 0).any():
                    merged['mutation_confidence'] = np.exp(merged['mutation_confidence'])
            generate_model_report(f'{path}/{mutation_type}.pdf', merged['mutation_confidence'], merged['fitness'], y_true_wt = wt_thresh)

if __name__ == '__main__':
    # EQGAT
    full_dataset_model_report('/Users/antoniaboca/Downloads/EQGAT_results/EQGAT_AF_only=False-correct_only=False/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/EQGAT_results/EQGAT_AF_only=False-correct_only=True/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/EQGAT_results/EQGAT_AF_only=True-correct_only=False/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/EQGAT_results/EQGAT_AF_only=True-correct_only=True/')

    # GVP
    full_dataset_model_report('/Users/antoniaboca/Downloads/GVP_results/GVP_AF_only=False-correct_only=False/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/GVP_results/GVP_AF_only=False-correct_only=True/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/GVP_results/GVP_AF_only=True-correct_only=False/')
    full_dataset_model_report('/Users/antoniaboca/Downloads/GVP_results/GVP_AF_only=True-correct_only=True/')