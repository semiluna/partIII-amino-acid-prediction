import argparse
import os
from pathlib import Path
import pandas as pd
import csv
from scipy.stats import spearmanr

from protein_engineering.protein_gym import ProteinGymDataset

TRANCEPTION_PATH = './data/ProteinGym_Tranception_scores/substitutions'

def main(args):
    work_dir = args.work_dir
    mutation_dirs = []
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            mutation_dirs.append(os.path.join(root, dir))

    all_stats = []
    empties = []
    # import ipdb; ipdb.set_trace()
    results_path = Path('./data/tranception_corr.csv')
    if not results_path.exists():
        with open(results_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            header = ['name', 'model', 'ranking', 'spearmanr', 'better_than_WT_spearman', 'worse_than_WT_spearman']
            writer.writerow(header)
    
    for mutation_dir in mutation_dirs:
        print(mutation_dir)
        df = pd.read_csv(os.path.join(mutation_dir, f'{args.ranking}.csv'))
        
        # df = pd.read_csv(file)
        # path = Path(file)
        # name = path.stem
        if len(df) == 0:
            empties.append(mutation_dir)
            continue
        
        name = df.iloc[0]['name']
        tranception_df = pd.read_csv(os.path.join(args.tranception_dir,f'{name}.csv'))
        tranception_df = tranception_df.rename(columns={
            'Tranception_L_retrieval': 't_score',
            'mutant': 'variant',
            'DMS_score': 'fitness'
        })
        dataset = ProteinGymDataset(Path(f'./data/ProteinGym_substitutions/{name}.csv'))
        true_scores = dataset.data
        wt_thresh = true_scores[true_scores['is_wildtype'] == True].iloc[0]['fitness']
        
        df = df.rename(columns={'mutation_code': 'variant'})
        merged = df.merge(tranception_df, on='variant', how='inner')
        if len(merged) == 0:
            continue
        
        better_mask = (merged['fitness'] > wt_thresh)
        better_than_WT = spearmanr(merged[better_mask]['mutation_confidence'], merged[better_mask]['t_score'])[0]
        worse_than_WT = spearmanr(merged[~better_mask]['mutation_confidence'], merged[~better_mask]['t_score'])[0]
        avg_spearman = spearmanr(merged['mutation_confidence'], merged['t_score'])[0]

        with open(results_path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            header = [name, args.model, args.ranking, avg_spearman, better_than_WT, worse_than_WT]
            writer.writerow(header)

    print(f'Empty directories: {empties}')

# Global
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--ranking', type=str, choices=['positional', 'global'], default='global')
    parser.add_argument('--tranception_dir', type=str, default=TRANCEPTION_PATH)

    args = parser.parse_args()
    main(args)

