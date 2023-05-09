import os
from evaluation_2 import plot_evaluations, binned_reduce
from protein_engineering.protein_gym import ProteinGymDataset
from pathlib import Path
from scipy.stats import spearmanr
import torchmetrics as tm

import argparse
import numpy as np
import pandas as pd
import torch
import math
import glob

def compute_all_stats(name, y_pred, y_true, rank_pred, wt_thresh):
    if (y_pred < 0).any():
        y_pred = np.exp(y_pred)


    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    rank_pred = torch.tensor(rank_pred)
    sorted_indices = torch.argsort(y_true, descending=True)
    true_rank = torch.argsort(sorted_indices) + 1
    
    spearman = spearmanr(rank_pred, true_rank)[0]
    mask = (y_true) > wt_thresh
    worse_than_WT_spearman = spearmanr(rank_pred[~mask], true_rank[~mask])[0]
    better_than_WT_spearman = spearmanr(rank_pred[mask], true_rank[mask])[0]
    ndcg = tm.functional.retrieval_normalized_dcg(y_pred, true_rank)
    # mcc = tm.functional.matthews_corrcoef(
    #     torch.where(y_pred < wt_thresh, torch.tensor(0), torch.tensor(1)), 
    #     torch.where(y_true < wt_thresh, torch.tensor(0), torch.tensor(1)),
    #     num_classes=2)
    
    y_true_binary = (y_true > wt_thresh).long()
    precision, recall, k = tm.functional.retrieval_precision_recall_curve(
            y_pred, y_true_binary, max_k=min(100, len(y_pred))
        )

    precision = precision.numpy()
    recall = recall.numpy()
    k = k.numpy()
    return {
        'name': [name],
        'spearman': [float(spearman)],
        'worse_than_WT_spearman': [float(worse_than_WT_spearman)],
        'better_than_WT_spearman': [float(better_than_WT_spearman)],
        'ndcg': [float(ndcg)],
        'precision': [np.array2string(precision)],
        'recall': [np.array2string(recall)],
        'precision_recall_ks': [np.array2string(k)]
    }


def main(args):
    work_dir = args.work_dir

    mutation_dirs = []
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            mutation_dirs.append(os.path.join(root, dir))

    all_stats = []
    # import ipdb; ipdb.set_trace()
    for mutation_dir in mutation_dirs:
        df = pd.read_csv(os.path.join(mutation_dir, f'{args.ranking}.csv'))
        # df = pd.read_csv(file)
        # path = Path(file)
        # name = path.stem
        name = df.iloc[0]['name']
        dataset = ProteinGymDataset(Path(f'./data/ProteinGym_substitutions/{name}.csv'))
        scores = dataset.data
        
        mask = scores['variant'].str.contains(',')
        scores = scores[~mask].reset_index(drop=True)
        # import ipdb; ipdb.set_trace()
        df = df.rename(columns={'mutation_code': 'variant'})
        merged = df.merge(scores, on='variant', how='inner')
        if len(merged) == 0:
            continue
        wt_thresh = scores[scores['is_wildtype'] == True].iloc[0]['fitness']

        # sorted_indices = torch.argsort(torch.tensor(merged['mutation_confidence']), descending=True)
        # pred_rank = torch.argsort(sorted_indices) + 1
        # merged['rank'] = pred_rank
        if args.ranking == 'positional':
            merged['mutation_confidence'] = 1. / merged['rank']
        stats = compute_all_stats(name, merged['mutation_confidence'], merged['fitness'], merged['rank'], wt_thresh)
        all_stats.append(pd.DataFrame(stats))
    return pd.concat(all_stats)

# Global
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--out_file')
    parser.add_argument('--ranking', type=str, choices=['positional', 'global'], default='global')

    args = parser.parse_args()
    final_dataframe = main(args)
    final_dataframe.to_csv(f'./data/results/{args.ranking}/{args.out_file}.csv', index=False)

