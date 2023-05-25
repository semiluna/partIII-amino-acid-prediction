import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from protein_engineering.utils.plotstyle import matplotlib_defaults

if __name__ == '__main__':
    matplotlib.rcParams.update(matplotlib_defaults())

    all_names = set()
    for model in ['EQGAT', 'GVP']:
        for ranking in ['positional', 'global']:
            df = pd.read_csv(f'./data/results/{ranking}/{model}_AF_only=False-correct_only=True.csv')
            mask = df['better_than_WT_spearman'].isna()
            masked = df[~mask]
            name_model = set(list(masked['name'].unique()))
            all_names = all_names.union(name_model)
        
    names = list(all_names)
    idx_map = {}
    for idx, name in enumerate(names):
        idx_map[name] = idx

    for model in ['EQGAT', 'GVP']:
        plt.figure(figsize=(12, 3))
        # set_dim(fig, ratio=1./4, width=TEXTWIDTHS.PHD_THESIS)
        for ranking in ['positional', 'global']:

            df = pd.read_csv(f'./data/results/{ranking}/{model}_AF_only=False-correct_only=True.csv')
            mask = df['better_than_WT_spearman'].isna()
            masked = df[~mask]

            vals = np.ones(len(names)) * np.nan
            for idx, row in masked.iterrows():
                vals[idx_map[row['name']]] = row['better_than_WT_spearman']
            
            plt.scatter(range(len(names)), vals, marker='o')
        
        tranception = pd.read_csv('./data/results/Tranception.csv')
        mask = tranception['name'].isin(names)
        tranception = tranception[mask]

        vals = np.ones(len(names)) * np.nan
        for idx, row in tranception.iterrows():
            vals[idx_map[row['name']]] = row['better_than_WT_spearman']
            
        plt.scatter(range(len(names)), vals, marker='o', color='black', alpha=0.7)
        
        plt.xticks(range(len(names)), labels=names, rotation=90)
        plt.ylim(bottom=-1.1, top=1.1)
        plt.hlines(y=0.0, xmin=0,xmax=48, colors='gray')
        
        plt.grid(alpha=0.5)
        plt.title(f'{model}')
        plt.ylabel('Spearman rho (better than WT)')
        plt.legend(['positional ranking', 'global ranking', 'Tranception'])
        plt.show()
        # plt.savefig(f'/Users/antoniaboca/{model}_top_10_precision_per_dataset.pdf', format='pdf', bbox_inches='tight')