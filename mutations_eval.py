import os
import csv
import argparse
import glob

import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch.nn import Softmax
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as geom_DataLoader
from Bio.Data.IUPACData import protein_letters_3to1, protein_letters_1to3

from res_task.main import ModelWrapper

from protein_engineering.protein_gym import ProteinGymDataset
from protein_engineering.utils.de_dataset import DirectedEvolutionDataset
from protein_engineering.pipeline import AADataset

MODEL_PATH = '/Users/antoniaboca/partIII-amino-acid-prediction/data/eqgat_debug.ckpt'
DATA_DIR = '/Users/antoniaboca/partIII-amino-acid-prediction/data'
MAPPER = '/Users/antoniaboca/partIII-amino-acid-prediction/data/mapping.csv'

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

_codes = lambda x: {
    0: 'ALA',
    1: 'ARG',
    2: 'ASN',
    3: 'ASP',
    4: 'CYS',
    5: 'GLU',
    6: 'GLN',
    7: 'GLY',
    8: 'HIS',
    9: 'ILE',
    10: 'LEU',
    11: 'LYS',
    12: 'MET',
    13: 'PHE',
    14: 'PRO',
    15: 'SER',
    16: 'THR',
    17: 'TRP',
    18: 'TYR',
    19 : 'VAL',
}.get(x, 'UNK')

_1toInt = lambda aa: _amino_acids(protein_letters_1to3[aa.upper()].upper())
_3to1 = lambda aa: protein_letters_3to1[aa.capitalize()]

class SingleMutation:
    def __init__(self, sequence, name, confidence, position, original_res, new_res):
        self.sequence = sequence
        self.name = name
        self.confidence = float(confidence)
        self.position = int(position)
        self.original_res = original_res
        self.new_res = new_res

class PositionMutation:
    def __init__(self, sequence, name, confidence, position, original_res, top_res):
        self.sequence = sequence
        self.name = name
        self.confidence = float(confidence)
        self.position = int(position)
        self.top_res = top_res
        self.original_res = original_res

def mutation_scoring(
    wildtype : DirectedEvolutionDataset,
    mapper : pd.DataFrame,
    work_dir : str,
    model : str = 'eqgat',
    model_path : str = MODEL_PATH,   
    n_layers : int = 5,
    data_dir : str = DATA_DIR,
    correct_only : bool = False,
    top_k : int = 4,
    af_only: bool = False,
    full_structure : bool = True,
    **loader_kwargs,
):
    assert model in ['eqgat', 'gvp'], 'Unrecognised model.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AADataset(
        wildtype, 
        mapper, 
        structure_dir=os.path.join(data_dir, 'ProteinGym_assemblies_clean'),
        alphafold_only = af_only,
        full_structure = full_structure,
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
    trainer.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    trainer.to(device)
    trainer.eval()

    softmax = Softmax(dim=-1)
    global_mutations = []
    position_mutations = []

    for batch in tqdm(loader):
        with torch.no_grad():
            batch = batch.to(device)
            out = trainer.model(batch)
            probs = softmax(out)

            labels = batch.label
            if probs.dim() == 2:
                # THIS IS A MONOMER
                log_res = torch.log(probs)
                res = torch.argmax(log_res, dim=-1)
            else:
                # WHEN DEALING WITH HOMO-OLIGOMERS, WE TAKE THE AVERAGE OF
                # THE PROBABILITY OF AN AMINO-ACID APPEARING ON ALL CHAINS

                assert probs.dim() == 3 and probs.shape[-1] == 20
                log_res = torch.log(probs).mean(dim=1)
                res = torch.argmax(log_res, dim=-1)

            for g_idx in range(len(batch)):
                predicted_res = res[g_idx]
                if (correct_only and predicted_res == labels[g_idx]) or (not correct_only):
                    # EXTRACT GLOBAL CONFIDENCES
                    for aa in range(20):
                        
                        confidence   = log_res[g_idx][aa]
                        sequence     = dataset.sequence
                        original_res = _3to1(_codes(int(labels[g_idx])))
                        new_res      = _3to1(_codes(aa))
                        position     = batch[g_idx].masked_res_id
                        name         = dataset.name
                    
                        global_mutations.append(SingleMutation(sequence, name, confidence, position, original_res, new_res))    

                    # EXTRACT POSITION CONFIDENCES
                    confidence   = log_res[g_idx][predicted_res]
                    sequence     = dataset.sequence
                    position     = batch[g_idx].masked_res_id
                    name         = dataset.name
                    original_res = _3to1(_codes(int(labels[g_idx])))

                    top_k_confidence, top_k_residues = torch.topk(log_res[g_idx], top_k, dim=-1)                    
                    mutations = [(float(conf), _3to1(_codes(int(resi)))) for conf, resi in zip(top_k_confidence, top_k_residues)]
                    position_mutations.append(PositionMutation(sequence, name, confidence, position, original_res, mutations))

    global_sorted = sorted(global_mutations, key=lambda x: x.confidence, reverse=True)
    position_sorted = sorted(position_mutations, key=lambda x: x.confidence)

    global_path = os.path.join(dataset_dir, 'global.csv')

    with open(global_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        header = ['wildtype', 'name', 'mutation_position', 'mutation_code', 'mutation_confidence', 'rank']
        # write the header
        writer.writerow(header)
        
        rank = 1
        for mutation in global_sorted:
            data = [
                mutation.sequence, 
                mutation.name,
                mutation.position, 
                f'{mutation.original_res}{mutation.position}{mutation.new_res}',
                mutation.confidence, 
                rank
            ]
            # write the data  
            writer.writerow(data)

            rank += 1
    
    positional_path = os.path.join(dataset_dir, 'positional.csv')
    with open(positional_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        header = ['wildtype', 'name', 'mutation_position', 'position_confidence', 'mutation_code', 'mutation_confidence', 'rank']
        writer.writerow(header)

        rank = 1

        for mutation in position_sorted:
            for single in mutation.top_res:
                data = [
                    mutation.sequence, 
                    mutation.name,
                    mutation.position, 
                    mutation.confidence,
                    f'{mutation.original_res}{mutation.position}{single[1]}',
                    single[0], 
                    rank
                ]
            
                writer.writerow(data)
                rank += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--mapper', type=str, default=MAPPER)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--model', type=str, default='eqgat')
    parser.add_argument('--out_dir', type=str, default=DATA_DIR)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--AF_only', action='store_true', default=False)
    parser.add_argument('--correct_only', action='store_true', default=False)
    parser.add_argument('--local_structure', action='store_true', default=False)
    args = parser.parse_args()

    mapper = pd.read_csv(args.mapper)
    
    work_dir = os.path.join(args.out_dir, 'mutation_generation', args.model, f'AF_only={args.AF_only}-correct_only={args.correct_only}-full_structure={not args.local_structure}')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    failed = []
    if args.dataset == 'all':
        csv_files = glob.glob(args.out_dir + '/ProteinGym_substitutions/*.csv')
        for file in csv_files:
            try:
                print(file)
                dataset = ProteinGymDataset(Path(file))
                mutation_scoring(dataset, mapper, work_dir, 
                            model=args.model, model_path=args.model_path, data_dir=args.out_dir, 
                            batch_size=args.batch_size, af_only=args.AF_only, correct_only=args.correct_only, full_structure=(not args.local_structure))
            except Exception as e:
                print(f'Failed on {file}. Error message: \n{e}')
                failed.append(Path(file).stem)
        print(f'Failed on {len(failed)} datasets: {failed}')
    else:
        dataset = ProteinGymDataset(Path(args.dataset))
        mutation_scoring(dataset, mapper, work_dir, 
                        model=args.model, model_path=args.model_path, data_dir=args.out_dir, 
                        batch_size=args.batch_size, af_only=args.AF_only, correct_only=args.correct_only, full_structure=(not args.local_structure))

