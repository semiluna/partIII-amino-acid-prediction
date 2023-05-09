import torch
import os
import argparse
from gvp_antonia.main import ModelWrapper
from gvp_antonia.lmdb_dataset import LMDBDataset
from gvp_antonia.protein_graph import AtomGraphBuilder
from gvp_antonia.models.gvp import RES_GVP
from gvp_antonia.models.eqgat import RES_EQGATModel


from torch_geometric.loader import DataLoader as geom_DataLoader
from torch.utils.data import DataLoader, IterableDataset

import numpy as np
import math
import random
import pytorch_lightning as pl


class RESDataset(IterableDataset):
    def __init__(self, dataset_path, max_len=None, sample_per_item=None, shuffle=False):
        self.dataset = LMDBDataset(dataset_path)
        self.graph_builder = AtomGraphBuilder(_element_mapping)
        self.shuffle = shuffle
        self.max_len = max_len
        self.sample_per_item = sample_per_item

    def __iter__(self):
        length = len(self.dataset)
        if self.max_len:
            length = min(length, self.max_len)
        indices = list(range(length))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(indices)
        else:  
            per_worker = int(math.ceil(length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, length)
            gen = self._dataset_generator(indices[iter_start:iter_end])
        return gen

    def _dataset_generator(self, indices):
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            item = self.dataset[idx]

            atoms = item['atoms']
            # Mask the residues one at a time for a single graph
            if self.sample_per_item:
                limit = self.sample_per_item
            else:
                limit = len(item['labels'])
            for sub in item['labels'][:limit].itertuples():
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa >= 20:
                    continue
                assert aa is not None

                my_atoms = atoms.iloc[item['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) != 1: continue
                graph = self.graph_builder(my_atoms)
                graph.label = aa
                graph.ca_idx = int(ca_idx)
                graph.ensemble = item['id']
                yield graph


_element_mapping = lambda x: {
        'H' : 0,
        'C' : 1,
        'N' : 2,
        'O' : 3,
        'F' : 4,
        'S' : 5,
        'Cl': 6, 'CL': 6,
        'P' : 7
    }.get(x, 8)

DATASET_PATH = '/Users/antoniaboca/Downloads/split-by-cath-topology/data'
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
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

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

MODEL_SELECT = {'gvp': RES_GVP, 'mace': None, 'eqgat': RES_EQGATModel }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--data_file')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = 'gvp'
    model_path = args.model_path
    n_layers = 5
    data_file = args.data_file
    test_dataset = RESDataset(os.path.join(data_file, 'test'))
    test_dataloader = geom_DataLoader(test_dataset, num_workers=8, batch_size=args.batch_size)
    example = next(iter(test_dataset))

    model = ModelWrapper(model, 1e-3, example, 0.0, n_layers=n_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    trainer = pl.Trainer( 
            accelerator='gpu',
            devices=1,
            num_nodes=1,
            #strategy='ddp'
        ) 

    test_result = trainer.test(model, test_dataloader)
    print(test_result)