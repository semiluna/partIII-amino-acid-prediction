
import math
import random

import numpy as np
import torch

from protein_graph import AtomGraphBuilder, _element_alphabet
from lmdb_dataset import LMDBDataset

from torch.utils.data import IterableDataset

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

class RESDataset(IterableDataset):
    def __init__(self, dataset_path, max_len=None, sample_per_item=None, shuffle=False):
        self.dataset = LMDBDataset(dataset_path)
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
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
                if aa == 20:
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
