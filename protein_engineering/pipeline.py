import os
import csv
import math
import heapq
import pickle
import pathlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from typing import Union
from tqdm import tqdm

import torch
from torch.nn import Softmax
from biopandas.pdb import PandasPdb
from Bio.Data.IUPACData import protein_letters_3to1

from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.loader import DataLoader as geom_DataLoader

from gvp_antonia.protein_graph import AtomGraphBuilder, _element_alphabet
from protein_engineering.protein_gym import ProteinGymDataset

STRUCTURE_PATH = '/Users/antoniaboca/partIII-amino-acid-prediction/data/ProteinGym_structures_clean'
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
# TODO STEP 1: recover the structure
_3to1 = lambda aa: protein_letters_3to1[aa.capitalize()]


# TODO STEP 2: mask out every amino-acid in a dataframe / graph
# I am assuming .pdb incoming files
# !!!!!TODO!!!!! I am giving the ENTIRE molecule to the model

class SingleMutation:
    def __init__(self, sequence, name, confidence, position, original_res, new_res):
        self.sequence = sequence
        self.name = name
        self.confidence = float(confidence)
        self.position = int(position)
        self.original_res = original_res
        self.new_res = new_res

class AADataset(IterableDataset):
    """
    Dataset of all possible masked graphs for a single molecule from ProteinGym
    WARNING: THIS DATASET ASSUMES THAT THE PDBs PROVIDED HAVE ALREADY BEEN CLEANED.
    """
    def __init__(self, 
                wildtype : ProteinGymDataset, 
                mapper : pd.DataFrame, 
                use_alphafold : bool = False,
                max_len : int = None, 
                structure_dir : Union[str, pathlib.Path] = STRUCTURE_PATH):

        self.max_len = max_len
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        self.sequence = wildtype.data[wildtype.data['is_wildtype'] == True].iloc[0]['sequence']
        self.name = wildtype.name
        self.skip = False
        self.residues = len(self.sequence)

        # DETERMINE ALL PDBS THAT MATCH THE WILDTYPE SEQUENCE
        str_identifiers = mapper[mapper['wildtype'] == self.sequence]['identifiers']
        if len(str_identifiers) > 1:
            raise Exception('Multiple wildtpye sequences found. Aborting.')
        structures = str_identifiers.iloc[0].split(',')
        
        # CHOOSE TO WORK WITH ALPHAFOLD PREDICTED STRUCTURES
        if use_alphafold:
            structures = list(filter(lambda id: id.startswith('AF'), structures)) # ASSUME ALPHAFOLD STRUCTURES LOOK LIKE AFP06654F1
            structures = [f'AF-{x[2:-2]}-F1' for x in structures]
        else:
            structures = list(filter(lambda id: not id.startswith('AF'), structures))
        
        if len(structures) > 1:
            print('WARNING: Multiple structures have been reported for this sequence.')
        
        # LOAD FIRST CLEAN STRUCTURE WE CAN FIND
        found = False
        pdb = None
        for structure in structures:
            file = Path(os.path.join(structure_dir, f'{structure}.pdb'))
            try:
                if file.exists():
                    print(file)
                    pdb = PandasPdb().read_pdb(str(file)).df['ATOM']
                    found = True
            except:
                pass
            if found:
                break
                
        if not found:
            print(f'WARNING: no clean structure found for {self.name}. Skipping for now.')
            self.skip = True
            return

        pdb = pdb.rename(columns={
                'x_coord': 'x', 
                'y_coord':'y', 
                'z_coord': 'z', 
                'element_symbol': 'element', 
                'atom_name': 'name', 
                'residue_name': 'resname'})
        
        # CHECK WHETHER THE ASSEMBLY IS HOMOMERIC
        self.chains = pdb['chain_id'].nunique()
        if self.chains > 1:
            chains = pdb['chain_id'].unique()
            sequence = pdb[pdb['chain_id'] == chains[0]].sort_values(by='residue_number')['resname'].reset_index(drop=True)
            homomeric = True
            for chain in chains[1:]:
                aux_sequence = pdb[pdb['chain_id'] == chain].sort_values(by='residue_number')['resname'].reset_index(drop=True)
                homomeric = homomeric and sequence.equals(aux_sequence)
                if not homomeric:
                    break

            if not homomeric: 
                print(f'Detected heteromeric assembly. Skipping for now. Are you sure you are using AlphaFold?')
                self.skip = True
            
        
        # DROPPING TRAILING AMINO-ACIDS (expected to have negative values or really high values)
        pdb = pdb[(pdb['residue_number'] >= 1) & (pdb['residue_number'] <= self.residues)].reset_index(drop=True)
        self.pdb = pdb

    def __iter__(self):
        length = self.residues if not self.skip else 0
        if self.max_len:
            length = min(length, self.max_len)
        indices = [x+1 for x in list(range(length))]
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
    
    def _dataset_generator(self, res_ids):
        pdb = self.pdb
        for res_id in res_ids:
            to_remove = (pdb['residue_number'] == res_id) & (~pdb['name'].isin(['N', 'CA', 'O', 'C'])) 
            my_atoms = pdb[~to_remove].reset_index(drop=True)
            label = pdb[(pdb['residue_number'] == res_id) & (pdb['name'] == 'CA')]['resname'].iloc[0]
            ca_idx = np.where((my_atoms['residue_number'] == res_id) & (my_atoms['name'] == 'CA'))[0]
            if len(ca_idx) == 1: 
                ca_idx = int(ca_idx) 
            else:
                # THIS IS A HOMOMERIC ASSEMBLY. WE MASKED ALL COPIES OF THE SAME RESIDUE.
                ca_idx = torch.tensor(ca_idx).unsqueeze(0)               

            graph = self.graph_builder(my_atoms)
            graph.label = _amino_acids(label)
            graph.ca_idx = ca_idx
            graph.masked_res_id = res_id
            graph.name = self.name
            graph.sequence = self.sequence

            yield graph

# TODO STEP 3: Pass the amino-acid through the trained model
# TODO STEP 4: Find where the model is most uncertain
# TODO STEP 5: Choose top three mutations in a single spot
def uncertainty_search(trainer, dataloader, locs=1, k=3, correct_only=True): 
    """NOTE: THIS SEARCH ASSUMES A SINGLE MOLECULE AT A TIME."""

    trainer.eval()

    all_mutations = []
    softmax = Softmax(dim=-1)

    for batch in tqdm(dataloader):
        with torch.no_grad():
            out = trainer.model(batch)
            probs = softmax(out)

            print(out)
            labels = batch.label
            res = torch.argmax(probs, dim=-1, keepdim=True)
            print(res)
            if res.dim() == 2:
                # THIS IS A HOMOMER
                for g_idx in range(len(batch)):
                    if (correct_only and res[g_idx] == labels[g_idx]) or (not correct_only):
                        # IN THIS BRANCH WE ONLY CONSIDER PLACES WHERE THE MODEL IS CORRECT
                        for aa in range(20):
                            
                            confidence = probs[g_idx][aa]
                            sequence = batch[g_idx].sequence
                            original_res = _3to1(_codes(int(labels[g_idx])))
                            new_res = _3to1(_codes(aa))
                            position = batch[g_idx].masked_res_id
                            name = batch[g_idx].name
                            
                            # pushing negative values so the lowest positive confidences have the highest priority
                            all_mutations.append(SingleMutation(sequence, name, -confidence, position, original_res, new_res))    
            else:
                raise NotImplementedError
    
    all_mutations.sort(key=lambda x: x.confidence)

    with open('mutations_all.pkl', 'wb') as handle:
        pickle.dump(all_mutations, handle)

    with open('mutations_all.csv', 'a+', encoding='UTF8') as f:
        writer = csv.writer(f)

        header = ['wildtype', 'name', 'mutation_position', 'mutation_code', 'mutation_confidence', 'rank']
        # write the header
        writer.writerow(header)
        
        rank = 1
        for mutation in all_mutations:
            data = [
                mutation.sequence, 
                mutation.name,
                mutation.position, 
                f'{mutation.original_res}{mutation.position}{mutation.new_res}',
                -mutation.confidence, 
                rank
            ]
            # write the data  
            writer.writerow(data)

            rank += 1


# TODO STEP 6: recover the sequence 
# TODO STEP 7: check against protein engineering benchmarks (ProteinGym)

DATASET_PATH = '/Users/antoniaboca/ProteinGym_substitutions'

