import csv
import math
import heapq
import pickle
from collections import defaultdict
import numpy as np

from tqdm import tqdm
import torch
from torch.nn import Softmax
from biopandas.pdb import PandasPdb
from Bio.Data.IUPACData import protein_letters_3to1

from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.loader import DataLoader as geom_DataLoader

from gvp_antonia.protein_graph import AtomGraphBuilder, _element_alphabet

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

class Pair:
    def __init__(self, pdb, confidence, mutations, mutation_confidences, masked_res, og_res):
        self.pdb = pdb
        self.confidence = float(confidence)
        self.mutations = [int(x) for x in mutations]
        self.confs = [float(x) for x in mutation_confidences]
        self.masked_residue = int(masked_res) 
        self.og_res = int(og_res)
    
    def __lt__(self, other):
        return self.confidence < other.confidence

class AADataset(IterableDataset):
    """Dataset of all possible masked graphs for a single molecule"""
    def __init__(self, wildtype, pdb=None, file=None, code=None, max_len=None):
        # TODO figure out what the wildtype is : a string? a code?
        assert sum([x is not None for x in [pdb, file, code]]) == 1, "Input must be ONE of [biopandas dataframe, path to pdb file, pdb code]"

        self.max_len = max_len
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        self.wildtype = wildtype

        if file is not None:
            pdb = PandasPdb().read_pdb(file).df['ATOM']
            pdb = pdb.rename(columns={
                    'x_coord': 'x', 
                    'y_coord':'y', 
                    'z_coord': 'z', 
                    'element_symbol': 'element', 
                    'atom_name': 'name', 
                    'residue_name': 'resname'})
        elif code is not None:
            pdb = PandasPdb().fetch_pdb(code).df['ATOM']
            pdb = pdb.rename(columns={
                    'x_coord': 'x', 
                    'y_coord':'y', 
                    'z_coord': 'z', 
                    'element_symbol': 'element', 
                    'atom_name': 'name', 
                    'residue_name': 'resname'})
        
        self.pdb = pdb
        self.residues = max(pdb['residue_number'])
    
    def __iter__(self):
        length = self.residues
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
        # TODO add some sort of ensemble information
        for res_id in res_ids:
            to_remove = (pdb['residue_number'] == res_id) & (~pdb['name'].isin(['N', 'CA', 'O', 'C'])) 
            my_atoms = pdb[~to_remove].reset_index(drop=True)
            label = pdb[(pdb['residue_number'] == res_id) & (pdb['name'] == 'CA')]['resname'].iloc[0]
            ca_idx = np.where((my_atoms['residue_number'] == res_id) & (my_atoms['name'] == 'CA'))[0]
            if len(ca_idx) != 1: 
                continue

            graph = self.graph_builder(my_atoms)
            graph.label = _amino_acids(label)
            graph.ca_idx = int(ca_idx)
            graph.masked_res_id = res_id
            graph.wildtype = self.wildtype
            graph.pdb = pdb

            yield graph

# TODO STEP 3: Pass the amino-acid through the trained model
# TODO STEP 4: Find where the model is most uncertain
# TODO STEP 5: Choose top three mutations in a single spot
def uncertainty_search(trainer, dataloader, locs=1, k=3): 
    trainer.eval()

    top_k = defaultdict(list)
    softmax = Softmax(dim=-1)

    for batch in tqdm(dataloader):
        with torch.no_grad():
            out = trainer.model(batch)
            probs = softmax(out)

            labels = batch.label
            # TODO: we only look at places where it is correct?
            res = torch.argmax(probs, dim=-1)
            conf = torch.max(probs, dim=-1)
            mutation_confs, tops = torch.topk(probs, k, dim=-1)

            for g_idx in range(len(batch)):
                if res[g_idx] == labels[g_idx]:
                    # it is correct, but what is the confidence?
                    confidence = conf[g_idx]
                    graph = batch[g_idx]
                    pq = top_k[graph.wildtype]
                    mutations = tops[g_idx]
                    confs = mutation_confs[g_idx]

                    # pushing negative values so the lowest positive confidences have the highest priority
                    heapq.heappush(pq, Pair(graph.pdb, -confidence, mutations, confs, graph.masked_res_id, graph.label))
                    if len(pq) > locs:
                        # We are interested in top K least cofident predictions
                        heapq.heappop(pq)
    
    with open('mutations.pkl', 'wb') as handle:
        pickle.dump(top_k, handle)

    with open('mutations.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        header = ['wildtype', 'og_confidence', 'mutation_position', 'mutation_codes', 'mutation_confidence']
        # write the header
        writer.writerow(header)
        
        # For every wildtype
        for wildtype, mutations in top_k.items():
            # for every position in the original wildtype
            for pair in mutations:
                # for every top k mutation on that location
                for x, conf in zip(pair.mutations, pair.confs):
                    data = [
                        str(wildtype), 
                        -pair.confidence, 
                        pair.masked_residue, 
                        f'{_3to1(_codes(pair.og_res))}{pair.masked_residue}{_3to1(_codes(x))}',
                        conf,
                    ]
            
                    # write the data  
                    writer.writerow(data)


# TODO STEP 6: recover the sequence 
# TODO STEP 7: check against protein engineering benchmarks (ProteinGym)

DATASET_PATH = '/Users/antoniaboca/ProteinGym_substitutions'

