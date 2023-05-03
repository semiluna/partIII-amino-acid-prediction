import sys
import numpy as np
import torch
import pandas as pd
import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import biotite.structure as bs

# sys.path.append('/Users/antoniaboca/partIII-amino-acid-prediction')

from protein_engineering.protein_gym import ProteinGymDataset
from protein_engineering.models.ProteinMPNN.protein_mpnn_utils import ProteinMPNN, _scores
from gvp_antonia.utils.biotite_utils import *

MODEL_WEIGHTS = './protein_engineering/models/ProteinMPNN/vanilla_model_weights'
DATA_PATH = './data/ProteinGym_substitutions'
PROTEIN_PATH = './data/ProteinGym_assemblies_clean'
RES_PATH = './data/ProteinGym_ProteinMPNN_scores'

BATCH_SIZE = 4

def get_protein(file):
    # pdb, chain = pdb_and_chain.split(".")
    # pdb = pdb.lower()
    structure = load_structure(file, model=1)
    protein = structure[bs.filter_amino_acids(structure)]
    return protein

def tokenise_sequence(seq):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    return torch.tensor([alphabet.index(aa) for aa in seq], dtype=torch.long)

def extract_backbone(protein):
    return protein[np.isin(protein.atom_name, ["N", "CA", "C", "O"])]

def prepare_for_mpnn(protein, sequences, batch_size):

    assert len(sequences[0]) < 400
    assert len(sequences) == batch_size

    protein = protein[bs.filter_amino_acids(protein)]
    backbone = extract_backbone(protein)
    X = torch.tensor(backbone.coord.reshape(1, -1, 4, 3), dtype=torch.float32).repeat(batch_size, 1, 1, 1)

    chains = bs.get_chains(protein)
    no_chains = len(chains)

    #NOTE we are assuming homo-oligomers or monomers
    S_list = []
    for sequence in sequences:
        seq_chains = [sequence] * no_chains
        S_list.append(
            torch.concat(
                [tokenise_sequence(seq) for seq in seq_chains]
            )
        )

    S = torch.stack(S_list)
    L = S.shape[-1]
    chain_encoding_all = torch.concat(
        [(idx + 1) * torch.ones(len(seq)) for idx, seq in enumerate(seq_chains)]
    ).unsqueeze(0).repeat(batch_size, 1)
    
    jumps = torch.concat(
        [idx * 400 * torch.ones(len(seq)) for idx, seq in enumerate(seq_chains)]
    ).unsqueeze(0).repeat(batch_size, 1)

    residue_idx = torch.arange(L).unsqueeze(0).repeat(batch_size, 1) + jumps

    mask = torch.ones_like(S, dtype=torch.float32)
    chain_M = torch.ones_like(S, dtype=torch.float32)
    randn = torch.randn_like(S, dtype=torch.float32)
    
    return X, S, mask, chain_M, residue_idx, chain_encoding_all, randn


def main(args):

    # ========================== LOAD PROTEINMPNN ==========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #v_48_010=version with 48 edges 0.10A noise
    model_name = "v_48_020" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

    backbone_noise=0.00               # Standard deviation of Gaussian noise to add to backbone atoms

    path_to_model_weights=args.model_path       
    hidden_dim = 128
    num_layers = 3 
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device) 
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")

    # ========================== EVAL PROTEINGYM ==========================
    mapper = pd.read_csv('./data/mapping.csv')

    batch_size = args.batch_size
    data_path = args.data_path
    protein_path = args.structure_path
    res_path = args.out_path

    csv_files = glob.glob(data_path + '/*.csv')
    no_skipped = 0
    skipped = []

    
    for file in csv_files:
        path = Path(file)
        dataset_name = path.stem
        hetero_oligomer = mapper[mapper['name'] == dataset_name]['is_hetero_oligomer'].iloc[0]
        structure_exists = mapper[mapper['name'] == dataset_name]['structure_exists'].iloc[0]
        if hetero_oligomer or (not structure_exists):
            print(f'Skipping {dataset_name}.')
            continue
            
        print(f'Evaluating single-point mutations in {dataset_name}')

        dataset = ProteinGymDataset(path)
        single_mutant_mask = (~dataset.data['variant'].str.contains(','))
        data = dataset.data[single_mutant_mask]
        wildtype = dataset.wildtype_data['sequence'].iloc[0]

        pdbs = mapper[mapper['wildtype'] == wildtype]['identifiers'].iloc[0].split(',')
        structures = list(filter(lambda x: not x.startswith('AF'), pdbs))
        alphafolds = list(filter(lambda x: x.startswith('AF'), pdbs))

        experimental, af = None, None
        if len(structures) > 0:
            structure = structures[0]
            structure_file = protein_path + '/' + structure.upper() + '.pdb'
            experimental = get_protein(structure_file)
        
        if len(alphafolds) > 0:
            alphafold = f'AF-{alphafolds[0][2:-2]}-F1'
            af_file = protein_path + '/' + alphafold.upper() + '.pdb'
            af = get_protein(af_file)    

        if (experimental is None) and (af is None):
            print(f'No structure found for {dataset.name}. Skipping.')
            skipped.append(dataset.name)
            no_skipped += 1
            continue
        
        pdb = None
        if experimental is not None:
            ex_sequence = list(get_sequence(experimental).values())[0]
            if len(ex_sequence) == len(wildtype):
                pdb = experimental
        if (pdb is None) and (af is not None):
            af_sequence = list(get_sequence(af).values())[0]
            if len(af_sequence) == len(wildtype):
                pdb = af
            
        if pdb is None:
            print(f'No structure found for {dataset.name}. Skipping.')
            skipped.append(dataset.name)
            no_skipped += 1
            continue

        all_scores = []
        sequences = np.array(data['sequence']) 
        try:
            for idx in tqdm(range(0, len(sequences), batch_size)):
                batch = sequences[idx:(idx + batch_size)]
                # sequence = row['sequence']
                X, S, mask, chain_M, residue_idx, chain_encoding_all, randn = prepare_for_mpnn(pdb, batch, batch_size)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
                scores = _scores(S, log_probs, torch.ones_like(S))
                native_score = scores.cpu().data.numpy()
                all_scores.append(native_score)

            data['ProteinMPNN_score'] = np.concatenate(all_scores)

            data.to_csv(f'{res_path}/{dataset.name}.csv', index=False)
            print(f'Written all ProteinMPNN scores for {dataset.name}.')

        except Exception as error:
            print(f'Failed ProteinMPNN evaluation on {dataset.name}. Error message: \n{error}')

    print(f'Failed {no_skipped} datasets. Skipped datasets:\n {skipped}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--model_path', default=MODEL_WEIGHTS, type=str)
    parser.add_argument('--data_path', default=DATA_PATH, type=str)
    parser.add_argument('--structure_path', default=PROTEIN_PATH, type=str)
    parser.add_argument('--out_path', default=RES_PATH, type=str)

    args = parser.parse_args()

    main(args)
