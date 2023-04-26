"""
Code for generating aaindex embeddings from the AAIndex database, taken from
https://github.com/gitter-lab/nn4dms/blob/master/code/parse_aaindex.py
"""


import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STANDARD_AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")
DATA_PATH = '/Users/antoniaboca/partIII-amino-acid-prediction/protein_engineering/utils'
AA_INDEX_RAW_PATH = f'{DATA_PATH}/aaindex.txt'

def _parse_raw_aaindex_data(aa_index_path=AA_INDEX_RAW_PATH):
    """load and parse the raw aa index data"""
    # read the aa index file
    with open(aa_index_path) as f:
        lines = f.readlines()

    # set up an empty dataframe (will append to it)
    data = pd.DataFrame(
        [],
        columns=[
            "accession number",
            "description",
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ],
    )

    # the order of amino acids in the aaindex file
    line_1_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I"]
    line_2_order = ["L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    all_entries = []
    current_entry = {}
    reading_aa_props = 0
    for line in lines:
        if line.startswith("//"):
            all_entries.append(current_entry)
            current_entry = {}
        elif line.startswith("H"):
            current_entry.update({"accession number": line.split()[1]})
        elif line.startswith("D"):
            current_entry.update({"description": " ".join(line.split()[1:])})
        elif line.startswith("I"):
            reading_aa_props = 1
        elif reading_aa_props == 1:
            current_entry.update(
                {k: v if v != "NA" else 0 for k, v in zip(line_1_order, line.split())}
            )
            reading_aa_props = 2
        elif reading_aa_props == 2:
            current_entry.update(
                {k: v if v != "NA" else 0 for k, v in zip(line_2_order, line.split())}
            )
            reading_aa_props = 0

    data = data.append(all_entries)
    return data


def _perform_pca(features, n_components: int = 19, seed: int = 7):
    np.random.seed(seed)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    print("Captured variance: %s" % sum(pca.explained_variance_ratio_))
    return principal_components


def generate_aaindex_pca_embeddings(out_dir=DATA_PATH, n_components: int = 19, seed: int = 7):
    save_path = Path(f"{out_dir}/aa_index_pca{n_components}.npy")
    if save_path.exists():
        print(f"PCA already exists at {save_path}")
        return np.load(save_path)

    # parse raw aa_index data
    data = _parse_raw_aaindex_data()

    # standardize each aa feature onto unit scale
    aa_features = data.loc[:, STANDARD_AMINO_ACIDS].values.astype(np.float32)
    # for standardization and PCA, we need it in [n_samples, n_features] format
    aa_features = aa_features.transpose()  # [20, n_feat]
    # standardize
    aa_features = StandardScaler().fit_transform(aa_features)

    # pca
    pcs = _perform_pca(
        aa_features,
        n_components=n_components,
    )
    np.save(save_path, pcs)

    return pcs


def main():
    generate_aaindex_pca_embeddings()


if __name__ == "__main__":
    main()