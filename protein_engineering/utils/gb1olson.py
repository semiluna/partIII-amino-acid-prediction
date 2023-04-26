import numpy as np
import pathlib
import pandas as pd
import requests

from protein_engineering.archive.de_dataset import DirectedEvolutionDataset
from protein_engineering.archive.mutations import Variant

PROJECT_PATH = pathlib.Path('/Users/antoniaboca/partIII-amino-acid-prediction/data/GB1_double_mutations.csv')

class GB1Olson(DirectedEvolutionDataset):
    _WILDTYPE_SEQ = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    _PAPER_LINK = (
        "https://www.sciencedirect.com/science/article/pii/S0960982214012688?via%3Dihub#mmc2"
    )
    _DATA_LINK = "https://ars.els-cdn.com/content/image/1-s2.0-S0960982214012688-mmc2.xlsx"

    def __init__(self, process: bool = True):
        super().__init__(dataset_path=PROJECT_PATH)
        if not self.dataset_path.exists():
            self.download()
        encoding = 'utf-8'
        self._raw_data = pd.read_csv(
            self.dataset_path, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), encoding=encoding,
        )
        if process:
            self.process()

    def download(self) -> None:
        # Downloading data from _DATA_LINK and storing at self.dataset_path
        request = requests.get(self._DATA_LINK)
        with open(self.dataset_path, "wb") as f:
            f.write(request.content)

    def process(self, evaluate_ranking: bool = True, **kwargs) -> pd.DataFrame:
        df = self._raw_data.copy()
        print(df.columns)
        df["variant1"] = (
            df["Mut1 WT amino acid"] + df["Mut1 Position"].astype(str) + df["Mut1 Mutation"]
        )
        df["variant2"] = (
            df["variant1"]
            + ","
            + df["Mut2 WT amino acid"]
            + df["Mut2 Position"].astype(str)
            + df["Mut2 Mutation"]
        )

        wildtype = pd.DataFrame({"variant": [""], "fitness": [1.0]})
        single_mutants = (
            df[["variant1", "Mut1 Fitness"]]
            .drop_duplicates()
            .rename(columns={"variant1": "variant", "Mut1 Fitness": "fitness"})
            .reset_index(drop=True)
        )
        double_mutants = (
            df[["variant2", "Mut2 Fitness"]]
            .drop_duplicates()
            .rename(columns={"variant2": "variant", "Mut2 Fitness": "fitness"})
            .reset_index(drop=True)
        )

        df = pd.concat([wildtype, single_mutants, double_mutants], axis=0).reset_index(drop=True)

        data = pd.concat([wildtype, single_mutants, double_mutants], axis=0).reset_index(drop=True)
        data["variant"] = data.variant.apply(lambda x: Variant.from_str(x))
        data["sequence"] = data.variant.apply(lambda x: x.get_sequence(self.wildtype_seq))
        data["hamming_to_wildtype"] = data.variant.apply(len)
        data["is_wildtype"] = data.hamming_to_wildtype == 0

        data["fitness_raw"] = data.fitness
        data["fitness"] = np.log(data.fitness)

        if evaluate_ranking:
            data["fitness_percentile"] = data.fitness.rank(pct=True)
            data["fitness_rank"] = data.fitness.rank(method="dense", ascending=False).astype(int)

        self.data = data
        return data

    @property
    def wildtype_seq(self) -> str:
        return self._WILDTYPE_SEQ

    @property
    def sequence_length(self) -> int:
        return len(self.wildtype_seq)