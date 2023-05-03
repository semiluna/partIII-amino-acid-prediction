import copy
from typing import Optional, List
from pathlib import Path
import pandas as pd

from protein_engineering.utils.de_dataset import DirectedEvolutionDataset
from protein_engineering.utils.mutations import parse_variant as Parser

STANDARD_AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")

class IRED(DirectedEvolutionDataset):
    def __init__(
        self, dataset_path=Path('/Users/antoniaboca/partIII-amino-acid-prediction/data/srired_v2022_12_02.csv'), process: bool = True
    ):
        super().__init__(dataset_path)
        self._raw_data = None
        self._aa_seqs = None
        self._fitness = None
        self.common_length: bool = True
        self.data = None
        if process:
            self.process()

    @property
    def raw_data(self):
        if self._raw_data is None:
            raw_data = pd.read_csv(self.dataset_path, index_col=0)
            raw_data["aa_len"] = raw_data["aa_seq"].str.len()
            self._raw_data = raw_data
        return self._raw_data

    @staticmethod
    def _process_raw_data(
        raw_data: pd.DataFrame,
        target_seq_len: Optional[int] = 290,
        max_aa_mutations: Optional[int] = 15,
        parse_variant: bool = True,
        evaluate_ranking: bool = True,
    ) -> pd.DataFrame:
        data = copy.deepcopy(raw_data)

        seq_is_na = data.aa_seq.isna()
        print("Remove %d NaN sequences" % seq_is_na.sum())
        data = data[~seq_is_na]

        fitness_is_na = data.fitness.isna()
        print("Remove %d NaN fitness values" % fitness_is_na.sum())
        data = data[~fitness_is_na]

        ends_in_stop = data.aa_seq.str.endswith("*")
        print("Remove %d sequences not ending in stop" % (~ends_in_stop).sum())
        data = data[ends_in_stop]

        # Remove stop codon at end of sequence
        data.aa_seq = data.aa_seq.str.replace("\*$", "", regex=True)
        has_intermediate_stop = data.aa_seq.str.contains("*", regex=False)
        print("Remove %d sequences with intermediate stop" % has_intermediate_stop.sum())
        data = data[~has_intermediate_stop]

        # NOTE: Experiment uses a linker such that non-methionine start codons are possible
        # starts_with_methoinine = data.aa_seq.str.startswith("M")
        # _l("Remove %d sequences not starting with methionine" % (~starts_with_methoinine).sum())
        # data = data[starts_with_methoinine]

        has_non_standard_aa = data.aa_seq.str.contains(
            f"[^{''.join(STANDARD_AMINO_ACIDS)}]", regex=True
        )
        print("Remove %d sequences with non-standard AAs" % has_non_standard_aa.sum())
        data = data[~has_non_standard_aa]

        if target_seq_len is not None:
            seq_len = data.aa_seq.str.len()
            print(
                "Remove %d sequences with length != %d"
                % ((seq_len != target_seq_len).sum(), target_seq_len)
            )
            data = data[seq_len == target_seq_len]
        data["aa_len"] = data.aa_seq.str.len()

        if max_aa_mutations is not None:
            mut_count = data.Nham_aa
            print(
                "Remove %d sequences with >%d mutations"
                % ((mut_count > max_aa_mutations).sum(), max_aa_mutations)
            )
            data = data[mut_count <= max_aa_mutations]

        # Drop columns we don't need and rename
        data["is_wildtype"] = data["Nham_aa"] == 0
        data = data.rename({"aa_seq": "sequence", "Nham_aa": "hamming_to_wildtype"}, axis=1)

        assert data.is_wildtype.sum() == 1, "Found more than one wildtype sequence"

        if parse_variant:
            print("Parsing variants from sequences")
            wildtype_seq = data[data.is_wildtype == True].sequence.iloc[0]
            data["variant"] = data.sequence.apply(
                lambda x: str(Parser(x, wildtype_seq))
            )

        if evaluate_ranking:
            print("Evaluating ranking")
            data["fitness_percentile"] = data.fitness.rank(pct=True)
            data["fitness_rank"] = data.fitness.rank(method="dense", ascending=False).astype(int)

        return data

    def process(
        self,
        target_seq_len: int = 290,
        max_aa_mutations: int = 15,
        parse_variant: bool = True,
        evaluate_ranking: bool = True,
        **kwargs,
    ) -> None:
        self.data = self._process_raw_data(
            self.raw_data,
            target_seq_len=target_seq_len,
            max_aa_mutations=max_aa_mutations,
            parse_variant=parse_variant,
            evaluate_ranking=evaluate_ranking,
        )
        assert self._validate_catalytic_site(self.wildtype_seq)

    @property
    def catalytic_residues(self) -> List[int]:
        # Note: This is 0-indexed
        return [94 - 1, 170 - 1, 178 - 1]  # Should be [S, D, W] for SrIRED in python convention

    def _validate_catalytic_site(self, seq: str) -> bool:
        return "".join([seq[p] for p in self.catalytic_residues]) == "SDW"

    @property
    def sequence_length(self) -> int:
        return len(self.wildtype_seq)  # should be 290