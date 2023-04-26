import pathlib
import requests
import json
import os

import pandas as pd
from biopandas.pdb import PandasPdb
from typing import Union
from pathlib import Path

from protein_engineering.utils.de_dataset import DirectedEvolutionDataset
from protein_engineering.utils.mutations import Variant

# TODO: Remove hardcoding
PROTEIN_GYM_PATH = Path('/Users/antoniaboca/partIII-amino-acid-prediction/data/ProteinGym_substitutions')
PROTEIN_GYM_DATASETS = list(PROTEIN_GYM_PATH.glob("*.csv"))
DOWNLOAD_PATH = Path('/Users/antoniaboca/partIII-amino-acid-prediction/data/ProteinGym_structures')

class ProteinGymDataset(DirectedEvolutionDataset):
    def __init__(self, 
                dataset_path: Union[str, pathlib.Path], 
                process: bool = True, 
                structure_path : Union[str, pathlib.Path] = DOWNLOAD_PATH):

        super().__init__(dataset_path)
        self.name = dataset_path.stem
        if process:
            self.process()
        self.experimental = []
        self.alpha_fold = []
        self.downloaded = False
        self.structure_path = structure_path

    @staticmethod
    def infer_wt_data(data: pd.DataFrame, fitness_is_log_transformed: bool):
        wt_seq = Variant.from_str(data.iloc[0].variant).invert().get_sequence(data.iloc[0].sequence)
        fitness = 0.0 if fitness_is_log_transformed else 1.0
        wt_data = {
            "variant": [""],
            "fitness": [0.0],
            "sequence": [wt_seq],
            "DMS_score_bin": [1],
            "fitness": [fitness],
        }
        return pd.DataFrame.from_dict(wt_data)

    def process(self, evaluate_ranking: bool = True, **kwargs) -> pd.DataFrame:
        data = pd.read_csv(self.dataset_path)
        data = data.rename(
            columns={
                "mutated_sequence": "sequence",
                "DMS_score": "fitness",
                "mutant": "variant",
            }
        )
        data.variant = data.variant.str.replace(":", ",")

        # Sanity check whether it is conceivable that DMS_score corresponds to log fitness
        fitness_is_log_transformed: bool = data.fitness.min() < 0.0
        if not fitness_is_log_transformed:
            print("It seems the fitness data is not log transformed.")

        # Check if wildtype data is contained
        wildtype_exists = data.variant.str.len().min() <= 2
        if wildtype_exists:
            # Set wildtype `variant` data to empty string
            data.loc[data.hamming_to_wildtype == 0, "variant"] = ""
        else:
            print("No wildtype data found. Inferring wildtype data.")
            data = pd.concat([self.infer_wt_data(data, fitness_is_log_transformed), data])

        # Infer Hamming distance (WARNING: Currently only supports substitutions)
        data["hamming_to_wildtype"] = data.variant.apply(lambda x: len(Variant.from_str(x)))

        # Infer wildtype identity
        data["is_wildtype"] = data.hamming_to_wildtype == 0

        if evaluate_ranking:
            data["fitness_percentile"] = data.fitness.rank(pct=True)
            data["fitness_rank"] = data.fitness.rank(method="dense", ascending=False).astype(int)

        self.data = data
        return data
    
    def download_structure(self):
        if self.downloaded:
            return

        base_url = 'https://search.rcsb.org/rcsbsearch/v2/query'

        query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
            "value": self.data.iloc[0]['sequence'],
            "evalue_cutoff": 1,
            "identity_cutoff": 0.9,
            "sequence_type": "protein",
            }
        },
        "return_type": "entry",
        "request_options": {
            "scoring_strategy": "sequence",
            "results_content_type": [
            "computational",
            "experimental"
            ]
        }
        }

        # Convert the query dictionary to JSON
        json_query = json.dumps(query)

        # Construct the URL-encoded query string
        url = f"{base_url}?json={json_query}"
        response = requests.get(url)
        if response.status_code == 200:
            # Extract the response JSON
            response_json = response.json()
            # Process the response as needed
            print(response_json)
        else:
            print("Request failed with status code:", response.status_code)
        
        experimental = []
        alpha_fold = []
        for result in response_json['result_set']:
            if result['score'] < 1.0:
                continue
            
            id = result['identifier']
            if id.startswith('AF_'):
                alpha_fold.append(id[3:])
            else:
                experimental.append(id)
        

        if len(alpha_fold) > 1:
            print('WARNING: multiple predicted structures found.')
        if len(alpha_fold) == 0:
            print('WARNING: no Alpha fold structure found.')
        if len(experimental) > 1:
            print('WARNING: multiple experimental structures found.')
        
        base_rcsb = 'https://files.rcsb.org/download/'
        for pdb_code in experimental:
            pdb_url = f'{base_rcsb}{pdb_code}.pdb1'
            download_file = f'{self.structure_path}/{pdb_code}.pdb'

            os.system(f'curl {pdb_url} -o {download_file}')
            self.experimental.append(PandasPdb().read_pdb(download_file))
        
        for af in alpha_fold:
            alphafold_ID = f'AF-{af[2:8]}-F1'
            database_version = 'v4'
            model_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_ID}-model_{database_version}.pdb'
            download_file = f'{self.structure_path}/{alphafold_ID}.pdb'
            os.system(f'curl {model_url} -o {download_file}')
            self.alpha_fold.append(PandasPdb().read_pdb(download_file))
        
        self.downloaded = True
        
        return experimental, alpha_fold
