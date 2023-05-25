# Structure-informed protein engineering with equivariant graph neural networks

Before running any code in this repository, install the libraries present in the `requirements.txt` file in a conda environment. This project uses `python3.8`.

## Phase 1: pre-training on the RES task
To train the EQGAT or the GVP on the RES task, run the following command:
```
python3 ./res_task/main.py --model [specify 'gvp' or 'eqgat'] --data_file /path/to/RES_task/data
```
## Phase 2: Mutation generation
For the mutation generation pipeline, I provide a mapping from each wildtype sequence to the experimental structure + the AlphaFold predicted structure I found in the Protein Data Bank. This mapping can be found in `./data/mapping.csv`. 
Ensure all structures are downloaded, cleaned and made available in the `./data/ProteinGym_assemblies_clean/` before proceeding. 
Ensure you have downloaded the **ProteinGym substitutions dataset** in the directory `./data/ProteinGym_substitutions/` before proceeding.

To generate mutations for all 49 DMS assays used in this project, run the following command:
```
python3 mutations_eval.py --mapper './data/mapping.csv' --model_path 'path/to/trained/EGNN/model' --model [specify 'gvp' or 'eqgat']  --out_dir './data'
```
If you want to discard mutations at wrongly predicted positions, use the additional flag `--correct_only`. If you want to use only AlphaFold predicted structures, use the flag `--AF_only`. 

## Phase 3: Augmented linear models
To generate all results of the ridge regression models, run the following command:
```
python3 ridge_regression.py --mapper './data/mapping.csv' --model_path '/path/to/trained/EGNN/model' --model [choose from 'gvp', 'eqgat', 'tranception'] --add_score --embeddings_type [choose from 'aa_index' or 'one_hot'] --out_dir './data'
```
If you want to run the basic linear regression model, remove `--model_path`, `--model` and `--add_score`. 

