# Missing Indicator Method
Code to reproduce results from paper "The Missing Indicator Method: From Low to High Dimensions"

## Environment

To set up environment, run `pip install -r requirements.txt`. To use in conda, first innstall pip in conda using `conda install pip` then using the path to the conda environment run `/path/to/conda/env/bin/pip install -r requirements.txt`.

## Run Scripts

Use the following script to reproduce the corrresponding results:

- Figure 2: `sim_low_dim_reg.py`
- Figure 3: `sim_high_dim.py`
- Figure 4 and 5: `openml_experiment.py`
- Figure 6: `mimic_mortality.py`, `mimic_phenotype.py`, `mimic_los.py`
- Figure 7: `openml_real_missing.py`

## MIMIC data

The data for the mortality task can be found in `mimic_data/`. The data files for length-of-stay and phenotyping tasks are too big to fit on Github. To recreate these files, please follow the following instructions.

1) Download the raw MIMIC-III data csvs from physionet, and store them in some directory. You will need to gain access to the raw data by completing the required training if not done already.

2) Follow the steps in the mimic3-benchmark README, which can be found in `mimic_benchmark_scripts/mimic3-benchmarks`, through the "Train / validation split" step. This will generate train, validation, and test data for the 4 clinical tasks, but in raw panel data form. Note that this whole process will take several hours to get through all the steps.

3) Use the scripts named `mimic/make_processed_{taskname}_data.py` to generate the tabular datasets from the raw time series data for each of the 3 tasks. Run each of these scripts to generate the corresponding csv files for each task that will populate the processed_data directory in `mimic_data/`.

