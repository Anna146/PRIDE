# PRIDE

Requirements: Python 3.6.4 and libs from scripts/requirements.txt

All commands below are run from the project directory

You will have to run the scrips on the slurm cluster
https://slurm.schedmd.com/quickstart.html

## 0 Download data
Download data.zip from https://www.dropbox.com/s/50fkwv1ex4moyu2/data.rar?dl=0
and extract into the project directory

## 1 Create data
`bash scripts/run_data.sh`

## 2 Run baselines
`sbatch scripts/baselines/run_rnn.sh`

`sbatch scripts/baselines/run_ham.sh`

`sbatch scripts/baselines/run_bert_conv.sh`

`sbatch scripts/baselines/run_bert_ddrel.sh`

## 3 Create relationship dimensions' representations
`sbatch scripts/dimensions/run_training.sh`

Wait for it to finish and create checkpoints in data/checkpoints/dimensions

`sbatch scripts/dimensions/run_inference.sh`

## 4 Run PRIDE
`sbatch scripts/pride/run_pretrain.sh`

Wait for it to finish and create checkpoints in data/checkpoints/pride

`sbatch scripts/pride/run_pride.sh`

## 5 Evaluate results
`python scripts/gather_results.py`

