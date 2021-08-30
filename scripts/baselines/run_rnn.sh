#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-5
#SBATCH --gres gpu:1
#SBATCH -t 02:00:00

python -u scripts/baselines/rnn.py --exp_name rnn --config_path "data/configs/rnn_config.txt" --fold_num ${SLURM_ARRAY_TASK_ID}