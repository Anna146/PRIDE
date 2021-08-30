#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-5
#SBATCH --gres gpu:1
#SBATCH -t 01:00:00

python -u scripts/pride/pretrain.py --exp_name tmp --config_path "data/configs/pride_config.txt" --fold_num ${SLURM_ARRAY_TASK_ID}
