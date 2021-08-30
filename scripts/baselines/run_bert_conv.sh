#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-5
#SBATCH --gres gpu:1
#SBATCH -t 01:00:00

python -u scripts/baselines/bert_conv.py --exp_name bert_conv --config_path "data/configs/bert_conv_config.txt" --fold_num ${SLURM_ARRAY_TASK_ID}