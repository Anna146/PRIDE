#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-5
#SBATCH --gres gpu:1
#SBATCH -t 01:00:00

python -u scripts/baselines/bert_ddrel.py --exp_name bert_ddrel --config_path "data/configs/bert_ddrel_config.txt" --fold_num ${SLURM_ARRAY_TASK_ID}
