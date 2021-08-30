#!/bin/bash
#SBATCH -p gpu20
#SBATCH -a 1-14
#SBATCH --gres gpu:1
#SBATCH -t 00:59:00

AR=('xy' 'cooperative vs. noncooperative (interactions)' 'cooperative vs. noncooperative (relationships)' 'equal vs. hierarchical' 'intense vs. superficial' 'pleasure vs. work oriented' 'active vs. passive (interactions)' 'active vs. passive (relationships)' 'intimate vs.unintimate' 'temporary vs. long term' 'concurrent vs. non concurrent' 'near vs. distant')

python -u /home/tigunova/PycharmProjects/pride_code/scripts/dimensions/dimension_train.py --dim_name "${AR[${SLURM_ARRAY_TASK_ID}]}"
