# PRIDE


Requirements: Python 3.6.4 and libs from scripts/requirements.txt

All commands below are run from the project directory

You will have to run the scrips on the slurm cluster
https://slurm.schedmd.com/quickstart.html

## 0 Download data
Download data.zip from https://www.dropbox.com/s/50fkwv1ex4moyu2/data.rar?dl=0 and extract into the project directory.
More details on data are in data folder readme.

## 1 Create data
`bash scripts/run_data.sh`

That creates the split of the data into 5 folds (the best split is precomputed according to the sampling algorithm described in the paper). All folds in the split contain disjoint set of movies and are balanced to have almost equal numbers of instances per each label.
Based on the fold split the input datasets for all models are created.

## 2 Run baselines
Run RNN baseline from [Welch et al.](https://arxiv.org/abs/1904.11610)

`sbatch scripts/baselines/run_rnn.sh`

Run HAM baseline from [Tigunova et al.](https://dl.acm.org/doi/10.1145/3308558.3313498)

`sbatch scripts/baselines/run_ham.sh`

Run BERT_conv baseline from [Lu et al.](https://dl.acm.org/doi/10.1145/3397271.3401255)

`sbatch scripts/baselines/run_bert_conv.sh`

Run BERT_ddrel baseline from [Jia et al.](https://arxiv.org/abs/2012.02553)

`sbatch scripts/baselines/run_bert_ddrel.sh`

Running these scripts will create the raw output prediction files.

## 3 Create relationship dimensions' representations
The used relationship dimensions are described in [Rashid et al.](https://aclanthology.org/D18-1470.pdf). We use the data provided by the authors of the paper to train BERT classifiers for each of the 11 interpersonal dimensions.

`sbatch scripts/dimensions/run_training.sh`

Wait for it to finish and create checkpoints in data/checkpoints/dimensions. Then the trained classifiers are used to create interpersonal dimensions' representations for the utterances in our input data, which will later be used in PRIDE.

`sbatch scripts/dimensions/run_inference.sh`

## 4 Run PRIDE
First pretrain the PRIDE model without any external representations added (no ages or dimensions representations.
`sbatch scripts/pride/run_pretrain.sh`

Wait for it to finish and create checkpoints in data/checkpoints/pride. Finally train the whole PRIDE model.

`sbatch scripts/pride/run_pride.sh`

This will create the raw output files.

## 5 Evaluate results
Run the script which calculates statistics (F1, precision, recall) on all models.
`python scripts/gather_results.py`

