#!/bin/bash

# create data split
python scripts/prepare_data/create_speaker_whitelists.py

# input for pride
python scripts/prepare_data/ages_to_buckets.py
python scripts/prepare_data/dimension_train_input.py
python scripts/prepare_data/dimension_inference_input.py
python scripts/prepare_data/input_pride.py

# input for baselines
python scripts/prepare_data/input_ham.py
python scripts/prepare_data/input_bert_conv.py
python scripts/prepare_data/input_bert_ddrel.py
python scripts/prepare_data/input_rnn.py