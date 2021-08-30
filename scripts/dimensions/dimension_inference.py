import time
import urllib

from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
import sys
sys.path.insert(0, project_dir + '/scripts')

import torch
import numpy as np
import pescador
import os
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
from transformers import BertForSequenceClassification, BertConfig
import random
random.seed(time.time())
from fast_utils import *
import argparse

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)
np.set_printoptions(threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--dim_name", type=str, default=None)
args = parser.parse_args()

def to_onehot(labels, n_categories, dtype=torch.float32):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
    for i, label in enumerate(labels):
        label = torch.LongTensor(int(label) + 1)
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
    return one_hot_labels

##########################  PARAMETERS  ##################################

# Input files
dim_name = args.dim_name
test_file = project_dir + "/data/dimensions_training/inference.txt"
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

##################################  Train and eval  ######################

from collections import defaultdict
bert_size = 512
net = None
hidden_size = 768

#################################################

config = BertConfig()
net = BertForSequenceClassification(config).from_pretrained('bert-base-uncased', num_labels=3).to(device)
pretrained_dict = torch.load(project_dir + "/data/checkpoints/dimensions/" + dim_name + ".pkl")
net.load_state_dict(pretrained_dict)
del pretrained_dict

#################################################

def validate():
    global net
    results = defaultdict(list)
    writing_dict = dict()
    with torch.no_grad():
        net.eval()
        streamer = pescador.Streamer(feature_gen_ham, test_file, 1, ["input_ids", "guid", "segment_ids", "input_mask"])
        for features in streamer:
            features = features[0]
            mask = torch.tensor(features["input_mask"], dtype=torch.long).to(device)
            samples = torch.tensor(features["input_ids"], dtype=torch.long).to(device)
            segment = torch.tensor(features["segment_ids"], dtype=torch.long).to(device)
            input_dict = {"input_ids": samples.unsqueeze(0), "attention_mask": mask.unsqueeze(0), "token_type_ids": segment.unsqueeze(0)}
            words_reps = net.bert(**input_dict)[1]
            results[features["guid"]].append(words_reps)
        for idd, reses in dict(results).items():
            reses = torch.max(torch.stack(reses, dim=0), dim=0)[0].cpu().data.numpy()[0].tolist()
            writing_dict[idd] = reses
    Path(project_dir + "/data/dimensions/").mkdir(exist_ok=True, parents=True)
    open(project_dir + "/data/dimensions/"+ dim_name + ".txt", "w").write(repr(writing_dict))

validate()