import time
import urllib

from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
import sys
sys.path.insert(0, project_dir + '/scripts')

import torch
import torch.nn as nn
import numpy as np
import pescador
import os
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
from transformers import BertForSequenceClassification, BertConfig, AdamW, BertModel
import random
import torch.nn.functional as F
random.seed(time.time())
from fast_utils import *
import argparse
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report


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
train_file = project_dir + "/data/dimensions_training/" + args.dim_name + "/train.txt"
test_file = project_dir + "/data/dimensions_training/" + args.dim_name + "/test.txt"
dev_file = project_dir + "/data/dimensions_training/" + args.dim_name + "/dev.txt"
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

############################################

# Training
num_epochs = 30
max_batch_epoch = 100 // 4
from collections import defaultdict
batch_size = 4
net = None
hidden_size = 768

config = BertConfig()
net = BertForSequenceClassification(config).from_pretrained('bert-base-uncased', num_labels=3).to(device)

Path(project_dir + "/data/checkpoints/dimensions").mkdir(exist_ok=True, parents=True)
#################################################
def train():
    global net
    net.train()

    criterion = nn.CrossEntropyLoss() ###
    bert_params = {'params': list(net.parameters()), 'lr': 2e-05}
    optimizer = AdamW([bert_params])

    streams = pescador.Streamer(feature_gen_ham, train_file, batch_size, ["input_ids", "label_ids", "segment_ids", "input_mask"])
    scaler = GradScaler()

    best_stats_by_mrr = None
    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        for i, features in enumerate(streams):
            optimizer.zero_grad()
            answers = torch.LongTensor([int(f["label_ids"]) + 1 for f in features]).to(device)
            with autocast():
                samples = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features]).to(device)
                segment = torch.stack([torch.tensor(f["segment_ids"], dtype=torch.long) for f in features]).to(device)
                mask = torch.stack([torch.tensor(f["input_mask"], dtype=torch.long).to(device) for f in features]).to(device)
                input_dict = {"input_ids": samples, "attention_mask": mask, "token_type_ids": segment}
                scores = net(**input_dict)[0]
            loss = criterion(scores, answers)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 20 == 0:
                print('Epoch [%d/%d], Batch [%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, loss.item()))
            if i // batch_size > max_batch_epoch:
                break

        if (epoch + 1) % 1 == 0:
            stats = validate(epoch)
            stats["epoch"] = epoch
            stats = stats["weighted avg"]
            if best_stats_by_mrr != None:
                best_stats_by_mrr = max([best_stats_by_mrr, stats], key=lambda x:x["f1-score"])
            else:
                best_stats_by_mrr = stats
            if best_stats_by_mrr == stats:
                torch.save(net.state_dict(),
                           project_dir + "/data/checkpoints/dimensions/" + dim_name + ".pkl")
            net.to(device)
            net.train()

def validate(epoch):
    global net
    with torch.no_grad():
        net.eval()
        predicted = []
        true = []
        streamer = pescador.Streamer(feature_gen_ham, dev_file, 1, ["input_ids", "label_ids", "segment_ids", "input_mask"])
        for features in streamer:
            features = features[0]
            mask = torch.tensor(features["input_mask"], dtype=torch.long).to(device)
            samples = torch.tensor(features["input_ids"], dtype=torch.long).to(device)
            segment = torch.tensor(features["segment_ids"], dtype=torch.long).to(device)
            input_dict = {"input_ids": samples.unsqueeze(0), "attention_mask": mask.unsqueeze(0), "token_type_ids": segment.unsqueeze(0)}
            words_reps = net(**input_dict)[0]
            predicted.append(torch.max(words_reps, dim=1)[1][0].item())
            true.append(int(features["label_ids"]) + 1)

        return classification_report(true, predicted, output_dict=True)

train()