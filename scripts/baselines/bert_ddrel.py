import time
import urllib

t1 = time.time()
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

import torch
import torch.nn as nn
import numpy as np
import pescador
import os
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
from transformers import BertForSequenceClassification, BertConfig, AdamW, BertModel
import random
random.seed(time.time())
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
import argparse
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder

device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
torch.manual_seed(33)
torch.set_printoptions(precision=6, threshold=100000)
np.set_printoptions(threshold=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=None)
parser.add_argument("--fold_num", type=str, default=None)
parser.add_argument("--exp_name", type=str, default=None)
args = parser.parse_args()
args.fold_num = int(args.fold_num) - 1

class MainConfig():
    def __init__(self):
        config_dict = eval(open(args.config_path).read())
        self.__dict__ = config_dict

main_config = MainConfig()
batch_size = main_config.batch_size

def to_onehot(labels, n_categories, dtype=torch.float32):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
    for i, label in enumerate(labels):
        label = torch.LongTensor(label)
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
    return one_hot_labels

##########################  PARAMETERS  ##################################

# Input files
train_files_path = project_dir + "/data/bert_ddrel_input/" + str(args.fold_num) + "/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/bert_ddrel_input/" + str(args.fold_num) + "/test.txt"
dev_file = project_dir + "/data/bert_ddrel_input/" + str(args.fold_num) + "/dev.txt"
predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

############################################

# Training
num_epochs = 100
max_batch_epoch = 100 // main_config.batch_size

##################################  Train and eval  ######################

from collections import defaultdict
bert_size = 512
net = None
hidden_size = 768
hyperparams = repr(main_config.__dict__)

config_num = args.config_path.split("/")[-1].strip(".txt")

def train():
    print(hyperparams)
    global net
    config = BertConfig()
    net = BertForSequenceClassification(config).from_pretrained('bert-base-cased', num_labels=len(predicate_list)).to(device)
    net.train()

    criterion = nn.BCEWithLogitsLoss() ###
    optimizer = AdamW(net.parameters(), lr=main_config.lr)  # To reproduce BertAdam specific behavior set correct_bias=False
    streams = [pescador.Streamer(feature_gen_ham, ff, 1, ["input_ids", "label_ids", "segment_ids", "input_mask", "guid"])
               for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)  # .StochasticMux(streams, n_active=len(streams), rate=0.5, mode="with_replacement") #
    batch_accumulator = BatchAccumulator(mux_stream, batch_size)

    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        for i, features in enumerate(batch_accumulator):
            optimizer.zero_grad()
            mask = torch.tensor([f["input_mask"] for f in features], dtype=torch.long).to(device)
            inputs = np.array([f["input_ids"] for f in features])
            samples = torch.tensor(inputs, dtype=torch.long).to(device)
            answers = to_onehot([f["label_ids"] for f in features], n_categories=len(predicate_list)).to(device)
            segment = torch.tensor([f["segment_ids"] for f in features], dtype=torch.long).to(device)

            outputs = net.bert(
                input_ids=samples,
                attention_mask=mask,
                token_type_ids=segment
            )
            pooled_output = outputs[1]
            pooled_output = net.dropout(pooled_output)
            logits = net.classifier(pooled_output)

            loss = criterion(logits, answers)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [%d/%d], Batch [%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, loss.item()))
            if i // batch_size > max_batch_epoch:
                break
        if (epoch + 1) % 1 == 0:
            stats = validate(epoch)
            print('Epoch %d, %s\n' % (epoch, str(stats)))
            net.to(device)
            net.train()


def validate(epoch):
    global net
    with torch.no_grad():
        net.eval()
        # files
        Path(project_dir + "/data/outputs/test/" + str(args.exp_name)).mkdir(exist_ok=True, parents=True)
        Path(project_dir + "/data/outputs/dev/" + str(args.exp_name)).mkdir(exist_ok=True, parents=True)
        test_output_file = project_dir + "/data/outputs/test/" + str(args.exp_name) + "/" + str(
            args.fold_num) + "_" + str(epoch) + ".txt"
        dev_output_file = project_dir + "/data/outputs/dev/" + str(args.exp_name) + "/" + str(
            args.fold_num) + "_" + str(epoch) + ".txt"

        for output_file, streaming_file in zip([dev_output_file, test_output_file], [dev_file, test_file]):
            by_guid = defaultdict(list)
            by_guid_answ = dict()
            with open(output_file, "w") as f_test_out:
                streamer = pescador.Streamer(feature_gen_ham, streaming_file, 1,
                                             ["input_ids", "label_ids", "guid", "segment_ids", "input_mask"])
                for features in streamer:
                    features = features[0]
                    curr_mask = torch.tensor(features["input_mask"], dtype=torch.long).to(device)
                    curr_words = torch.tensor(features["input_ids"], dtype=torch.long).to(device)
                    curr_segments = torch.tensor(features["segment_ids"], dtype=torch.long).to(device)

                    one_logit = net(curr_words.unsqueeze(0), token_type_ids=curr_segments.unsqueeze(0), attention_mask=curr_mask.unsqueeze(0))[0]

                    for log in one_logit.cpu().data.numpy():
                        res = dict((v, 1.0 / (k+1)) for k, v in enumerate([y[0] for y in sorted(enumerate(log), key=lambda x: x[1], reverse=True)]))
                        by_guid[features["guid"]].append(res)
                        by_guid_answ[features["guid"]] = features["label_ids"]

                for guid, utts in by_guid.items():
                    cls_dict = dict()
                    for utt in utts:
                        for k, v in utt.items():
                            cls_dict[k] = cls_dict.get(k, 0) + v
                    utts = [y for y in sorted(list(cls_dict.items()), key=lambda x: x[1], reverse=True)]
                    utts = [(utts[0][0], 1)] + [(x[0], -1) for x in utts[1:]]
                    f_test_out.write(str(guid) + "\t" + repr(by_guid_answ[guid]) + '\t' + '\t'.join([str(x) for x in utts]) + '\n')

    stats = compute_whatever_stats(open(dev_output_file).readlines())
    return stats

train()