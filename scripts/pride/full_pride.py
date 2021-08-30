import time
import urllib
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
from torch.cuda.amp import autocast, GradScaler

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
train_files_path = project_dir + "/data/pride_input/" + str(args.fold_num) + "/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/pride_input/" + str(args.fold_num) + "/test.txt"
dev_file = project_dir + "/data/pride_input/" + str(args.fold_num) + "/dev.txt"
predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

dims_list = []
for ff in os.listdir(project_dir + "/data/dimensions/"):
    dims_list.append(eval(open(project_dir + "/data/dimensions/" + ff).read()))

p_ages_dict = eval(open(project_dir + "/data/age_difference.txt").read())
ages_dict = dict()
for k, v in p_ages_dict.items():
   ages_dict[k] = v
   ages_dict[k + 1000] = 5 - v

############################################

# Training
num_epochs = 100
max_batch_epoch = 100 // main_config.batch_size
from collections import defaultdict
net = None
hidden_size = 768
dimension_size = 768
age_emb_size = main_config.age_size

#################################################

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Pride(nn.Module):
    def word_attention(self, x):
        attention = self.softmax(self.word_fc(x))
        x = x * attention.expand_as(x)
        return x.sum(dim=0), attention

    def pair_attention(self, x):
        #x = x.T
        attention = self.softmax(self.pair_fc(x))
        x = x * attention.expand_as(x)
        return x.sum(dim=0).squeeze()

    def __init__(self, **config):
        super(Pride, self).__init__()
        config = BertConfig()
        self.bert = BertModel(config).from_pretrained('bert-base-uncased').to(device)
        self.word_fc = torch.nn.Linear(hidden_size, 1).to(device)
        self.pair_fc = torch.nn.Linear(hidden_size, 1).to(device)
        self.classifier = nn.Linear(hidden_size + dimension_size * 11 + age_emb_size, len(predicate_list)).to(device)
        enc_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=16).to(device)
        encoder_norm = LayerNorm(hidden_size).to(device)
        self.interaction = TransformerEncoder(enc_layer, num_layers=12, norm=encoder_norm).to(device)
        self.softmax = torch.nn.Softmax(dim=0)
        self.speaker_enc = nn.Embedding(2, hidden_size).to(device)
        self.pos_enc = PositionalEncoding(hidden_size).to(device)
        self.age_embedding = nn.Embedding(7, age_emb_size).to(device)
        self.cls = torch.Tensor(1, hidden_size).to(device)
        torch.nn.init.normal_(self.cls)
        self.cls.requires_grad = True

#################################################
from pathlib2 import Path
hyperparams = repr(main_config.__dict__)
zero = torch.LongTensor([0]).to(device)

def train():
    print(hyperparams)
    global net
    net = Pride()
    net.train()

    pretrained_dict = torch.load(
        project_dir + "/data/checkpoints/pride/pretrained_" + str(args.fold_num) + ".pkl")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    del pretrained_dict

    criterion = nn.BCEWithLogitsLoss() ###
    bert_params = {'params': net.bert.parameters(), 'lr': main_config.lr}
    fc_params = {'params': list(net.interaction.parameters()), 'lr': main_config.lr}
    fc1_params = {'params': list(net.age_embedding.parameters()) + list(net.speaker_enc.parameters()) + list(net.classifier.parameters()) + list(net.pair_fc.parameters()) + list(net.word_fc.parameters()), 'lr': main_config.cls_lr}
    optimizer = AdamW([bert_params, fc_params, fc1_params])
    streams = [pescador.Streamer(feature_gen_ham, ff, 1, ["input_ids", "label_ids", "segment_ids", "input_mask", "splits", "guid"])
               for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)

    batch_accumulator = BatchAccumulator(mux_stream, batch_size)
    scaler = GradScaler()

    # Train the Model
    for n,p in net.named_parameters():
        if "age" not in n and "classifier" not in n:
            p.requires_grad = False
    for epoch in range(num_epochs):
        if epoch == main_config.unfreeze_epoch:
            for n, p in net.named_parameters():
                p.requires_grad = True
        print("Epoch " + str(epoch))
        for i, features in enumerate(batch_accumulator):
            optimizer.zero_grad()
            answers = to_onehot([f["label_ids"] for f in features], n_categories=len(predicate_list)).to(device)
            all_by_utt = []
            all_segs = []
            with autocast():
                for f in features:
                    # speaker one
                    if len(f["input_ids"]) > 15:
                        f["input_ids"] = f["input_ids"][:15]
                        f["input_mask"] = f["input_mask"][:15]
                        f["segment_ids"] = f["segment_ids"][:15]
                    mask = torch.tensor(f["input_mask"], dtype=torch.long).to(device)
                    samples = torch.tensor(f["input_ids"], dtype=torch.long).to(device)
                    segment = torch.tensor(f["segment_ids"], dtype=torch.long).to(device)
                    splits = [[x[0] for x in l] for l in f["splits"]][:15]
                    dims = []
                    for dimm in dims_list:
                        dims.append(torch.stack([torch.tensor(dimm[f["guid"] % 1000]).to(device) for f in features]))
                    dims = torch.cat(dims, dim=1)
                    ages1 = torch.stack([torch.tensor(ages_dict[f["guid"]]).to(device) for f in features])
                    input_dict = {"input_ids":samples, "attention_mask":mask, "token_type_ids":segment}
                    words_reps = net.bert(**input_dict)[0]
                    if main_config.att_word == "avg":
                        by_utt1 = [torch.stack(
                            [torch.mean(x, dim=0) for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for j
                                   in range(len(splits))]
                    elif main_config.att_word == "max":
                        by_utt1 = [torch.stack(
                            [torch.max(x, dim=0)[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    elif main_config.att_word == "attention":
                        by_utt1 = [torch.stack(
                            [net.word_attention(x)[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    elif main_config.att_word == "cls":
                        by_utt1 = [torch.stack([x[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    by_seg = torch.cat([torch.stack([x[0] for x in torch.split(segment[j], splits[j], dim=0)][1:-1]) for
                                        j in range(len(splits))], dim=0)
                    all_segs.append(by_seg)
                    by_utt1 = torch.cat(by_utt1, dim=0)
                    all_by_utt.append(by_utt1)

                scores = []
                for ab_main, segments in zip(all_by_utt, all_segs):
                    if main_config.att_pair == "cls":
                        ab_main = torch.cat([net.cls, ab_main])
                        segments = torch.cat([zero, segments])
                    res = net.interaction(net.pos_enc(ab_main.unsqueeze(1)) + net.speaker_enc(segments).unsqueeze(1))
                    if main_config.att_pair == "max":
                        res = torch.max(res, dim=0)[0]
                    elif main_config.att_pair == "avg":
                        res = torch.mean(res, dim=0)
                    elif main_config.att_pair == "attention":
                        res = net.pair_attention(res.squeeze(1)).unsqueeze(0)
                    elif main_config.att_pair == "cls":
                        res = res[0]
                    scores.append(res)
                scores = torch.stack(scores).squeeze()
                ages1 = net.age_embedding(ages1)
                scores = torch.cat((scores, ages1, dims), dim=1)

                scores = net.classifier(scores)

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
            print('Epoch %d, %s\n' % (epoch, str(stats)))
            net.to(device)
            net.train()

import string

def validate(epoch):
    global net
    with torch.no_grad():
        net.eval()
        Path(project_dir + "/data/outputs/test/" + str(args.exp_name)).mkdir(exist_ok=True, parents=True)
        Path(project_dir + "/data/outputs/dev/" + str(args.exp_name)).mkdir(exist_ok=True, parents=True)
        test_output_file = project_dir + "/data/outputs/test/" + str(args.exp_name) + "/" + str(args.fold_num) + "_" + str(epoch) + ".txt"
        dev_output_file = project_dir + "/data/outputs/dev/" + str(args.exp_name) + "/" + str(args.fold_num) + "_" + str(epoch) + ".txt"

        for output_file, streaming_file in zip([dev_output_file, test_output_file], [dev_file, test_file]):
            by_guid = defaultdict(list)
            by_guid_answ = dict()
            with open(output_file, "w") as f_test_out:
                streamer = pescador.Streamer(feature_gen_ham, streaming_file, 1,
                                             ["input_ids", "label_ids", "guid", "segment_ids", "input_mask", "splits"])
                for features in streamer:
                    features = features[0]
                    # speaker one
                    mask = torch.tensor(features["input_mask"], dtype=torch.long).to(device)
                    samples = torch.tensor(features["input_ids"], dtype=torch.long).to(device)
                    segment = torch.tensor(features["segment_ids"], dtype=torch.long).to(device)
                    dims = []
                    for dimm in dims_list:
                        dims.append(torch.tensor(dimm[features["guid"] % 1000]).to(device))
                    dims = torch.cat(dims, dim=0)
                    ages1 = torch.tensor(ages_dict[features["guid"]], dtype=torch.long).to(device)
                    splits = [[x[0] for x in l] for l in features["splits"]]
                    input_dict = {"input_ids": samples, "attention_mask": mask, "token_type_ids": segment}
                    words_reps = net.bert(**input_dict)[0]
                    if main_config.att_word == "avg":
                        by_utt1 = [torch.stack(
                            [torch.mean(x, dim=0) for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]])
                                   for j in range(len(splits))]
                    elif main_config.att_word == "max":
                        by_utt1 = [torch.stack(
                            [torch.max(x, dim=0)[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    elif main_config.att_word == "attention":
                        by_utt1 = [torch.stack(
                            [net.word_attention(x)[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    elif main_config.att_word == "cls":
                        by_utt1 = [torch.stack([x[0] for x in torch.split(words_reps[j], splits[j], dim=0)[1:-1]]) for
                                   j in range(len(splits))]
                    by_utt1 = torch.cat(by_utt1, dim=0)

                    segments = torch.cat(
                        [torch.stack([x[0] for x in torch.split(segment[j], splits[j], dim=0)][1:-1]) for
                         j in range(len(splits))], dim=0)
                    if main_config.att_pair == "cls":
                        by_utt1 = torch.cat([net.cls, by_utt1])
                        segments = torch.cat([zero, segments])
                    res = net.interaction(net.pos_enc(by_utt1.unsqueeze(1)) + net.speaker_enc(segments).unsqueeze(1))
                    if main_config.att_pair == "max":
                        res = torch.max(res, dim=0)[0]
                    elif main_config.att_pair == "avg":
                        res = torch.mean(res, dim=0)
                    elif main_config.att_pair == "attention":
                        res = net.pair_attention(res)
                    elif main_config.att_pair == "cls":
                        res = res[0]
                    ages1 = net.age_embedding(ages1).unsqueeze(0)
                    res = torch.cat((res, ages1, dims.unsqueeze(0)), dim=1)
                    one_logit = net.classifier(res).squeeze(0).cpu().data.numpy()

                    by_guid[features["guid"]].append(one_logit)
                    by_guid_answ[features["guid"]] = features["label_ids"]

                for guid, utts in by_guid.items():
                    utts = np.mean(np.array(utts), axis=0)
                    f_test_out.write(str(guid) + "\t" + repr(by_guid_answ[guid]) + '\t' + '\t'.join(
                                [str(y) for y in sorted(enumerate(utts), key=lambda x: x[1], reverse=True)]) + '\n')
        stats = compute_whatever_stats(open(dev_output_file).readlines())
        return stats

train()