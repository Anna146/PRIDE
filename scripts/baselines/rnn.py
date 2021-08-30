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
import random
random.seed(time.time())
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
import argparse

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
config_num = args.config_path.split("/")[-1].strip(".txt")

def hot_label(labels, n_categories, dtype=torch.long):
    one_hot_labels = torch.ones(size=(n_categories, len(labels)), dtype=dtype)
    for lab in range(len(predicate_list)):
        occur = [i for i, x in enumerate(labels) if lab in x]
        one_hot_labels[lab] = one_hot_labels[lab].scatter_(dim=0, index=torch.LongTensor(occur), value=0)
    return one_hot_labels

##########################  PARAMETERS  ##################################

# Input files
train_files_path = project_dir + "/data/rnn_input/" + str(args.fold_num) + "/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/rnn_input/" + str(args.fold_num) + "/test.txt"
dev_file = project_dir + "/data/rnn_input/" + str(args.fold_num) + "/dev.txt"
predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

num_epochs = 200
max_batch_epoch = 100 // main_config.batch_size

from collections import defaultdict
net = None
hidden_size = 768

weights = np.load(project_dir + "/data/embeddings/weights_glove.npy")
vocab_len = weights.shape[0]
weights = np.append(weights.astype(float), np.zeros(shape=(1,300)), axis=0)

#################################################

def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings-1)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    emb_layer.weight.requires_grad = False
    return emb_layer

class Net(nn.Module):
    def __init__(self, weights_matrix):
        super(Net, self).__init__()
        self.embedding = create_emb_layer(weights_matrix)
        self.sigmoid = nn.Sigmoid()
        self.rnn = torch.nn.LSTM(input_size=300, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.rnn_back = torch.nn.LSTM(input_size=300, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.dec = torch.nn.Linear(hidden_size, len(predicate_list))
        # idiotic binary layers
        self.bin_array = nn.ModuleList([nn.Linear(hidden_size * 2, 2) for _ in range(len(predicate_list))])
        self.out_dropout = nn.Dropout()

    def forward(self, x, x_reversed, lens):
        x = self.embedding(x)
        x_reversed = self.embedding(x_reversed)
        # forward and backward being direction 0 and 1 respectively
        x = self.rnn(x)[0]
        x_reversed = self.rnn_back(x_reversed)[0]
        lens = lens.view(x.size()[0], 1, 1).expand(x.size()[0], 1, hidden_size)
        x = x.gather(1, lens).squeeze(1)
        x_reversed = x_reversed.gather(1, lens).squeeze(1)
        x = torch.cat((x, x_reversed), dim = 1)
        out_array = []
        for layer in self.bin_array:
            out_array.append(layer(self.out_dropout(x)))
        return torch.stack(out_array)

#################################################
from pathlib2 import Path
hyperparams = repr(main_config.__dict__)

def train():
    print(hyperparams)
    global net
    net = Net(weights_matrix=weights).to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=main_config.lr)
    streams = [pescador.Streamer(feature_gen_ham, ff, 1, ["input_ids", "label_ids"]) for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)

    batch_accumulator = BatchAccumulator(mux_stream, batch_size)

    best_stats_by_mrr = None
    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        for i, features in enumerate(batch_accumulator):
            inputs = [f["input_ids"] for f in features]
            max_len = max([len(x) for x in inputs])
            seq_lens = torch.LongTensor([len(x) - 1 for x in inputs]).to(device)
            samples = torch.stack(
                [torch.nn.functional.pad(torch.tensor(x), (0, max_len - len(x)), "constant", len(weights) - 1) for
                 x in inputs]).to(device)
            reverse = torch.stack([torch.nn.functional.pad(torch.tensor(np.flip(x, axis=0).copy()),
                                                           (0, max_len - len(x)), "constant", len(weights) - 1)
                                   for x in inputs]).to(device)
            answers = hot_label([f["label_ids"] for f in features], n_categories=len(predicate_list)).to(device)
            optimizer.zero_grad()
            outputs = net(samples, reverse, seq_lens)
            loss = 0
            for p, a in zip(outputs, answers):
                loss += criterion(p, a)
            loss /= batch_size

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

sm = torch.nn.Softmax()
lsm = torch.nn.LogSoftmax()
from collections import Counter

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
            by_guid_cnt = defaultdict(int)

            with open(output_file, "w") as f_test_out:
                streamer = pescador.Streamer(feature_gen_ham, streaming_file, 1,
                                             ["input_ids", "label_ids", "guid", "segment_ids", "input_mask", "splits"])
                for features in streamer:
                    features = features[0]
                    samples = torch.tensor(features["input_ids"], dtype=torch.long).to(device)
                    reverse = torch.tensor(np.flip(features["input_ids"], axis=0).copy()).to(device)
                    output = net(samples.unsqueeze(0), reverse.unsqueeze(0),
                                 torch.LongTensor([len(features["input_ids"]) - 1]).to(device))
                    entry = sm(output.squeeze(1)).cpu().data.numpy()[:, 0]
                    predict = np.where(entry >= 0.5)[0].tolist()
                    by_guid[features["guid"]].extend(predict)
                    by_guid_answ[features["guid"]] = features["label_ids"]
                    by_guid_cnt[features["guid"]] += 1

                for guid, utts in by_guid.items():
                    utts = dict(Counter(utts))
                    utts = [1 if utts.get(i, 0) > by_guid_cnt[guid] * main_config.mv else -1 for i in range(len(predicate_list))]
                    f_test_out.write(str(guid) + "\t" + repr(by_guid_answ[guid]) + '\t' + '\t'.join(
                        [str(y) for y in sorted(enumerate(utts), key=lambda x: x[1], reverse=True)]) + '\n')
        stats = compute_whatever_stats(open(dev_output_file).readlines())
        return stats

train()