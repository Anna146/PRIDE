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
train_files_path = project_dir + "/data/ham_input/" + str(args.fold_num) + "/train/"
train_files = [os.path.join(train_files_path, f) for f in os.listdir(train_files_path)]
test_file = project_dir + "/data/ham_input/" + str(args.fold_num) + "/test.txt"
dev_file = project_dir + "/data/ham_input/" + str(args.fold_num) + "/dev.txt"
predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

############################################

# Training
num_epochs = 50
max_batch_epoch = 1000 // main_config.batch_size

##################################  Train and eval  ######################

from collections import defaultdict
bert_size = 512
net = None
char_len = 100
utter_len = 100
hidden_size = main_config.hidden_size
hidden_size_attention = main_config.hidden_size_attention

vocab_weights = np.load(project_dir + "/data/embeddings/weights_movie.npy")
vocab_len = vocab_weights.shape[0]
vocab_weights = np.append(vocab_weights.astype(float), np.zeros(shape=(1,300)), axis=0)

############################################# Model #######################################################

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=num_embeddings-1)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out = self.sigm(self.fc(x))
        return out

class AttentionLayer2(nn.Module):
    def __init__(self, input_size, hidden_size_attention):
        super(AttentionLayer2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_attention)
        self.fc2 = nn.Linear(hidden_size_attention, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.sigm(self.fc1(x))
        out = self.sigm(self.fc2(x))
        return out


class Net(nn.Module):
    def __init__(self, hidden_size, char_len , utter_len, predicate_num, weights_matrix, hidden_size_attention=150, attention_type = 1):
        super(Net, self).__init__()
        self.embedding, num_embeddings, self.embedding_len = create_emb_layer(weights_matrix, True)
        self.fc1 = nn.Linear(self.embedding_len, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.attention1 = AttentionLayer2(self.embedding_len, hidden_size_attention) if attention_type == 1 else AttentionLayer(self.embedding_len)
        self.attention2 = AttentionLayer2(self.embedding_len, hidden_size_attention) if attention_type == 1 else AttentionLayer(self.embedding_len)
        self.softmax1 = nn.Softmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 2)
        self.char_len = char_len
        self.utter_len = utter_len
        self.fc = nn.Linear(hidden_size, predicate_num)

    def forward(self, x):
        # 0 load embeddings
        x = self.embedding(x).view(-1, self.char_len, self.utter_len, self.embedding_len)

        # 1 combine words
        att_matrix = self.attention1(x)
        att_matrix = torch.where(x.narrow(3, 0, 1) != 0, att_matrix, torch.ones_like(att_matrix) * np.NINF)
        att_matrix = self.softmax2(att_matrix)
        att_matrix = torch.where(x.narrow(2, 0, 1).narrow(3, 0, 1) != 0, att_matrix, torch.zeros_like(att_matrix))
        weights = att_matrix.view(-1, self.char_len * self.utter_len).cpu().data.numpy()
        att_matrix = att_matrix.expand_as(x)
        x = torch.mul(att_matrix, x)
        x = torch.sum(x, dim=2)
        x = x.view(-1, self.char_len, self.embedding_len)

        # 2 combine utternaces
        att_matrix = self.attention2(x)
        att_matrix = torch.where(x.narrow(2, 0, 1) != 0, att_matrix, torch.ones_like(att_matrix) * np.NINF)
        att_matrix = self.softmax1(att_matrix)
        utter_weights = att_matrix.cpu().data.numpy()
        att_matrix = att_matrix.expand_as(x)
        x = torch.mul(att_matrix, x)
        x = torch.sum(x, dim=1)

        # 3 do feed forward
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc(out)

        return out, weights, utter_weights

############################################################################################################

hyperparams = repr(main_config.__dict__)

config_num = args.config_path.split("/")[-1].strip(".txt")

def train():
    print(hyperparams)
    global net
    net = Net(hidden_size, char_len, utter_len, len(predicate_list), vocab_weights, hidden_size_attention)
    net.to(device)
    net.train()

    criterion = nn.BCEWithLogitsLoss() ###
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    streams = [pescador.Streamer(feature_gen_ham, ff, 1, ["input_ids", "label_ids", "segment_ids", "input_mask", "splits"])
               for ff in train_files]
    mux_stream = pescador.ShuffledMux(streams, random_state=33)
    batch_accumulator = BatchAccumulator(mux_stream, batch_size)

    # Train the Model
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch))
        for i, features in enumerate(batch_accumulator):
            samples = torch.tensor([f["input_ids"] for f in features], dtype=torch.long).to(device)
            answers = to_onehot([f["label_ids"] for f in features], n_categories=len(predicate_list)).to(device)
            optimizer.zero_grad()
            outputs, weights, utter_weights = net(samples)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
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
            with open(output_file, "w") as f_test_out:
                streamer = pescador.Streamer(feature_gen_ham, streaming_file, 1,
                                             ["input_ids", "label_ids", "guid", "segment_ids", "input_mask", "splits"])
                for features in streamer:
                    samples = torch.tensor([f["input_ids"] for f in features], dtype=torch.long).to(device)
                    labels = [f["label_ids"] for f in features]
                    output, weights, utter_weights = net(samples)
                    entry = output.cpu().data.numpy()[0]
                    f_test_out.write(str(features[0]["guid"]) + "\t" + str(list(labels[0])) + '\t' + '\t'.join(
                        [str(y) for y in sorted(enumerate(entry), key=lambda x: x[1], reverse=True)]) + '\n')

        stats = compute_whatever_stats(open(dev_output_file).readlines())
        return stats

train()