import os
import urllib
import random
from collections import defaultdict

import numpy as np

random.seed(33)
import math
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
print("\nCreating rnn input\n")

import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *

import re
from nltk import tokenize
import csv
from nltk.corpus import stopwords
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
vocab_path = project_dir + "/data/embeddings/vocabulary_glove.txt"
vocab = dict((x[1].strip(), x[0]) for x in enumerate(open(vocab_path).readlines()))
oov_index = len(vocab)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

stop_words = set(stopwords.words('english'))
indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())
allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)
predicate_file = project_dir + "/data/relationship_list.txt"
predicate_list = [line.strip() for line in open(predicate_file, "r")]

def input_bert(fold):

    def preprocess_text(txt):
        global curr_words
        txt = [w.lower() for w in tokenize.word_tokenize(txt)]
        txt = [re.sub(r'[^a-zA-Z]', "", x) for x in txt]
        txt = [x for x in txt if len(x) > 1 and x not in stop_words]
        txt = [vocab[w] if w in vocab else oov_index for w in txt]
        return txt

    inp_folder = project_dir + "/data/text_movies/"
    train_folder = project_dir + "/data/rnn_input/" + str(fold) + "/train/"
    test_file = project_dir + "/data/rnn_input/" + str(fold) + "/test.txt"
    dev_file = project_dir + "/data/rnn_input/" + str(fold) + "/dev.txt"

    train_wl = project_dir + "/data/speaker_whitelists/" + str(fold) + "/train/"
    test_wl = project_dir + "/data/speaker_whitelists/" + str(fold) + "/test.txt"
    dev_wl = project_dir + "/data/speaker_whitelists/" + str(fold) + "/dev.txt"

    whitelists = dict()
    for line in open(test_wl):
        whitelists[int(line.strip())] = test_file
    for line in open(dev_wl):
        whitelists[int(line.strip())] = dev_file
    for fi in os.listdir(train_wl):
        for line in open(os.path.join(train_wl, fi)):
            whitelists[int(line.strip())] = os.path.join(train_folder, fi)
    Path(train_folder).mkdir(exist_ok=True, parents=True)

    # Clear contents
    for ff in os.listdir(train_folder):
        os.remove(train_folder + ff)
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(dev_file):
        os.remove(dev_file)

    for film in os.listdir(inp_folder):
        cut_film = film
        if "(" in film:
            reses = re.findall("(.*)(\(.*\))", film)
            cut_film = reses[0][0].strip()
        if cut_film not in truth_data:
            continue
        for ff in os.listdir(inp_folder + film):
            probe_names = ff.replace(".txt", "").split("___")
            label = []
            if tuple(probe_names) in truth_data[cut_film]:
                label = [predicate_list.index(x) for x in truth_data[cut_film][tuple(probe_names)]]
                names = tuple(probe_names)
            elif tuple([probe_names[1], probe_names[0]]) in truth_data[cut_film]:
                label = [predicate_list.index(x) for x in truth_data[cut_film][tuple([probe_names[1], probe_names[0]])]]
                names = tuple([probe_names[1], probe_names[0]])
            label = [x for x in label if x in allowed_labels]
            if label == -1 or label == []:
                continue
            ctr = indexing_dict[tuple([names[0], names[1], cut_film])]
            texts = []
            speakers = []
            characters = dict((x[1], x[0]) for x in enumerate(names))
            for line in open(inp_folder + film + "/" + ff):
                if len(line.split("\t")) != 2:
                    continue
                char, txt = line.split("\t")
                txt = preprocess_text(txt)
                if len(txt) < 1:
                    continue
                char = characters[char]
                txt = [vocab["s1"]] + txt if char == 0 else [vocab["s3"]] + txt
                texts.append(txt)
                speakers.append(char)
            concatenated = []
            for i, ttxt in enumerate(texts):
                concatenated.extend(ttxt)
                if i % 20 == 19:
                    res_features = InputFeatures(input_ids=concatenated, label_ids=label, guid=ctr)
                    res_features.save(whitelists[ctr])
                    concatenated = []
            if concatenated != []:
                res_features = InputFeatures(input_ids=concatenated, label_ids=label, guid=ctr)
                res_features.save(whitelists[ctr])

for fo in range(5):
    print("processing fold", fo)
    input_bert(fo)