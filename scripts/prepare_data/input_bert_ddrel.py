import os
import urllib
import random
from collections import defaultdict
from progressbar import ProgressBar

import numpy as np
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
print("\nCreating bert ddrel input\n")

random.seed(33)
import math

import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *

import re
from nltk import tokenize
import csv
from nltk.corpus import stopwords
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-cased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=True)

stop_words = set(stopwords.words('english'))
indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())
allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())

num_files = 30

cls = bert_vocab.index("[CLS]")

predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict

rares = set([line.strip() for line in open(project_dir + "/data/rares.txt")])

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)

names_list = eval(open(project_dir + "/data/names.txt").read())

def input_bert(fold):
    print("processing fold", fold)
    pbar = ProgressBar()
    max_seq_length = 512

    def preprocess_text(txt):
        txt = [w for w in tokenize.word_tokenize(txt) if w.lower() not in rares and w.lower() not in names_list]
        txt = [re.sub(r'[^a-zA-Z]', "", x) for x in txt]
        txt = " ".join([x for x in txt if len(x) > 0])
        txt = tokenizer.tokenize(txt)
        txt = tokenizer.convert_tokens_to_ids(txt)
        return txt

    inp_folder = project_dir + "/data/text_movies/"
    train_folder = project_dir + "/data/bert_ddrel_input/" + str(fold) + "/train/"
    test_file = project_dir + "/data/bert_ddrel_input/" + str(fold) + "/test.txt"
    dev_file = project_dir + "/data/bert_ddrel_input/" + str(fold) + "/dev.txt"

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

    for film in pbar(os.listdir(inp_folder)):
        cut_film = film
        if "(" in film:
            reses = re.findall("(.*)(\(.*\))", film)
            cut_film = reses[0][0].strip()
        if cut_film not in truth_data:
            continue
        for ff in os.listdir(inp_folder + film):
            probe_names = ff.replace(".txt", "").split("___")
            if tuple(probe_names) in truth_data[cut_film]:
                label = [predicate_list.index(x) for x in truth_data[cut_film][tuple(probe_names)]]
                names = tuple(probe_names)
            elif tuple([probe_names[1], probe_names[0]]) in truth_data[cut_film]:
                label = [predicate_list.index(x) for x in truth_data[cut_film][tuple([probe_names[1], probe_names[0]])]]
                names = tuple([probe_names[1], probe_names[0]])
            else:
                continue
            label = [x for x in label if x in allowed_labels]
            if label == -1 or label == []:
                continue
            ctr = indexing_dict[tuple([names[0], names[1], cut_film])]
            texts = []
            curr_seq = []
            for line in open(inp_folder + film + "/" + ff):
                if len(line.split("\t")) != 2:
                    if len(line.strip()) < 1:
                        if len(curr_seq) >= 3:
                            texts.append(curr_seq)
                    curr_seq = []
                    continue
                char, txt = line.split("\t")
                curr_text = preprocess_text(txt)
                if len(curr_text) < 2:
                    continue
                curr_seq.append(curr_text)
            if len(curr_seq) >= 3:
                texts.append(curr_seq)
            segs = []
            inps = []
            for seq in texts:
                seq = [cls] + [x for l in seq for x in l]
                seq = seq[:min(len(seq), max_seq_length)]
                segs.append([1] * len(seq) + [0] * (max_seq_length - len(seq)))
                seq += [0] * (max_seq_length - len(seq))
                inps.append(seq)
            for inp, seg in zip(inps, segs):
                res_features = InputFeatures(input_ids=inp, input_mask=seg,
                                             segment_ids=seg, label_ids=label, guid=ctr)
                res_features.save(whitelists[ctr])

for fo in range(5):
    input_bert(fo)