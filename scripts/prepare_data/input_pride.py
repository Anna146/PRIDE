import os
import random
import urllib
from collections import defaultdict

import numpy as np
from progressbar import ProgressBar

random.seed(33)
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
print("\nCreating PRIDE input <3<3<3 \n")

import math
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
import re
from nltk import tokenize
import csv
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())
allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())

cls = bert_vocab.index("[CLS]")
sep = bert_vocab.index("[SEP]")
eot = bert_vocab.index("[unused1]")
s1 = bert_vocab.index("[unused2]")
s2 = bert_vocab.index("[unused3]")

predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict

def collapser(segments, texts):
    new_segments = []
    new_texts = []
    prev_label = None
    prev_text = None
    for i in range(len(segments)):
        if segments[i] == prev_label:
            prev_text += texts[i][1:]
        if segments[i] == 3:
            if prev_label != None:
                new_segments.append(prev_label)
                new_texts.append(prev_text)
                new_segments.append(3)
                new_texts.append(texts[i])
            prev_text = None
            prev_label = None
            continue
        if segments[i] != prev_label:
            if prev_label != None:
                new_segments.append(prev_label)
                new_texts.append(prev_text)
            prev_text = texts[i]
            prev_label = segments[i]
    return new_segments, new_texts

rares = set([line.strip() for line in open(project_dir + "/data/rares.txt")])

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)

names_list = eval(open(project_dir + "/data/names.txt").read())

def crop_splits(sp):
    exc_len = sum(sp) - max_seq_length + 2
    while exc_len > 0:
        if sp[-1] <= exc_len:
            sp = sp[:-1]
        else:
            sp[-1] -= exc_len
        exc_len = sum(sp) - max_seq_length + 2
    return sp

def preprocess_text(txt):
    txt = [w.lower() for w in tokenize.word_tokenize(txt) if w.lower() not in rares and w.lower() not in names_list]
    txt = [re.sub(r'[^a-zA-Z]', "", x) for x in txt]
    txt = " ".join([x for x in txt if len(x) > 0])
    txt = tokenizer.tokenize(txt)
    txt = tokenizer.convert_tokens_to_ids(txt) + [eot]
    return txt

oov_index = 0
max_seq_length = 512

def input_bert(fold):
    pbar = ProgressBar()
    print("processing fold", fold)

    inp_folder = project_dir + "/data/text_movies/"
    train_folder = project_dir + "/data/pride_input/" + str(fold) + "/train/"
    test_file = project_dir + "/data/pride_input/" + str(fold) + "/test.txt"
    dev_file = project_dir + "/data/pride_input/" + str(fold) + "/dev.txt"

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
            texts = []
            segments = []
            speakers = dict((x[1], x[0]) for x in enumerate(names))
            for line in open(inp_folder + film + "/" + ff):
                if len(line.split("\t")) != 2:
                    continue
                char, txt = line.split("\t")
                curr_text = preprocess_text(txt)
                if len(curr_text) < 2:
                    continue
                texts.append(curr_text)
                segments.append(speakers[char])
            segments, texts = collapser(segments, texts)
            acc_texts = []
            acc_segs = []
            try:
                ctr = indexing_dict[tuple([names[0], names[1], cut_film])]
            except:
                print(tuple([names[0], names[1], cut_film]))
                continue
            text_arr = []
            seg_arr = []
            mask_arr = []
            split_arr = []
            curr_seq = 0
            if segments[0] == 1:
                texts = [[bert_vocab.index("the")]] + texts
                segments = [0] + segments
            for t, s in zip(texts, segments):
                if curr_seq + len(t) >= max_seq_length:
                    acc_texts1 = [acc_texts[i] for i in range(len(acc_texts))]
                    acc_segs1 = [[acc_segs[i]] * len(acc_texts[i]) for i in range(len(acc_texts))]
                    splits1 = [len(x) for x in acc_texts1]
                    splits1 = [(1, 0)] + list(zip(crop_splits(splits1), acc_segs))
                    assert len(acc_texts1) == len(acc_segs1)
                    acc_texts1 = [cls] + [x for l in acc_texts1 for x in l][:max_seq_length - 2] + [sep]
                    acc_segs1 = [0] + [x for l in acc_segs1 for x in l][:max_seq_length - 2] + [1]
                    mask1 = [1] * len(acc_texts1)
                    splits1 += [(1, 1)] if max_seq_length - len(acc_texts1) <= 0 else [
                        (max_seq_length - len(acc_texts1) + 1, 1)]
                    acc_texts1 = np.array(acc_texts1 + [oov_index] * (max_seq_length - len(acc_texts1)))
                    acc_segs1 = np.array(acc_segs1 + [1] * (max_seq_length - len(acc_segs1)))
                    mask1 = np.array(mask1 + [0] * (max_seq_length - len(mask1)))
                    ##
                    acc_segs = []
                    acc_texts = []
                    text_arr.append(acc_texts1)
                    mask_arr.append(mask1)
                    seg_arr.append(acc_segs1)
                    split_arr.append(splits1)
                    curr_seq = 0
                curr_seq += len(t)
                acc_segs.append(s)
                acc_texts.append(t)
            if len(acc_texts) != 0:
                acc_texts1 = [acc_texts[i] for i in range(len(acc_texts))]
                acc_segs1 = [[acc_segs[i]] * len(acc_texts[i]) for i in range(len(acc_texts))]
                splits1 = [len(x) for x in acc_texts1]
                splits1 = [(1, 0)] + list(zip(crop_splits(splits1), acc_segs))
                assert len(acc_texts1) == len(acc_segs1)
                acc_texts1 = [cls] + [x for l in acc_texts1 for x in l][:max_seq_length - 2] + [sep]
                acc_segs1 = [0] + [x for l in acc_segs1 for x in l][:max_seq_length - 2] + [1]
                mask1 = [1] * len(acc_texts1)
                splits1 += [(1, 1)] if max_seq_length - len(acc_texts1) <= 0 else [
                    (max_seq_length - len(acc_texts1) + 1, 1)]
                acc_texts1 = np.array(acc_texts1 + [oov_index] * (max_seq_length - len(acc_texts1)))
                acc_segs1 = np.array(acc_segs1 + [1] * (max_seq_length - len(acc_segs1)))
                mask1 = np.array(mask1 + [0] * (max_seq_length - len(mask1)))
                ##
                text_arr.append(acc_texts1)
                mask_arr.append(mask1)
                seg_arr.append(acc_segs1)
                split_arr.append(splits1)
            res_features = InputFeatures(input_ids=text_arr, input_mask=mask_arr,
                                         segment_ids=seg_arr, label_ids=label, guid=ctr, splits=split_arr)
            res_features.save(whitelists[ctr])

for fo in range(5):
    input_bert(fo)