import os
import urllib
import random
from collections import defaultdict
from progressbar import ProgressBar
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
import numpy as np

random.seed(33)
import math

import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
print("\nCreating ham input\n")

import re
from nltk import tokenize
import csv
from nltk.corpus import stopwords
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

stop_words = set(stopwords.words('english'))
indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())

allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())

num_files = 30

cls = bert_vocab.index("[CLS]")
sep = bert_vocab.index("[SEP]")
eot = bert_vocab.index("[unused1]")
s1 = bert_vocab.index("[unused2]")
s2 = bert_vocab.index("[unused3]")

names_list = eval(open(project_dir + "/data/names.txt").read())

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
            prev_text += texts[i]
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

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)
predicate_file = project_dir + "/data/relationship_list.txt"
predicate_list = [line.strip() for line in open(predicate_file, "r")]

rares = set([line.strip() for line in open(project_dir + "/data/rares.txt")])

def input_bert(fold):
    vocab_path = project_dir + "/data/embeddings/vocabulary_movie.txt"
    vocab = dict((x[1].strip(), x[0]) for x in enumerate(open(vocab_path).readlines()))
    oov_index = len(vocab)
    max_words = 100
    max_utterance = 100

    pbar = ProgressBar()

    def preprocess_text(txt):
        txt = [w.lower() for w in tokenize.word_tokenize(txt) if w.lower() not in rares and w.lower() not in names_list]
        txt = [re.sub(r'[^a-zA-Z]', "", x) for x in txt]
        txt = [x for x in txt if len(x) > 0]
        txt = [vocab[w] for w in txt if w in vocab]
        txt = txt[:max_words]
        txt = [x for x in txt if x != oov_index]
        return txt

    inp_folder = project_dir + "/data/text_movies/"
    train_folder = project_dir + "/data/ham_input/" + str(fold) + "/train/"
    test_file = project_dir + "/data/ham_input/" + str(fold) + "/test.txt"
    dev_file = project_dir + "/data/ham_input/" + str(fold) + "/dev.txt"

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
            speakers = []
            speakers_dict = dict((x[1], x[0]) for x in enumerate(names))
            for line in open(inp_folder + film + "/" + ff):
                if len(line.split("\t")) != 2:
                    continue
                char, txt = line.split("\t")
                curr_text = preprocess_text(txt)
                if len(txt) < 1:
                    continue
                texts.append(curr_text)
                speakers.append(speakers_dict[char])
            speakers, texts = collapser(speakers, texts)
            texts_new = []
            for t in texts:
                if len(t) < 1:
                    continue
                t.extend([oov_index] * (max_words - len(t)))
                texts_new.append(t[:max_words])
            texts = texts_new
            texts = texts[:max_utterance]
            speakers = speakers[:max_utterance]
            speaker_set = list(set(speakers))
            assert len(speaker_set) == 2
            speakers.extend(["NONE"] * (max_utterance - len(speakers)))
            segments = [[0] * max_words if x == names[0] else [1] * max_words for x in speakers]
            texts.extend([[oov_index] * max_words] * (max_utterance - len(texts)))
            texts = np.array([str(x) for t in texts for x in t], dtype=int)
            mask = np.where(texts == oov_index, np.zeros_like(texts, dtype=int), np.ones_like(texts, dtype=int))
            segments = np.array([x for l in segments for x in l], dtype=int)
            res_features = InputFeatures(input_ids=texts, input_mask=mask,
                                         segment_ids=segments, label_ids=label, guid=ctr)
            res_features.save(whitelists[ctr])

for fo in range(5):
    print("processing fold", fo)
    input_bert(fo)