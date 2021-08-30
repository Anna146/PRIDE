import os
import urllib
import random
from collections import defaultdict
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
print("\nCreating bert conv input\n")

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

num_files = 30
allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())

split_utt_num = 70 # split if chars have more

cls = bert_vocab.index("[CLS]")
sep = bert_vocab.index("[SEP]")
eot = bert_vocab.index("[unused0]")

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)

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

min_per_file = 200
predicate_file = project_dir + "/data/relationship_list.txt"
predicate_list = [line.strip() for line in open(predicate_file, "r")]


def input_bert(fold_num):
    oov_index = 0
    max_words = 512
    bert_batch = 1
    max_seq_length = bert_batch * max_words

    def preprocess_text(txt):
        txt = tokenizer.tokenize(txt)
        txt = tokenizer.convert_tokens_to_ids(txt)
        return txt

    inp_folder = project_dir + "/data/text_movies/"
    train_folder = project_dir + "/data/bert_conv_input/" + str(fold) + "/train/"
    test_file = project_dir + "/data/bert_conv_input/" + str(fold) + "/test.txt"
    dev_file = project_dir + "/data/bert_conv_input/" + str(fold) + "/dev.txt"

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

    filewise_disr = defaultdict(list)
    filewise_counts = defaultdict(int)

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
                texts.append(curr_text)
                segments.append([speakers[char]] * len(curr_text))
            segments, texts = collapser(segments, texts)
            acc_texts = []
            acc_segs = []
            ctr = indexing_dict[tuple([names[0], names[1], cut_film])]
            text_arr = []
            seg_arr = []
            mask_arr = []
            curr_seq = 0
            for t, s in zip(texts, segments):
                if curr_seq + len(t) >= max_seq_length - 2:
                    acc_texts = [cls] + [x for l in acc_texts for x in l][:max_seq_length - 2] + [sep]
                    acc_segs = [0] + [x for l in acc_segs for x in l][:max_seq_length - 2] + [0]
                    mask = [1] * len(acc_texts)
                    acc_texts = np.array(acc_texts + [oov_index] * (max_seq_length - len(acc_texts)))
                    acc_segs = np.array(acc_segs + [0] * (max_seq_length - len(acc_segs)))
                    mask = np.array(mask + [0] * (max_seq_length - len(mask)))
                    text_arr.append(acc_texts)
                    mask_arr.append(mask)
                    seg_arr.append(acc_segs)
                    acc_segs = []
                    acc_texts = []
                    curr_seq = 0
                curr_seq += len(t)
                acc_segs.append(s)
                acc_texts.append(t)
            if len(acc_texts) != 0:
                acc_texts = [cls] + [x for l in acc_texts for x in l][:max_seq_length - 2] + [sep]
                acc_segs = [0] + [x for l in acc_segs for x in l][:max_seq_length - 2] + [0]
                mask = [1] * len(acc_texts)
                acc_texts = np.array(acc_texts + [oov_index] * (max_seq_length - len(acc_texts)))
                acc_segs = np.array(acc_segs + [0] * (max_seq_length - len(acc_segs)))
                mask = np.array(mask + [0] * (max_seq_length - len(mask)))
                text_arr.append(acc_texts)
                mask_arr.append(mask)
                seg_arr.append(acc_segs)
            if "test" in whitelists[ctr] or "dev" in whitelists[ctr]:
                res_features = InputFeatures(input_ids=text_arr, input_mask=mask_arr,
                                             segment_ids=seg_arr, label_ids=label, guid=ctr)
                res_features.save(whitelists[ctr])
            else:
                which_file = whitelists[ctr].replace(".txt", "")[:-1]
                filewise_counts[which_file] += len(text_arr)
                filewise_disr[which_file].extend([InputFeatures(input_ids=a, input_mask=b,
                                             segment_ids=c, label_ids=label, guid=ctr) for a, b, c in zip(text_arr, mask_arr, seg_arr)])
    import math
    for f_name, feat_arr in filewise_disr.items():
        parts_to_split = math.ceil(len(feat_arr) / min_per_file)
        chunksize = int(math.ceil(len(feat_arr) / parts_to_split))
        feat_arr = [feat_arr[i * chunksize:i * chunksize + chunksize] for i in range(parts_to_split)]
        for i, sub_arr in enumerate(feat_arr):
            for feat in sub_arr:
                feat.save(f_name + str(i) + ".txt")

for fold in range(5):
    print("processing fold", fold)
    input_bert(fold)