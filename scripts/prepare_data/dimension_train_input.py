import csv
import urllib
import re
import random
from collections import defaultdict
import os
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP
input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
print("\nCreating relationship dimension training input\n")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *

interesting_dims = ['cooperative vs. noncooperative (interactions)', 'cooperative vs. noncooperative (relationships)',
                   'equal vs. hierarchical', 'intense vs. superficial', 'pleasure vs. work oriented',
                   'active vs. passive (interactions)', 'active vs. passive (relationships)', 'intimate vs.unintimate',
                   'temporary vs. long term', 'concurrent vs. non concurrent', 'near vs. distant']

fields = ["Person being talked about", 'Speaker', 'Utterance', 'The reference with POS', 'Utterances serial number in the original dataset', 'Total Utterances in the episode', 'cooperative vs. noncooperative (interactions)', 'cooperative vs. noncooperative (relationships)', 'equal vs. hierarchical', 'intense vs. superficial', 'pleasure vs. work oriented', 'active vs. passive (interactions)', 'active vs. passive (relationships)', 'intimate vs.unintimate', 'temporary vs. long term', 'concurrent vs. non concurrent', 'near vs. distant', 'Episode number', 'Utterances serial number in our dataset', 'Previous Utterance']
utt_ind = fields.index("Utterance")
prev_ind = fields.index('Previous Utterance')
bert_len = 512

def prepare_input(dimension):
    print("processing", dimension)
    dimm_ind = fields.index(dimension)
    reader = csv.reader(open(project_dir + "/data/dimensions.csv"), delimiter='#')
    all_lines = {"1": [], "-1": [], "0": []}
    for line in reader:
        utt = line[utt_ind]
        prev = line[prev_ind]
        utt = tokenizer.tokenize(utt)
        utt = tokenizer.convert_tokens_to_ids(utt)
        prev = tokenizer.tokenize(prev)
        prev = tokenizer.convert_tokens_to_ids(prev)
        inps = tokenizer.build_inputs_with_special_tokens(utt, prev)
        segms = tokenizer.create_token_type_ids_from_sequences(utt, prev)
        mask = [1] * len(inps)
        inps.extend([0] * (bert_len - len(inps)))
        segms.extend([0] * (bert_len - len(segms)))
        mask.extend([0] * (bert_len - len(mask)))
        feat = InputFeatures(input_ids=inps, input_mask=mask, segment_ids=segms, label_ids=line[dimm_ind])
        all_lines[line[dimm_ind]].append(feat)
    test_set = []
    train_set = []
    dev_set = []
    for k, v in all_lines.items():
        test_set.extend(v[:int(len(v) * 0.1)])
        dev_set.extend(v[int(len(v) * 0.1):int(len(v) * 0.2)])
        train_set.extend(v[int(len(v) * 0.2):])
    random.shuffle(test_set)
    random.shuffle(train_set)
    random.shuffle(dev_set)
    train_file = project_dir + "/data/dimensions_training/" + dimension + "/train.txt"
    test_file = project_dir + "/data/dimensions_training/" + dimension + "/test.txt"
    dev_file = project_dir + "/data/dimensions_training/" + dimension + "/dev.txt"
    Path(project_dir + "/data/dimensions_training/" + dimension).mkdir(exist_ok=True, parents=True)
    for f in train_set:
        f.save(train_file)
    for f in test_set:
        f.save(test_file)
    for f in dev_set:
        f.save(dev_file)

for dimm in interesting_dims:
    prepare_input(dimm)