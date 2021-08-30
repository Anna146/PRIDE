import os
import re
import urllib
from collections import defaultdict
print("\nCreating relationship dimensions inference input\n")

from progressbar import ProgressBar
from transformers.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP

input_vocab_path = PRETRAINED_VOCAB_FILES_MAP['vocab_file']['bert-base-uncased']
bert_vocab = [line.strip().decode("utf-8") for line in urllib.request.urlopen(input_vocab_path)]

from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *

bert_len = 512
inp_folder = project_dir + "/data/text_movies/"
indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())

pbar = ProgressBar()

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict

truth_data = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_data)

predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())
test_file = project_dir + "/data/dimensions_training/inference.txt"
if os.path.exists(test_file):
    os.remove(test_file)

def prepare_input():
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
            txts = []
            for line in open(inp_folder + film + "/" + ff):
                if len(line.split("\t")) != 2:
                    continue
                char, txt = line.split("\t")
                txt = tokenizer.tokenize(txt)
                txt = tokenizer.convert_tokens_to_ids(txt)
                txts.append(txt)
            txts = [[0]] + txts
            for i, utt in enumerate(txts[1:]):
                prev = txts[i]
                inps = tokenizer.build_inputs_with_special_tokens(utt, prev)[:bert_len]
                segms = tokenizer.create_token_type_ids_from_sequences(utt, prev)[:bert_len]
                mask = [1] * len(inps)
                inps.extend([0] * (bert_len - len(inps)))
                segms.extend([0] * (bert_len - len(segms)))
                mask.extend([0] * (bert_len - len(mask)))
                feat = InputFeatures(input_ids=inps, input_mask=mask, segment_ids=segms, guid=ctr)
                feat.save(test_file)

prepare_input()