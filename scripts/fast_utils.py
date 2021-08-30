from collections import defaultdict
from pathlib import Path
project_dir = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_dir + '/scripts')
import numpy as np

predicate_list = [x.strip() for x in open(project_dir + "/data/relationship_list.txt").readlines()]

import pickle

class InputFeatures(object):
    def __init__(self, input_ids = None, input_mask = None, segment_ids = None, label_ids = None, guid = None,
                 plain_texts = None, splits = None, slots = None, transformer_segments = None, ham_inp = None,
                 ham_labs = None, sentiment = None):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.guid = guid
        self.sentiment = sentiment
        self.plain_texts = plain_texts
        self.splits = splits
        self.slots = slots
        self.transformer_segments = transformer_segments
        self.ham_inp = ham_inp
        self.ham_labs = ham_labs
        if type(segment_ids) != type(None):
            self.segment_a = np.where(self.segment_ids == 0, np.ones_like(self.segment_ids), np.zeros_like(segment_ids)) * self.input_mask
            self.segment_b = np.where(self.segment_ids == 1, np.ones_like(self.segment_ids), np.zeros_like(segment_ids)) * self.input_mask
    def save(self, file):
        with open(file, "ab") as f:
            pickle.dump(self.__dict__, f)
    def load(self, f):
        self.__dict__ = pickle.load(f)

import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def compute_binary_multilabel_macro_recall_k(filepath, threshold, offset=0, pass_dict = False, printing=False, significance = False):
    global predicate_list
    prof_dict = defaultdict(lambda: [0, 0]) # [TP, TP + FN]
    big_recall = 0
    user_scores = dict()
    for line in filepath:
        answ = line[0]
        predctions = line[1:]
        cnt = 0
        rec = 0
        for cls, score in predctions:
                if score < -100:
                    score = - 10
                if cls in answ: # then it is a true
                    prof_dict[cls][1] += 1
                    if sigmoid(score) >= threshold: # then it is true postitve
                        rec += 1
                        prof_dict[cls][0] += 1
                elif cls not in answ and sigmoid(score) >= threshold:
                    prof_dict[cls][0] += 0
                cnt += 1
    if not pass_dict and printing:
        print('\nrecalls=eval("%s")' % repr([(predicate_list[x[0]], x[1]) for x in prof_dict.items()]))
    for prof, stats in prof_dict.items():
        if stats[1] > 0:
            big_recall += stats[0] * 1.0 / stats[1]
    if significance:
        return user_scores
    return big_recall / len(prof_dict), dict(prof_dict)

def compute_binary_multilabel_macro_precision_k(filepath, threshold, offset=0, pass_dict = False, printing=False, significance = False):
    global predicate_list
    prof_dict = defaultdict(lambda: [0, 0]) # [TP, TP+FP]
    big_precision = 0
    user_scores = dict()
    for line in filepath:
        answ = line[0]
        #idd = line[1]
        predctions = line[1:]
        prec = 0
        pos = 0
        for k in answ:
            prof_dict[k][1] += 0
        for cls, score in predctions:
            #try:
                if score < -100:
                    score = - 10
                if sigmoid(score) >= threshold: # then it is a positive
                    prof_dict[cls][1] += 1
                    pos += 1
                    if cls in answ: # then it is a true positive
                        prof_dict[cls][0] += 1
                        prec += 1
        #user_scores[idd] = prec * 1.0 / pos
            #except:
            #    print("ERROR ON", predctions)
            #    exit(0)
    if not pass_dict and printing:
        print('\nprecisions=eval("%s")' % repr([(predicate_list[x[0]], x[1]) for x in prof_dict.items()]))
    for prof, stats in prof_dict.items():
        if stats[1] != 0:
            big_precision += stats[0] * 1.0 / stats[1]
    return big_precision / len(prof_dict), prof_dict

def compute_micro_f1(filepath, threshold, printing=False, offset=1):
    global predicate_list
    big_prec = 0
    big_rec = 0
    big_count = 0
    for line in filepath:
        big_count += 1
        answ = line[0]
        predctions = line[1:]
        tp = 0
        p = 0
        t = 0
        for cls, score in predctions:
            if score < -100:
                score = - 10
            if sigmoid(score) >= threshold: # then it is a positive
                p += 1
                if cls in answ: # then it is a true positive
                    tp += 1
            if cls in answ: # then it is true
                t += 1
        big_prec += 0 if p == 0 else tp / p
        big_rec += 0 if t == 0 else tp / t
    big_prec /= big_count
    big_rec /= big_count
    if big_prec + big_rec == 0:
        return 0
    else:
        return 2 * big_prec * big_rec / (big_prec + big_rec)

def compute_multilabel_f1(filepath, offset, p_dict, r_dict, threshold, printing = False):
    global predicate_list
    big_sum = 0
    big_count = 0
    f1_dict = dict()
    for cl in set(p_dict.keys()).union(set(r_dict.keys())):
        big_count += 1
        if p_dict.get(cl, [])[0] == 0 or r_dict.get(cl, [])[0] == 0:
            ###print(predicate_list[cl], p_dict[cl], r_dict[cl])
            continue
        p_dict[cl] = p_dict[cl][0] * 1.0 / p_dict[cl][1]
        r_dict[cl] = r_dict[cl][0] * 1.0 / r_dict[cl][1]
        big_sum += 2 * p_dict[cl] * r_dict[cl] / (p_dict[cl] + r_dict[cl])
        f1_dict[predicate_list[cl]] = round(2 * p_dict[cl] * r_dict[cl] / (p_dict[cl] + r_dict[cl]),3)
    if printing:
        print("F1s", f1_dict)
    return big_sum / big_count, f1_dict

def read_file(filepath):
    newlines = []
    for line in filepath:
        fields = line.split("\t")
        answ = eval(fields[1])
        predctions = [eval(x) for x in fields[2:]]
        newlines.append([answ] + predctions)
    return newlines

def compute_whatever_stats(output_file, threshold=0.5, printing=False):
    output_file = read_file(output_file)
    stats = dict()
    precision, prec_dict = compute_binary_multilabel_macro_precision_k(output_file, threshold, 1, pass_dict = True)
    recall, rec_dict = compute_binary_multilabel_macro_recall_k(output_file, threshold, 1, pass_dict = True)
    f1 = compute_multilabel_f1(output_file, 1, p_dict = prec_dict, r_dict = rec_dict, threshold=threshold, printing=printing)
    stats["macro f1"], f1s = round(f1[0], 2), f1[1]
    stats["macro precision"] = round(precision, 2)
    stats["macro recall"] = round(recall, 2)
    stats["micro f1"] = round(compute_micro_f1(output_file, threshold=threshold, printing=printing), 2)
    if printing:
        return stats, f1s
    else:
        return stats

def feature_gen_ham(filepath, batch_size, fields):
    with open(filepath, "rb") as f_in:
        while True:
            batch = []
            i = 0
            while i < batch_size:
                sample = InputFeatures()
                try:
                    sample.load(f_in)
                except (EOFError, pickle.UnpicklingError):
                    raise StopIteration
                batch.append(dict((x, sample.__dict__[x]) for x in fields))
                i += 1
            yield batch

def BatchAccumulator(stream, batch_size):
    stream = stream.iterate()
    while True:
        batch = []
        i = 0
        while i < batch_size:
            try:
                sample = next(stream)
            except (EOFError, pickle.UnpicklingError):
                raise StopIteration
            batch.append(sample[0])
            i += 1
        if len(batch) == batch_size:
            yield batch
        else:
            raise StopIteration