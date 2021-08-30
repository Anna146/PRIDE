from pathlib import Path
project_dir = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_dir + '/scripts')
from fast_utils import *
import os
from joblib import Parallel, delayed
import argparse
import time

num_folds = 5

main_metrics = "macro f1"
secondary_metrics = "macro precision"

max_epochs = 100

def run_dev(ep):
    dev_temp_eval = []
    folds_covered = 0
    folds = []
    for fi in os.listdir(project_dir + "/data/outputs/dev/" + method):
        if fi.replace(".txt", "").split("_")[1] == str(ep):
            folds_covered += 1
            folds.append(fi.replace(".txt", "").split("_")[0])
            dev_temp_eval.extend(open(project_dir + "/data/outputs/dev/" + method + "/" + fi).readlines())
    if folds_covered == num_folds:
        tstats = compute_whatever_stats(dev_temp_eval, 0.5)
        param_dicts[str(ep)] = tstats

methods = ["rnn", "bert_ddrel", "ham", "bert_conv", "pride"]

for method in methods:
    param_dicts = dict()
    Parallel(n_jobs=-1, backend="threading")(delayed(run_dev)(ep) for ep in range(0, max_epochs))

    sorted_stats = sorted(param_dicts.items(), key=lambda x: (x[1][main_metrics], x[1][secondary_metrics], x[1]["macro recall"], x[1]["micro f1"]), reverse=True)
    best_stats = sorted_stats[0]
    best_epoch = best_stats[0]
    best_stats = best_stats[1]

    dev_temp_eval = []
    for fold in range(num_folds):
        dev_temp_eval.extend(open(project_dir + "/data/outputs/dev/" + method + "/" + str(fold) + "_" + str(best_epoch) + ".txt").readlines())
    stats_array = []
    thresholds = [x * 1.0 / 100 for x in range(100)]
    for th in thresholds:
        stats = compute_whatever_stats(dev_temp_eval, th)
        stats["threshold"] = th
        stats_array.append(stats)
    sort_order = [main_metrics, secondary_metrics, "macro recall", "micro f1"]
    best_stats = sorted(stats_array, key=lambda x: tuple(x[y] for y in sort_order), reverse=True)[0]
    gold_th = sorted(stats_array, key=lambda x: tuple(x[y] for y in sort_order), reverse=True)[0]["threshold"]

    print(method)
    dev_temp_eval = []
    for fold in range(num_folds):
        dev_temp_eval.extend(open(project_dir + "/data/outputs/test/" + method + "/" + str(fold) + "_" + str(best_epoch) + ".txt").readlines())
    tstats, f1s = compute_whatever_stats(dev_temp_eval, gold_th, printing = True)
    print("dev\t", best_stats)
    print("test\t", tstats)
    print("\n\n")
