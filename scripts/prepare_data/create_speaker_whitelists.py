import os
import re
from collections import defaultdict
import random
import math
import shutil
from pathlib import Path
print("\nCreating input splits\n")

project_dir = str(Path(__file__).absolute().parent.parent.parent)

def load_truth_labels(inp_dict):
    movie_dict = defaultdict(dict)
    for k, v in inp_dict.items():
        k = k.split(":::")
        movie_dict[k[0]][tuple(k[1:])] = v
    return movie_dict
truth_labels = eval(open(project_dir + "/data/movie_gt.txt").read())
truth_data = load_truth_labels(truth_labels)

indexing_file = project_dir + "/data/character_indexing.txt"
indexing_dict = dict()
allowed_labels = eval(open(project_dir + "/data/allowed_labels.txt").read())
predicate_file = project_dir + "/data/relationship_list.txt"
predicate_list = [line.strip() for line in open(predicate_file, "r")]

inp_folder = project_dir + "/data/text_movies/"
train_folder = project_dir + "/data/speaker_whitelists/"
splits = eval(open(project_dir + "/data/best_split.txt").read())
num_folds = len(splits)
max_per_file = 50

ctr = 0
for sp in range(len(splits)):
    if os.path.exists(train_folder + str(sp)):
        shutil.rmtree(train_folder + str(sp))
    Path(train_folder + str(sp) + "/train/").mkdir(exist_ok=True, parents=True)
    test_sp = sp
    dev_sp = sp - 1 if sp > 0 else len(splits) - 1
    label_dict = defaultdict(list)
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
            if label != -1 and label != []:
                if tuple([names[0], names[1], cut_film]) not in indexing_dict:
                    indexing_dict[tuple([names[0], names[1], cut_film])] = ctr
                    ctr += 1
                idd = indexing_dict[tuple([names[0], names[1], cut_film])]
                if cut_film in splits[test_sp]:
                    open(train_folder + str(sp) + "/test.txt", "a").write(str(idd) + "\n")
                elif cut_film in splits[dev_sp]:
                    open(train_folder + str(sp) + "/dev.txt", "a").write(str(idd) + "\n")
                else:
                    label_dict[label[0]].append(idd)
    for la, chrs in dict(label_dict).items():
        num_files = math.ceil(len(chrs) / max_per_file)
        random.shuffle(chrs)
        for i in range(len(chrs)):
            open(train_folder + str(sp) + "/train/" + predicate_list[la] + str(random.randint(0, num_files - 1)) + ".txt", "a").write(str(chrs[i]) + "\n")
    print("done fold", sp)
open(indexing_file, "w").write(repr(indexing_dict))
