from collections import defaultdict
from pathlib import Path
project_dir = str(Path(__file__).parent.parent.parent)
print("\nCreating age data\n")

indexing_dict = eval(open(project_dir + "/data/character_indexing.txt").read())
paramita_path = project_dir + "/data/movie_characters_imdb_age_gender.txt"
age_map = {(-100,-13): 0, (-12,-6): 1, (-5,-1): 2, (0,4): 3, (5,11): 4, (12,100): 5}

film_dict = defaultdict(lambda: defaultdict(list))

mchd = dict((line.strip().split(":::")[1],line.strip().split(":::")[0]) for line in open(project_dir + "/data/movie_name_chaging_dict.txt").readlines())

for line in open(paramita_path):
    line = line.strip().split(" +++$+++ ")
    age = -1 if line[8] == "?" or int(line[8]) > 100 else int(line[8])
    if line[3] in mchd:
        if mchd[line[3]] == "":
            continue
        line[3] = mchd[line[3]]
    film_dict[line[3].lower()][line[1]] = age

import csv

age_dict = dict()
for k, v in indexing_dict.items():
    if k[2].lower() not in film_dict:
            print("ACHTUNG", k[2])
            age_dict[v] = 0
            continue
    else:
        if k[0] not in film_dict[k[2].lower()] or k[1] not in film_dict[k[2].lower()]:
            print("ACHTUNG", k)
            age_dict[v] = 0
            continue
    ch1 = film_dict[k[2].lower()][k[0]]
    ch2 = film_dict[k[2].lower()][k[1]]
    diff = 0
    if ch1 != -1 and ch2 != -1:
        for kk, vv in age_map.items():
            if kk[0] <= (ch1 - ch2)  <= kk[1]:
                diff = vv
                break
    age_dict[v] = diff

age_out2 = open(project_dir + "/data/age_difference.txt", "w").write(repr(age_dict))
