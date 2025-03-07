import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import networkx as nx
import re

#function for extracting node details to array
def node_str_to_arr(string):
    string = string.replace("\t", "")
    pattern = r"(\d+):\[(.*?)] yes=(\d+),no=(\d+),missing=(\d+)"
    match = re.search(pattern, string)
    
    if match:
        # Extract the values from the match object and return as a list
        return [
            int(match.group(1)),            # Number before the colon
            match.group(2),                 # String inside the brackets
            int(match.group(3)),            # 'yes' value
            int(match.group(4)),            # 'no' value
            int(match.group(5))             # 'missing' value
        ]
    else:
        return []


with open(f'models/xgb_Full.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
model_s = model.get_booster()
n_trees = model_s.num_boosted_rounds()
dump = model_s.get_dump()
class_trees = [None] * 9
for i in range(9):
    class_trees[i] = [dump[j] for j in range(i, len(dump), 9)]
    for j in range(len(class_trees[i])):
        class_trees[i][j] = class_trees[i][j].split("\n")
        #for k in range(len(class_trees[i][j])):
            #class_trees[i][j][k] = node_str_to_arr(class_trees[i][j][k])

#dump_l = dump[0].split("\n")



'''
split_dump = [None] * len(dump)
for i in range(len(dump)):
    split_dump[i] = dump[i].split("\n")

ordered_dump = [None] * len(dump)
for i in range(len(split_dump)):
    for j in range(len(split_dump[i])):
        tmp = [None] * len(split_dump[i])
        tmp[j] = node_str_to_arr(split_dump[i][j])
    ordered_dump[i] = tmp




    


def plot_tree():
    G = nx.DiGraph()
'''