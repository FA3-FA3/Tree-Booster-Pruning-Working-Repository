import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import networkx as nx
import re

#function for extracting node details to array
def node_str_to_arr(string):
    string = string.replace("\t", "").strip()
    internal_pattern = r"(\d+):\[(.*?)] yes=(\d+),no=(\d+),missing=(\d+)"
    leaf_pattern = r"(\d+):leaf=(-?\d+\.\d+)"
    
    match = re.search(internal_pattern, string)
    if match:
        return [
            int(match.group(1)),  # Node number
            match.group(2),       # Condition inside brackets
            int(match.group(3)),  # 'yes' value
            int(match.group(4)),  # 'no' value
            int(match.group(5))   # 'missing' value
        ]
    
    match = re.search(leaf_pattern, string)
    if match:
        return [
            int(match.group(1)),  # Node number
            float(match.group(2)) # Leaf value
        ]
    
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
        for k in range(len(class_trees[i][j])):
            class_trees[i][j][k] = node_str_to_arr(class_trees[i][j][k])

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