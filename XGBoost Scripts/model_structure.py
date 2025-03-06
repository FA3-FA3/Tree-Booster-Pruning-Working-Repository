import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import networkx as nx
import re

with open(f'models/xgb_Full.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
model_s = model.get_booster()
n_trees = model_s.num_boosted_rounds()
dump = model_s.get_dump()
dump_l = dump[0].split("\n")

#functions for getting a tree of index, then get internal nodes, get leaves
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
    

'''
def plot_tree():
    G = nx.DiGraph()
'''