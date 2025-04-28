import matplotlib.pyplot as plt
import json
import re
import numpy as np

with open('results/manual_opt_xgb_results_arrays.txt', 'r') as input_file:
    lines = input_file.readlines()

f1_arrays = [json.loads(line) for line in lines]

with open('results/manual_opt_xgb_results.txt', 'r') as input_file:
    lines = input_file.readlines()

tree_lists = []

for line in lines:#Extract tree list
    match = re.search(r'Columns Removed: \{([^}]*)\}', line)
    if match:
        # Extract the numbers inside the braces
        tree_list = match.group(1)
        # Split into a list of integers
        tree_list = [int(x.strip()) for x in tree_list.split(',')]
        tree_lists.append(tree_list)


for i, arr in enumerate(f1_arrays):#Lists set to only macro-F1 scores and number of trees
    f1_arrays[i] = arr[1]
    tree_lists[i] = len(tree_lists[i])

plt.scatter(tree_lists, f1_arrays)
plt.title('Scatter of Macro-F1 Scores vs N Trees Pruned')
plt.xlabel('N Trees Pruned')
plt.ylabel('Macro-F1 Scores')
plt.show()