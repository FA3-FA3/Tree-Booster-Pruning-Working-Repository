import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import networkx as nx
import re
import matplotlib.pyplot as plt
import shap

#function for extracting node details to array
def node_str_to_arr(string):
    string = string.replace("\t", "").strip()
    internal_pattern = r"(\d+):\[(\d+)<(.*?)] yes=(\d+),no=(\d+),missing=(\d+)"
    leaf_pattern = r"(\d+):leaf=(-?\d+\.\d+)"
    
    match = re.search(internal_pattern, string)
    if match:
        return [
            int(match.group(1)),  # Node number
            int(match.group(2)),  # Integer before < operator
            match.group(3),       # The rest of the condition (string after <)
            int(match.group(4)),  # 'yes' value
            int(match.group(5)),  # 'no' value
            int(match.group(6))   # 'missing' value
        ]
    
    match = re.search(leaf_pattern, string)
    if match:
        return [
            int(match.group(1)),  # Node number
            float(match.group(2)) # Leaf value
        ]
    
    return []

def model_dump_to_array(model):
    dump = model.get_booster().get_dump()
    class_trees = [None] * 9
    for i in range(9):
        class_trees[i] = [dump[j] for j in range(i, len(dump), 9)]
        for j in range(len(class_trees[i])):
            class_trees[i][j] = class_trees[i][j].split("\n")
            for k in range(len(class_trees[i][j])):
                class_trees[i][j][k] = node_str_to_arr(class_trees[i][j][k])
    return class_trees

def plot_binary_tree(tree):
    G = nx.DiGraph()
    
    for i in range(len(tree)):
        if len(tree[i]) == 6:
            index = tree[i][0]
            feature_index = tree[i][1]
            condition = tree[i][2]
            yes_val = tree[i][3]
            no_val = tree[i][4]
            missing_val = tree[i][5]
            
            G.add_node(str(index), label=f"Index {index}\nFeature I {feature_index}\n<{condition}\nYes {yes_val}\nNo {no_val}\nMissing {missing_val}")
            

        elif len(tree[i]) == 2:
            index = tree[i][0]
            leaf_val = tree[i][1]
            G.add_node(str(index), label=f"Index {index}\n{leaf_val}")

    for i in range(len(tree)):
        if len(tree[i]) == 6:
            index = tree[i][0]
            yes_val = tree[i][3]
            no_val = tree[i][4]
            
            G.add_edge(str(index), str(yes_val))
            G.add_edge(str(index), str(no_val))
    
    G.graph['graph'] = {'ranksep': '2.0', 'nodesep': '8.0'}
    
    positions = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    
    plt.figure(figsize=(30, 15))
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos=positions, with_labels=True, labels=labels, 
            node_size=3000, node_color="skyblue", font_size=3.5, 
            font_weight="bold", edge_color="gray")
    plt.title("Binary Tree")
    plt.savefig("binary_tree.png", format="png", dpi=300)
    plt.show()

def shap_features(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    
    return shap_values

def shap_features_least(model, X_train):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # For multi-class: shap_values is (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Average over samples and classes
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # shape: (n_features,)
    else:
        raise ValueError("Unexpected SHAP value shape. Expected 3D array for multi-class.")

    # Get indices of least important features
    least_important_idx = np.argsort(mean_abs_shap)[:20]

    # Subset the training data
    X_least = X_train.iloc[:, least_important_idx]

    # Plot SHAP summary bar plot for least important
    shap.summary_plot(shap_values[:, least_important_idx, :], X_least, plot_type="bar")

    return shap_values



'''
with open(f'models/xgb_Full.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
class_trees = model_dump_to_array(model)

plot_binary_tree(class_trees[0][0])'''
