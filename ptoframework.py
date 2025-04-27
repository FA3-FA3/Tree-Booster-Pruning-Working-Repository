import pickle
import xgboost as xgb
import json
import tempfile
import pandas as pd

def prune_dump_edit_reload(model_name, index):#Broken/abandoned dump editing tree-based pruning function
    with open(f"models/{model_name}.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    booster = model.get_booster()

    json_str = booster.save_raw(raw_format='json').decode('utf-8')#Loading from here is valid, issue is after this
    model_json = json.loads(json_str)
    model_json2 = json.loads(json_str)

    model = model_json["learner"]["gradient_booster"]["model"]
    trees = model["trees"]
    tree_info = model["tree_info"]
    iteration_indptr = model["iteration_indptr"]

    start = iteration_indptr[index]
    end = iteration_indptr[index + 1]
    n_removed = end - start

    del trees[start:end]
    del tree_info[start:end]

    del iteration_indptr[index]

    for i in range(index, len(iteration_indptr)):
        iteration_indptr[i] -= n_removed

    num_parallel_tree = 9

    # Rebuild iteration_indptr safely
    iteration_indptr = [i * num_parallel_tree for i in range(len(trees) // num_parallel_tree + 1)]

    model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(trees))
    model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"] = str(num_parallel_tree)
    model_json["learner"]["gradient_booster"]["model"]["trees"] = trees
    model_json["learner"]["gradient_booster"]["model"]["tree_info"] = tree_info
    model_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = iteration_indptr
    
    temp_model = xgb.Booster()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_json:
        json.dump(model_json, tmp_json, indent=2, allow_nan=False)
        tmp_json.flush()
        #with open(tmp_json.name, "r") as f:
            #json_data = f.read()
        try:
            temp_model.load_model(tmp_json.name)
            print("Model loaded successfully.")
        except xgb.core.XGBoostError as e:
            print(f"Error loading model: {e}")

def predict_tree(tree, data_point):#Pass tree and the row as parameters
    node = tree
    
    while "leaf" not in node:
        split_feature = int(node['split'])
        split_value = node['split_condition']
        
        feature_value = data_point[split_feature]
        
        if pd.isna(feature_value):
            node = next(child for child in node['children'] if child['nodeid'] == node['missing'])
        elif feature_value <= split_value:
            node = next(child for child in node['children'] if child['nodeid'] == node['yes'])
        else:
            node = next(child for child in node['children'] if child['nodeid'] == node['no'])
    
    return node['leaf']
