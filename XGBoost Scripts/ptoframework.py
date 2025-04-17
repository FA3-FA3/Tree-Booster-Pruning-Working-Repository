import pickle
import xgboost as xgb
import json
import tempfile
import pprint
import pandas as pd

def prune_dump_edit_reload(model_name, index):#Broken
    with open(f"models/{model_name}.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    booster = model.get_booster()

    json_str = booster.save_raw(raw_format='json').decode('utf-8')#Loading from here is valid, issue is after this
    model_json = json.loads(json_str)

    objective = model_json.get("learner", {}).get("objective", {}).get("name", "Unknown")
    print(f"Current model objective: {objective}")

        # If the objective is binary, change it to multi-class
    if objective == "binary:logistic":
        print("Changing model objective to multi:softmax for multi-class classification")
        model_json["learner"]["objective"]["name"] = "multi:softmax"
        model_json["learner"]["learner_model_param"]["num_class"] = '9'

    model = model_json["learner"]["gradient_booster"]["model"]
    trees = model["trees"]
    tree_info = model["tree_info"]
    iteration_indptr = model["iteration_indptr"]


    start = iteration_indptr[index]
    end = iteration_indptr[index + 1]
    n_removed = end - start

    del trees[start:end]
    del tree_info[start:end]

    for i in range(index + 1, len(iteration_indptr)):
        iteration_indptr[i] -= n_removed

    del iteration_indptr[index + 1]

    model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(trees))
    model_json["learner"]["gradient_booster"]["model"]["trees"] = trees
    model_json["learner"]["gradient_booster"]["model"]["tree_info"] = tree_info
    model_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = iteration_indptr


    temp_model = xgb.Booster()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_json:
        json.dump(model_json, tmp_json, indent=2, allow_nan=False)
        tmp_json.flush()
        #with open(tmp_json.name, "r") as f:
            #json_data = f.read()
        temp_model.load_model(tmp_json.name)

    with open(f"models/{model_name}_opt{index}.pkl", "wb") as file:
        pickle.dump(temp_model, file)

def predict_tree(tree, data_point):
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

model_name = "xgb_Full"
index = 20

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

print("Before pruning:")
print(f"Number of trees: {len(trees)}")
print(f"Length of tree_info: {len(tree_info)}")
print(f"Length of iteration_indptr: {len(iteration_indptr)}")
print("num_trees: ", model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])

start = iteration_indptr[index]
end = iteration_indptr[index + 1]
n_removed = end - start

del trees[start:end]
del tree_info[start:end]

del iteration_indptr[index]

for i in range(index, len(iteration_indptr)):
    iteration_indptr[i] -= n_removed


model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(trees))
#model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"]
model_json["learner"]["gradient_booster"]["model"]["trees"] = trees
model_json["learner"]["gradient_booster"]["model"]["tree_info"] = tree_info
model_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = iteration_indptr

print("After pruning:")
print(f"Number of trees: {len(trees)}")
print(f"Length of tree_info: {len(tree_info)}")
print(f"Length of iteration_indptr: {len(iteration_indptr)}")
print("num_trees: ", model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])

#print("Iteration Indptr after pruning:", iteration_indptr)
print("Last entry of iteration_indptr:", iteration_indptr[-1])

assert iteration_indptr[-1] == len(trees)
assert len(tree_info) == len(trees)
assert all(isinstance(i, int) for i in iteration_indptr)
assert all(isinstance(i, int) for i in tree_info)



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

