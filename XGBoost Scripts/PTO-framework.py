import pickle
import xgboost as xgb
import json
import tempfile

def prune_tree_by_index(model_name, index):
    with open(f"models/{model_name}.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    booster = model.get_booster()

    
    num_trees = booster.num_boosted_rounds()
    if index < 0 or index >= num_trees:
        raise ValueError(f"Tree index {index} is out of bounds. Model has {num_trees} trees.")
    
    all_trees = booster.get_dump()
    keep_trees = [tree for i, tree in enumerate(all_trees) if i != index]
    
    new_booster = xgb.booster()
    new_booster.load_model_from_string("\n".join(keep_trees))
    
    new_model = type(model)()
    new_model._Booster = new_booster
    new_model._le = model._le if hasattr(model, "_le") else None
    
    for attr in ["n_features_in_", "feature_names_in_"]:
        if hasattr(model, attr):
            setattr(new_model, attr, getattr(model, attr))
    
    with open(f"models/{model_name}_opt{index}.pkl", "wb") as file:
        pickle.dump(new_model, file)


#prune_tree_by_index("xgb_Full", 20)

model_name = "xgb_Full"
index = 20

with open(f"models/{model_name}.pkl", "rb") as model_file:
    model = pickle.load(model_file)

booster = model.get_booster()

json_str = booster.save_raw(raw_format='json').decode('utf-8')
model_json = json.loads(json_str)

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
    
    with open(tmp_json.name, "r") as f:
        config_str = f.read()
    temp_model.load_model(bytearray(json_str, 'utf-8'))

with open(f"models/{model_name}_opt{index}.pkl", "wb") as file:
    pickle.dump(temp_model, file)