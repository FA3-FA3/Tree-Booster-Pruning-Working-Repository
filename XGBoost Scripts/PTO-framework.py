import pickle
import xgboost as xgb
import json
import os

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


with open(f"models/{model_name}.pkl", "rb") as model_file:
    model = pickle.load(model_file)

booster = model.get_booster()

json_str = booster.save_raw(raw_format='json').decode('utf-8')
model_json = json.loads(json_str)

all_trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
all_trees_info = model_json["learner"]["gradient_booster"]["model"]["tree_info"]

index = len(all_trees) - 1

if index < 0 or index >= len(all_trees):
    raise ValueError(f"Tree index {index} is out of bounds. Model has {len(all_trees)} trees.")

del all_trees[index]
del all_trees_info[index]

model_json["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(all_trees))

model_json["learner"]["gradient_booster"]["model"]["iteration_indptr"] = list(range(len(all_trees) + 1))


temp_path = "temp/temp_model.json"
with open(temp_path, "w") as f:
    json.dump(model_json, f, indent=2)

new_booster = xgb.Booster()


try:
    new_booster.load_model(temp_path)
except xgb.core.XGBoostError as e:
    print("Error loading modified JSON. Possible invalid structure.")
    raise e

new_model = type(model)()
new_model._Booster = new_booster
if hasattr(model, "_le"):
    new_model._le = model._le

for attr in ["n_features_in_", "feature_names_in_"]:
    if hasattr(model, attr):
        try:
            setattr(new_model, attr, getattr(model, attr))
        except AttributeError:
            print(f"Could not set attribute: {attr}, skipping.")

with open(f"models/{model_name}_opt{index}.pkl", "wb") as file:
    pickle.dump(new_model, file)