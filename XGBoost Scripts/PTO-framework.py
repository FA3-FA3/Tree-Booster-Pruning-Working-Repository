import pickle
import xgboost as xgb
import json

def model_to_json(model_name):#name xgb_Full
    with open(f'models/{model_name}.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    model.save_model(f"jsonmodels/{model_name}.json")

def prune_tree_by_index(model_name, index):
    #model_to_json(f"{model_name}")
    with open(f"jsonmodels/{model_name}.json", "r") as model_file:
        model = json.load(model_file)
    all_trees = model["learner"]["gradient_booster"]["model"]["trees"]
    trees_info = model["learner"]["gradient_booster"]["model"]["tree_info"]
    
    keep_trees = [tree for i, tree in enumerate(all_trees) if i != index]
    keep_trees_info = [info for i, info in enumerate(trees_info) if i != index]
    
    model["learner"]["gradient_booster"]["model"]["trees"] = keep_trees
    model["learner"]["gradient_booster"]["model"]["tree_info"] = keep_trees_info
    model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(keep_trees))
    
    with open(f"jsonmodels/{model_name}_opt{index}.json", "w") as file:
        json.dump(model, file)
    
    #SAVE AS PKL
    booster = xgb.Booster()
    booster.load_model(f"jsonmodels/{model_name}.json")
    model = xgb.XGBClassifier()
    model._Booster = booster
    with open(f'models/{model_name}_opt{index}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

def json_saveas_pkl(model_name, index):
    booster = xgb.Booster()
    booster.load_model(f"jsonmodels/{model_name}.json")
    model = xgb.XGBClassifier()
    model._Booster = booster
    with open(f'models/{model_name}_opt{index}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)


#model_to_json(f"{name}")
prune_tree_by_index("xgb_Full", 20)
#json_saveas_pkl(f"{name}_opt20", 20)
