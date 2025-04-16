import pickle
import xgboost as xgb

with open(f'models/xgb_Full_opt20.pkl', 'rb') as model_file:
    booster = pickle.load(model_file)

with open(f'tuning_results_5foldcv_2000_iter_ms_augmented/output_Full.txt', "r") as fp:
    server_outputs = fp.readlines()
    package_params = eval(server_outputs[1])

#Update n_estimators for n iterations removed
model = xgb.XGBClassifier(**package_params, n_estimators = 499, eta=0.1, tree_method='hist', random_state=1606421)

model._Booster = booster


with open("LabelEncoder.pkl", "rb") as f:
    mlb = pickle.load(f)



print(len(booster.get_dump()))