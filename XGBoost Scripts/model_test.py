import pickle

with open(f'models/xgb_Full_opt20.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

print(len(model.get_booster().get_dump()))