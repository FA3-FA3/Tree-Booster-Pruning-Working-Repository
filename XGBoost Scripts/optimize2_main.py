from timeit import default_timer as timer
import explore_commons
import numpy as np
import pandas as pd
from sklearn import metrics, utils, model_selection
#import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
#from XGBoost_Weighted import XGBClassifier_w
import misc
import tracemalloc
import ast
import json
import ptoframework as pto

# Set seed for reproducibility
np.random.seed(1606421)

idx = pd.IndexSlice

# Read to dataframe
df, classes, header = explore_commons.read_data('Data/df_plus_ms.dat',
                                                'Data/classes_plus_ms.dat')

# Preprocess df
df, classes = explore_commons.process_data_frame_2(df, classes, header[2:])

# Merge duplicate Wavelengths
df = explore_commons.merge_duplicate_wavelength_cols(df)

# Galactic Coordinate Conversion
df = explore_commons.convert_to_galactic_coords(df)

# Traditionally Most Relevant Variables Only
BasePackage = df.loc[:, idx["Fitted", ["Teff", "Lum"], "Value", :, :, :, :]]

# Physically Irrelevant Variables Only
BiasPackage = df.loc[:, idx["Adopted", ["RA", "Dec", "PMRA", "PMDec", "Distance"], "Value", :, :, :, :]]

# Spectral Measurements Only
SpectraPackage = pd.concat([df.loc[:, idx["Photometry", :, ["Error"], :, :, :, :]],
                            df.loc[:, idx[["Model", "Dereddened"], :, "Value", :, :, :, :]]], axis=1)


# Transformed Spectra
SpectraPackage = explore_commons.transform_Spectra(df, corrected_error_2=True)

# All Physically Relevant Variables
PhysicsPackage = pd.concat([BasePackage, df.loc[:, idx["Adopted", ["E(B-V)", 'logg', '[Fe/H]'], "Value", :, :, :, :]],
                            df.loc[:, idx["Ancillary", "Tspec", "Value", :, :, :, :]], SpectraPackage], axis=1)

# All Relevant Variables
FullPackage = pd.concat([BiasPackage, PhysicsPackage], axis=1)

all_models = []

# Split Indices
train_indices, test_indices = explore_commons.custom_train_test_split_2(classes, test_size=0.2, min_class_instances=40)

#DataPackages = {'Base': BasePackage, 'Bias': BiasPackage, 'Spectra': SpectraPackage, 'Physics': PhysicsPackage, 'Full': FullPackage}
DataPackages = {'Full': FullPackage}
weight_params = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8']
y_test = classes.iloc[test_indices]
y_train = classes.iloc[train_indices]

y_train, mlb = explore_commons.classes_to_multilabel([i for i in y_train["class"].str.split(", ")])
y_test = mlb.transform([i for i in y_test["class"].str.split(", ")])
pickle.dump(mlb, open(f'LabelEncoder.pkl', "wb"))


index = 20

for dat_pack in DataPackages.items():
    with open(f'models/xgb_Full.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    
    
    # Get train/test dataframes
    X_train = dat_pack[1].iloc[train_indices]
    X_test = dat_pack[1].iloc[test_indices]
    
    # Remove multiheader information
    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])
    
    X_train[10] = X_train[10].astype(float)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test[10] = X_test[10].astype(float)
    
    
    #INDICIES OF TREES TO PRUNE
    remove_trees = {20, 21}
        
    trees = model.get_booster().get_dump(dump_format='json')
    n_classes = 9
    trees_per_class = len(trees) // n_classes
    
    class_trees = [
        [json.loads(trees[i]) for i in range(cls_idx, len(trees), n_classes) if i not in remove_trees]
        for cls_idx in range(n_classes)
        ]
    
    raw_outputs = np.zeros((len(X_test), n_classes))
    
    for cls_idx, cls_tree_list in enumerate(class_trees):
        for tree_json in cls_tree_list:
            for i, x in enumerate(X_test.to_numpy()):
                raw_outputs[i, cls_idx] += pto.predict_tree(tree_json, x)
    
    y_pred_prob = 1 / (1 + np.exp(-raw_outputs))
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    for i in range(len(y_pred)):
        #print(f"i: {i}, y_pred_prob[i] shape: {y_pred_prob[i].shape}, values: {y_pred_prob[i]}")
        if np.sum(y_pred[i]) == 0:
            y_pred[i, np.argmax(y_pred_prob[i])] = 1
    
    output_text = (
    f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n"
    f"Macro F1 Score: {metrics.f1_score(y_pred, y_test, average='macro')}\n"
    f"Classification Report:\n{metrics.classification_report(y_test, y_pred)}\n"
)
    print(output_text)