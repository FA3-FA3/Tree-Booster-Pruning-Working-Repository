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
import tree_structure as ts

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
#pickle.dump(mlb, open(f'LabelEncoder.pkl', "wb"))


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
    
    
    
    #Feaature 65 trees 907 to 3337
    #Feature 64 trees 2144 to 3341
    #Feature 66 trees 970 to 2770
    #Feature 40 trees 511 to 1726
    #Feature 35 pt2 trees 2140 to 3445
    #Good^ macroF1=0.716298599030496
    #INDICIES OF TREES TO PRUNE
    remove_trees = {907, 1753, 1870, 1906, 2131, 2194, 2500, 2743, 3337, 
                    2144, 2783, 2905, 3341, 
                    970, 1114, 1271, 1303, 2221, 2770, 
                    511, 547, 601, 961, 1024, 1366, 1726, 
                    2140, 2149, 2196, 2296, 2446, 2473, 2554, 2772, 2779, 3445, 
                    }
    
    print("Inference started")
    memory_snapshots_infer = []
    
    trees = model.get_booster().get_dump(dump_format='json')
    n_classes = 9
    trees_per_class = len(trees) // n_classes
    
    class_trees = [
        [json.loads(trees[i]) for i in range(cls_idx, len(trees), n_classes) if i not in remove_trees]
        for cls_idx in range(n_classes)
        ]
    
    raw_outputs = np.zeros((len(X_test), n_classes))
    
    start1 = timer()
    tracemalloc.clear_traces()
    tracemalloc.start()
    
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])

    for cls_idx, cls_tree_list in enumerate(class_trees):
        for j, tree_json in enumerate(cls_tree_list):
            print(cls_idx," ", j)
            for i, x in enumerate(X_test.to_numpy()):
                raw_outputs[i, cls_idx] += pto.predict_tree(tree_json, x)
    
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])
    
    y_pred_prob = 1 / (1 + np.exp(-raw_outputs))
    
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])
    
    print("Calculating probabilities")
    
    for i in range(len(y_pred)):
        #print(f"i: {i}, y_pred_prob[i] shape: {y_pred_prob[i].shape}, values: {y_pred_prob[i]}")
        if np.sum(y_pred[i]) == 0:
            y_pred[i, np.argmax(y_pred_prob[i])] = 1
            
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])
    
    end1 = timer()
    
    trees_array = ts.model_dump_to_array_linear(model)
    tree_lengths = []
    for i in range(len(trees_array)):
        tree_lengths.append(len(trees_array[i]))
    avg_tree_dep = sum(tree_lengths) / len(tree_lengths)
    
    
    flops_infer = misc.manual_infer_estimate_flops(X_test.shape[0], len(class_trees), len(class_trees[0]), avg_tree_dep)
    av_mem_infer, max_mem_infer = misc.memory_stats(memory_snapshots_infer)
    
    output_text = (
    f"Columns Removed: {remove_trees}\n"
    f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n"
    f"Macro F1 Score: {metrics.f1_score(y_pred, y_test, average='macro')}\n"
    f"Classification Report:\n{metrics.classification_report(y_test, y_pred)}\n"
    f"Time taken(infer): {end1 - start1}\n"
    f"FLOPs(infer): {flops_infer:.2f}\n"
    f"Av Memory Usage(infer): {av_mem_infer / (1024 ** 2):.2f} MB\n"
    f"Max Memory Usage(infer): {max_mem_infer / (1024 ** 2):.2f} MB\n"
)
    print(output_text)
    
    output_array = [metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_pred, y_test, average='macro'), end1 - start1, flops_infer, av_mem_infer / (1024 ** 2), max_mem_infer / (1024 ** 2)]
    
    with open(f'results/manual_opt_xgb_results.txt', 'a') as output_file:
        output_file.write(output_text + '\n')
    
    with open(f'results/manual_opt_xgb_results_arrays.txt', 'a') as output_file:
        json.dump(output_array, output_file)
        output_file.write('\n')