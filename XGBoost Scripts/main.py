from timeit import default_timer as timer
import explore_commons
import numpy as np
import pandas as pd
from sklearn import metrics, utils, model_selection
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
#from XGBoost_Weighted import XGBClassifier_w
import misc
import tracemalloc

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
#pickle.dump(mlb, open(f'v3/models_3/LabelEncoder.pkl', "wb"))

for dat_pack in DataPackages.items():
    # Read model tuned model parameters
    with open(f'tuning_results_5foldcv_2000_iter_ms_augmented/output_{dat_pack[0]}.txt', "r") as fp:
        server_outputs = fp.readlines()
        package_params = eval(server_outputs[1])

    memory_snapshots_train = []
    memory_snapshots_infer = []

    start1 = timer()
    tracemalloc.start()
    
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])

    # Get train/test dataframes
    X_train = dat_pack[1].iloc[train_indices]
    X_test = dat_pack[1].iloc[test_indices]
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])

    # Remove multiheader information
    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])
    
    X_train[10] = X_train[10].astype(float)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test[10] = X_test[10].astype(float)
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])

    w = np.array([package_params.pop(w_i) for w_i in weight_params])
    print(w)
    model = xgb.XGBClassifier(**package_params, n_estimators = 500, eta=0.1, tree_method='hist', random_state=1606421)
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])
    
    print("Fitting now")
    
    train_snap1 = tracemalloc.take_snapshot()
    
    # Set class weights and fit model
    model.fit(X_train, y_train, sample_weight=[np.sum(w * i) for i in y_train])
    train_snap2 = tracemalloc.take_snapshot()
    memory_snapshots_train.append(tracemalloc.get_traced_memory()[0])
    
    end1 = timer()
    start2 = timer()
    
    infer_snap1 = tracemalloc.take_snapshot()
    
    # Make predictions
    y_pred = model.predict(X_test)
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])

    # No prediction is invalid so choose highest probability in that case
    y_pred_prob = model.predict_proba(X_train)
    for i, j in enumerate(y_pred):
        if sum(j) < 1:
            j[np.argmax(y_pred_prob[i])] = 1
    infer_snap2 = tracemalloc.take_snapshot()
    memory_snapshots_infer.append(tracemalloc.get_traced_memory()[0])

    end2 = timer()
    
    flops_train = misc.train_estimate_flops(len(X_train), X_train.shape[1], package_params['max_depth'], 500)
    flops_infer = misc.infer_estimate_flops(len(X_test), package_params['max_depth'], 500)
    
    av_mem_train, max_mem_train = misc.memory_stats(memory_snapshots_train)
    av_mem_infer, max_mem_infer = misc.memory_stats(memory_snapshots_infer)
    
    train_stats = train_snap2.compare_to(train_snap1, 'lineno')
    infer_stats = infer_snap2.compare_to(infer_snap1, 'lineno')

    #Output and save results
    output_text = (
    f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n"
    f"Macro F1 Score: {metrics.f1_score(y_pred, y_test, average='macro')}\n"
    f"Classification Report:\n{metrics.classification_report(y_test, y_pred)}\n"
    f"Time taken(train): {end1 - start1}\n"
    f"Time taken(infer): {end2 - start2}\n"
    f"FLOPs(train): {flops_train:.2f}\n"
    f"FLOPs(infer): {flops_infer:.2f}\n"
    f"Av Memory Usage(train): {av_mem_train / (1024 ** 2):.2f} MB\n"
    f"Max Memory Usage(train): {max_mem_train / (1024 ** 2):.2f} MB\n"
    f"Av Memory Usage(infer): {av_mem_infer / (1024 ** 2):.2f} MB\n"
    f"Max Memory Usage(infer): {max_mem_infer / (1024 ** 2):.2f} MB\n"
    f"Snapshot Stats(train): {train_stats[:10]}\n"
    f"Snapshot Stats(infer): {infer_stats[:10]}"
)
    
    print(output_text)
    with open(f'results/xgb_{dat_pack[0]}_results.txt', 'a') as output_file:
        output_file.write(output_text + '\n')
    
    #Save trained models
    with open(f'models/xgb_{dat_pack[0]}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
    model.save_model(f'models/xgb_{dat_pack[0]}.json')