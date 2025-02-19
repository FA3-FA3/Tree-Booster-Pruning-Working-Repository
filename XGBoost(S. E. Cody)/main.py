from timeit import default_timer as timer
import explore_commons
import numpy as np
import pandas as pd
from sklearn import metrics, utils, model_selection
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
#from XGBoost_Weighted import XGBClassifier_w

# Set seed for reproducibility
np.random.seed(1606421)

idx = pd.IndexSlice

# Read to dataframe
df, classes, header = explore_commons.read_data('data/ms_augmented/df_plus_ms.dat',
                                                'data/ms_augmented/classes_plus_ms.dat')

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

idx = pd.IndexSlice
dereddened = df.loc[:, idx["Dereddened", :, "Value", :, :, :, :]]
model = df.loc[:, idx["Model", :, "Value", :, :, :, :]]
phot_error = df.loc[:, idx["Photometry", :, "Error", :, :, :, :]]

raw_values = df.loc[:, idx["Photometry", :, "Value", :, :, :, :]]
ten_percent = raw_values.multiply(0.1, axis=1)
tmp_df = pd.concat([ten_percent, phot_error], axis=1)
sub_error = pd.DataFrame()
for i, col in enumerate(ten_percent.columns):
    tmp_indexer = list(col)
    tmp_indexer[2] = 'Error'
    sub_error[i] = np.sqrt(tmp_df[[col, tmp_indexer]] ** 2).sum(axis=1).to_numpy()
# transform_spectra
dereddened.columns = list(range(len(dereddened.columns)))
model.columns = list(range(len(model.columns)))
phot_error = pd.DataFrame(sub_error)
phot_error.columns = list(range(len(phot_error.columns)))
phot_error.replace(['0', 0], np.nan, inplace=True)
#new_error = dereddened.sub(model, fill_value=0, axis=0).div(phot_error.to_numpy(), axis=0)
transformed_spectra = np.log(dereddened.div(model, fill_value=1, axis=0)).div(np.log(phot_error.to_numpy() + 1), axis=0)


'''
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

DataPackages = {'Base': BasePackage, 'Bias': BiasPackage, 'Spectra': SpectraPackage, 'Physics': PhysicsPackage,
                'Full': FullPackage}
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

    start = timer()

    # Get train/test dataframes
    X_train = dat_pack[1].iloc[train_indices]
    X_test = dat_pack[1].iloc[test_indices]

    # Remove multiheader information
    X_train.columns = range(X_train.shape[1])
    X_test.columns = range(X_test.shape[1])
    

    w = np.array([package_params.pop(w_i) for w_i in weight_params])
    print(w)
    model = xgb.XGBClassifier(**package_params, n_estimators = 500, eta=0.1, tree_method='hist', random_state=1606421)
    print("Fitting now")

    # Set class weights and fit model
    model.fit(X_train, y_train, sample_weight=[np.sum(w * i) for i in y_train])

    # Make predictions
    y_pred = model.predict(X_test)

    # No prediction is invalid so choose highest probability in that case
    y_pred_prob = model.predict_proba(X_train)
    for i, j in enumerate(y_pred):
        if sum(j) < 1:
            j[np.argmax(y_pred_prob[i])] = 1

    end = timer()

    print(f'''''''
{str(metrics.accuracy_score(y_test, y_pred))}
{str(metrics.f1_score(y_pred, y_test, average='macro'))}
{str(metrics.classification_report(y_test, y_pred))}
Time taken: {str(end - start)}'''#)