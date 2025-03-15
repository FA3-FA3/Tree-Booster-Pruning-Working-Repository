import model_structure
import pickle
import numpy as np
import explore_commons
import pandas as pd

with open(f'models/xgb_Full.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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

DataPackages = {'Full': FullPackage}

for dat_pack in DataPackages.items():
    X_train = dat_pack[1].iloc[train_indices]
    X_train.columns = range(X_train.shape[1])
    X_train[10] = X_train[10].astype(float)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print("Applying SHAP")
    
    model_shap = model_structure.shap_features(model, X_train)