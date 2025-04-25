# Pre- and Post-Training Pruning of Machine Learning Tree Boosting Model XGBoost for Classifying Stars with Sparse Photometric Data
A Python-based implementation of pre-training regularization-based pruning and post-training tree-based pruning with the aim to improve performance metrics at minimal cost to accuracy and macro-F1 scores, utilizing an imbalanced multi-label dataset and XGBoost.
## Features
 - Manual inference on XGBoost models
 - SHAP-based feature engineering
 - Hyperparameter optimization and graphing of results
 - Functions for graphing tree details and photometric samples
 - Function for querying XGBoost model to find trees containing specified features
## Project Structure
 - `explore_commons.py`: Data loading and preprocessing
 - `feature_engineering_main.py`: Modified main file for SHAP-based feature engineering
 - `hyperparameter_graphs.py`: Functions for converting performance and accuracy output arrays to graphs
 - `main.py`: Original main file without modifications other than supporting code to record performance metrics
 - `manual_inference_main.py`: Modified main file for manual inference and tree-based pruning
 - `misc.py`: Utility functions for metrics and FLOPs
 - `optimize_hyperparameters_main.py`: Modified main file to test different hyperparameter configurations
 - `photometric_graph.py`: Functions for visualizing samples' photometric data
 - `ptoframework.py`: Functions supporting post-training optimization
 - `shap_routines.py`: Functions for extracting and visualizing SHAP feature importance data
 - `tree_querying.py`: Function for querying XGBoost model to find trees containing specified features
 - `tree_structure.py`: Functions for converting model dump into array format and visualizing trees in binary tree graphs
