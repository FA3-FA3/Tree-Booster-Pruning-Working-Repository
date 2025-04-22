import shap
import numpy as np

def shap_features(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return shap_values

def shap_plot(shap_values, X_train):
    shap.summary_plot(shap_values, X_train, plot_type="bar")

def shap_features_least(shap_values, X_train):
    # For multi-class: shap_values is (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Average over samples and classes
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # shape: (n_features,)
    else:
        raise ValueError("Unexpected SHAP value shape. Expected 3D array for multi-class.")

    # Get indices of least important features
    least_important_idx = np.argsort(mean_abs_shap)[:20]

    # Subset the training data
    X_least = X_train.iloc[:, least_important_idx]

    # Plot SHAP summary bar plot for least important
    shap.summary_plot(shap_values[:, least_important_idx, :], X_least, plot_type="bar")

def shap_features_by_group(shap_values, X_train, group=0):
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # shape: (n_features,)
    else:
        raise ValueError("Unexpected SHAP value shape. Expected 3D array for multi-class.")

    # Sort features by importance (descending)
    sorted_idx = np.argsort(-mean_abs_shap)

    # Determine number of features in this group
    start = group * 20
    end = min(start + 20, len(mean_abs_shap))

    group_indices = sorted_idx[start:end]

    # Subset for plotting
    X_subset = X_train.iloc[:, group_indices]
    shap.summary_plot(shap_values[:, group_indices, :], X_subset, plot_type="bar")