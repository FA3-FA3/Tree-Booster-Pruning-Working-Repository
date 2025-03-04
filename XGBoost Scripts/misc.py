def train_estimate_flops(n_samples, n_features, max_depth, n_estimators):
    flops_per_tree = n_samples * n_features * max_depth
    total_flops = n_estimators * flops_per_tree
    return total_flops

def infer_estimate_flops(n_samples, max_depth, n_estimators):
    total_flops = n_samples * n_estimators * max_depth
    return total_flops