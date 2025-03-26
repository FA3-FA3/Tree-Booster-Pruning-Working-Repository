def train_estimate_flops(n_samples, n_features, max_depth, n_estimators, gamma, min_child_weight, subsample):
    effective_samples = n_samples * subsample
    pruning_factor = 1 / (1 + gamma)
    flops_per_tree = effective_samples * n_features * max_depth * pruning_factor
    complexity_factor = 1 / (1 + min_child_weight)
    flops_per_tree *= complexity_factor
    total_flops = n_estimators * flops_per_tree
    return total_flops

def infer_estimate_flops(n_samples, max_depth, n_estimators, gamma):
    pruning_factor = 1 / (1 + gamma)
    total_flops = n_samples * n_estimators * max_depth * pruning_factor
    return total_flops

def memory_stats(memory_snapshots):
    memory_differences = [memory_snapshots[i] - memory_snapshots[i - 1] for i in range(len(memory_snapshots) - 1, 0, -1)]
    av_memory = sum(memory_differences) / len(memory_differences)
    max_memory = max(memory_differences)
    return av_memory, max_memory

def export_to_txt(df):
    with open("df_columns.txt", "w") as f:
        for col in df.columns:
            f.write(str(col) + "\n")  # Convert tuple to string and write each on a new line
