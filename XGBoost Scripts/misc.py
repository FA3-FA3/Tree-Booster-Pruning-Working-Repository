def train_estimate_flops(n_samples, n_features, max_depth, n_estimators):
    flops_per_tree = n_samples * n_features * max_depth
    total_flops = n_estimators * flops_per_tree
    return total_flops

def infer_estimate_flops(n_samples, max_depth, n_estimators):
    total_flops = n_samples * n_estimators * max_depth
    return total_flops

def memory_stats(memory_snapshots):
    memory_differences = [memory_snapshots[i] - memory_snapshots[i - 1] for i in range(len(memory_snapshots) - 1, 0, -1)]
    av_memory = sum(memory_snapshots) / len(memory_snapshots)
    max_memory = max(memory_snapshots)
    return av_memory, max_memory