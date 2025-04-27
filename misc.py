import json

#Estimate FLOPS for training
def train_estimate_flops(n_samples, n_features, max_depth, n_estimators, gamma, min_child_weight, subsample):
    effective_samples = n_samples * subsample
    pruning_factor = 1 / (1 + gamma)
    flops_per_tree = effective_samples * n_features * max_depth * pruning_factor
    complexity_factor = 1 / (1 + min_child_weight)
    flops_per_tree *= complexity_factor
    total_flops = n_estimators * flops_per_tree
    return total_flops

#Estimate FLOPS for inference
def infer_estimate_flops(n_samples, max_depth, n_estimators, gamma):
    pruning_factor = 1 / (1 + gamma)
    total_flops = n_samples * n_estimators * max_depth * pruning_factor
    return total_flops

#Estimate flops for manual inference
def manual_infer_estimate_flops(n_samples, n_classes, trees_per_class, avg_tree_depth, remove_trees=None, gamma=0.0):
    remove_trees = remove_trees or set()
    pruning_factor = 1 / (1 + gamma)
    ops_per_tree = avg_tree_depth + 1
    total_flops = 0.0
    
    for cls_idx in range(n_classes):
        removed_for_class = sum(1 for t in remove_trees if (t % n_classes) == cls_idx)
        effective_trees = trees_per_class - removed_for_class
        flops_cls = n_samples * effective_trees * ops_per_tree * pruning_factor
        total_flops += flops_cls
    return total_flops

#Calculate average and max memory from memory snapshots array
def memory_stats(memory_snapshots):
    memory_differences = [memory_snapshots[i] - memory_snapshots[i - 1] for i in range(len(memory_snapshots) - 1, 0, -1)]
    av_memory = sum(memory_differences) / len(memory_differences)
    max_memory = max(memory_differences)
    return av_memory, max_memory

#Export df to txt
def export_to_txt(df):
    with open("df_columns.txt", "w") as f:
        for col in df.columns:
            f.write(str(col) + "\n")  # Convert tuple to string and write each on a new line