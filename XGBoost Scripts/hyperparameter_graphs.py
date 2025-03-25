import json
import matplotlib.pyplot as plt

def seperate_arrays(arrays):
    opt_vals = [None] * len(arrays)
    accuracy_vals = [None] * len(arrays)
    macrof1_vals = [None] * len(arrays)
    train_times = [None] * len(arrays)
    infer_times = [None] * len(arrays)
    train_flops = [None] * len(arrays)
    infer_flops = [None] * len(arrays)
    train_avmem = [None] * len(arrays)
    train_maxmem = [None] * len(arrays)
    infer_avmem = [None] * len(arrays)
    infer_maxmem = [None] * len(arrays)
    for i in range(len(arrays)):
        opt_vals[i] = arrays[i][1]
        accuracy_vals[i] = arrays[i][2]
        macrof1_vals[i] = arrays[i][3]
        train_times[i] = arrays[i][4]
        infer_times[i] = arrays[i][5]
        train_flops[i] = arrays[i][6]
        infer_flops[i] = arrays[i][7]
        train_avmem[i] = arrays[i][8]
        train_maxmem[i] = arrays[i][9]
        infer_avmem[i] = arrays[i][10]
        infer_maxmem[i] = arrays[i][11]
    return opt_vals, accuracy_vals, macrof1_vals, train_times, infer_times, train_flops, infer_flops, train_avmem, train_maxmem, infer_avmem, infer_maxmem

def arrays_to_graphs(arrays, hyperparameter_str):
    opt_vals, accuracy_vals, macrof1_vals, train_times, infer_times, train_flops, infer_flops, train_avmem, train_maxmem, infer_avmem, infer_maxmem = seperate_arrays(arrays)
    #accuracy and macrof1 go together
    #avmem and maxmem go together
    plt.figure(figsize=(8, 5))

    #accuracy and macro F1
    plt.plot(opt_vals, accuracy_vals, label="Accuracy", marker="o")
    

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("Score")
    plt.title(f"Accuracy vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/accuracy.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(opt_vals, macrof1_vals, label="Macro F1 Score", marker="o")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("Score")
    plt.title(f"Macro F1 Score vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/macrof1.png", format="png", dpi=300)
    

    plt.figure(figsize=(8, 5))

    #train times
    plt.plot(opt_vals, train_times, label="Execution Time", marker="o")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("Train Time(s)")
    plt.title(f"Train Time vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/traintimes.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))
    
    #infer times
    plt.plot(opt_vals, infer_times, label="Execution Time", marker="o")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("Inference Time(s)")
    plt.title(f"Inference Time vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/inferencetimes.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))
    
    #train flops
    plt.plot(opt_vals, train_flops, label="FLOPs", marker="o")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("FLOPs")
    plt.title(f"Train FLOPs vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/trainflops.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))
    
    #infer times
    plt.plot(opt_vals, infer_flops, label="FLOPs", marker="o")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("FLOPs")
    plt.title(f"Inference FLOPs vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/inferflops.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))

    #accuracy and macro F1
    plt.plot(opt_vals, train_avmem, label="Average Memory Usage", marker="o")
    plt.plot(opt_vals, train_maxmem, label="Maximum Memory Usage", marker="s")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("MB")
    plt.title(f"Train Memory Usage vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/trainmemory.png", format="png", dpi=300)
    
    plt.figure(figsize=(8, 5))

    #accuracy and macro F1
    plt.plot(opt_vals, infer_avmem, label="Average Memory Usage", marker="o")
    plt.plot(opt_vals, infer_maxmem, label="Maximum Memory Usage", marker="s")

    # Labels and title
    plt.xlabel(f"{hyperparameter_str} Values")
    plt.ylabel("MB")
    plt.title(f"Inference Memory Usage vs {hyperparameter_str} Values")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{hyperparameter_str}/infermemory.png", format="png", dpi=300)


with open('results/gamma_opt_xgb_Full_results_arrays.txt', 'r') as input_file:
    lines = input_file.readlines()

gamma_arrays = [json.loads(line) for line in lines]

with open('results/max_depth_opt_xgb_Full_results_arrays.txt', 'r') as input_file:
    lines = input_file.readlines()

max_depth_arrays = [json.loads(line) for line in lines]

with open('results/min_child_weight_opt_xgb_Full_results_arrays.txt', 'r') as input_file:
    lines = input_file.readlines()

min_child_weight_arrays = [json.loads(line) for line in lines]

with open('results/subsample_opt_xgb_Full_results_arrays.txt', 'r') as input_file:
    lines = input_file.readlines()

subsample_arrays = [json.loads(line) for line in lines]

arrays_to_graphs(gamma_arrays, 'gamma')
arrays_to_graphs(max_depth_arrays, 'max_depth')
arrays_to_graphs(min_child_weight_arrays, 'min_child_weight')
arrays_to_graphs(subsample_arrays, 'subsample')