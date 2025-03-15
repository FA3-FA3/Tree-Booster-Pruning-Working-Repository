import ast


with open(f'tuning_results_5foldcv_2000_iter_ms_augmented/output_Full.txt', "r") as fp:
    server_outputs = fp.readlines()
    
    dict1 = ast.literal_eval(server_outputs[1])
    