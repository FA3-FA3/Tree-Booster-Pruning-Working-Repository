import pickle
import tree_structure as ms

#get indicies of trees containing a feature ordered in how many times the feature appears in such a tree

def get_trees_with_feature(model, feature_index):
    tree_arrays = ms.model_dump_to_array_linear(model)
    found_indicies = []
    for i in range(len(tree_arrays)):
        found = False
        n_occurances = 0
        for j in range(len(tree_arrays[i])):
            if tree_arrays[i][j][1] == feature_index:
                if found == False:
                    found = True
                n_occurances += 1
        if found == True:
            found_indicies.append([i, n_occurances])
    found_indicies = sorted(found_indicies, key=lambda x: x[1], reverse=True)
    return found_indicies
        


with open(f'models/xgb_Full.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

print(get_trees_with_feature(model, 38))