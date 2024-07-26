import torch
import dataPrep as dp

# Define the functions as provided
# (Include the definitions of rmat_2_adjmat_simple and rmat_2_adjmat2 here)

# Sample test data
num_users = 5
num_items = 3
rmat_data = {
    'rmat_index': ([0, 1, 2], [0, 1, 2]),
    'rmat_values': [1.0, 2.0, 3.0],
    'rmat_ts': [0.1, 0.2, 0.3],
    'rmat_abs_t_decay': [0.01, 0.02, 0.03],
    'rmat_rel_t_decay': [0.001, 0.002, 0.003]
}

# Run both functions
result1 = dp.rmat_2_adjmat_simple(num_users, num_items, rmat_data)
result2 = dp.rmat_2_adjmat2(num_users, num_items, rmat_data)

# Extract edge indices and convert to lists of lists
edge_index_1 = result1['edge_index'].tolist()
edge_index_2 = result2['edge_index'].tolist()

print(edge_index_1)
print(edge_index_2)
# Extract the other attributes for comparison
edge_values_1 = result1['edge_values']
edge_attr_ts_1 = result1['edge_attr_ts']
edge_attr_abs_decay_1 = result1['edge_attr_abs_decay']
edge_attr_rel_decay_1 = result1['edge_attr_rel_decay']

edge_values_2 = result2['edge_values']
edge_attr_ts_2 = result2['edge_attr_ts']
edge_attr_abs_decay_2 = result2['edge_attr_abs_decay']
edge_attr_rel_decay_2 = result2['edge_attr_rel_decay']

# Function to find the index of an edge in the second edge list
def find_edge_index(edge, edge_list):
    try:
        return edge_list[0].index(edge[0]), edge_list[1].index(edge[1])
    except ValueError:
        return -1, -1

# Compare the results
mismatch_found = False
for i in range(len(edge_index_1[0])):
    edge = (edge_index_1[0][i], edge_index_1[1][i])
    idx1, idx2 = find_edge_index(edge, edge_index_2)
    if idx1 == -1 or idx2 == -1 or idx1 != idx2:
        print(f"Edge {edge} from result1 not found in result2.")
        mismatch_found = True
    else:
        if (edge_values_1[i] != edge_values_2[idx1] or
            edge_attr_ts_1[i] != edge_attr_ts_2[idx1] or
            edge_attr_abs_decay_1[i] != edge_attr_abs_decay_2[idx1] or
            edge_attr_rel_decay_1[i] != edge_attr_rel_decay_2[idx1]):
            print(f"Mismatch found for edge {edge}:")
            print(f"Values from result1: {edge_values_1[i]}, {edge_attr_ts_1[i]}, {edge_attr_abs_decay_1[i]}, {edge_attr_rel_decay_1[i]}")
            print(f"Values from result2: {edge_values_2[idx1]}, {edge_attr_ts_2[idx1]}, {edge_attr_abs_decay_2[idx1]}, {edge_attr_rel_decay_2[idx1]}")
            mismatch_found = True

if not mismatch_found:
    print("All edges and their attributes match between the two results.")
