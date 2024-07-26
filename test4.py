#%%
import torch
import dataPrep as dp


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

# Compare results
for key in result1:
    print(f"Comparing {key}:")
    if torch.equal(result1[key], result2[key]):
        print(f"{key} matches.")
    else:
        print(f"{key} does not match.")


print(result1)
print('----------------------')
print(result2)
