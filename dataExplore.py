'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataPrep as dp
import random

# Generate timestamps
#timestamps = [random.randint(0, 2200) for i in range(2200)]
timestamps = [i for i in range(2200)]

# Create DataFrame
rating_df = pd.DataFrame({'timestamp': timestamps})
rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
_start = rating_df['timestamp'].min()
_end = rating_df['timestamp'].max()

print(f'min:{_start}, max:{_end}')

_total_dist = _end - _start 

_dist_unit = 1 # one day
 # hyperparameter that defines time distance weight
exp_beta = 0.015

_bias = 0 #0.0001
    
_beta = 0.05
rating_df['u_abs_decay_linear'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), 1) # linear
rating_df['u_abs_decay_log'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), _beta) # log
rating_df['u_abs_decay_recip'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), 1/_beta) # reciprocal
rating_df['u_abs_decay_exp'] = _bias + np.exp(-exp_beta * (rating_df['timestamp'] - _start) / _dist_unit) # exp

print(rating_df['u_abs_decay_exp'][1:25])

sorted_values = sorted(rating_df['u_abs_decay_linear'])
#plt.plot(sorted_values, label = 'linear')
    
sorted_values = sorted(rating_df['u_abs_decay_log'])
plt.plot(sorted_values, label = 'log')

sorted_values = sorted(rating_df['u_abs_decay_recip'])
#plt.plot(sorted_values, label = 'recip')

sorted_values = sorted(rating_df['u_abs_decay_exp'])
#plt.yscale('log')  # Set the y-axis to a log scale
plt.plot(sorted_values, label = 'exp')

# Add labels and title
plt.xlabel('Index (after sorting)')
plt.ylabel('u_abs_decay')
plt.title('Sorted Plot of u_abs_decay')
# Add legend
plt.legend(loc='upper left')

# Set the y-axis limits to be between 1 and 1.1
#plt.ylim(0, 5e-44)
#plt.xlim(2000, 2200)


# Show the plot
plt.show()

# %%
