#%%
import numpy as np

# Example time distances
time_distances = np.array([1, 2, 5, 10, 20, 100, 1000, 10000])

# Min-Max Scaling
min_time_distance = np.min(time_distances)
max_time_distance = np.max(time_distances)
normalized_time_distances = (time_distances - min_time_distance) / (max_time_distance - min_time_distance)

# Z-Score Standardization
mean_time_distance = np.mean(time_distances)
std_time_distance = np.std(time_distances)
standardized_time_distances = (time_distances - mean_time_distance) / std_time_distance

# Log Transformation
log_time_distances = np.log1p(time_distances)

# Robust Scaling
median_time_distance = np.median(time_distances)
iqr_time_distance = np.percentile(time_distances, 75) - np.percentile(time_distances, 25)
robust_scaled_time_distances = (time_distances - median_time_distance) / iqr_time_distance

# Clipping
lower_bound = np.percentile(time_distances, 1)
upper_bound = np.percentile(time_distances, 99)
clipped_time_distances = np.clip(time_distances, lower_bound, upper_bound)

# Display results
print("Normalized:", normalized_time_distances)
print("Standardized:", standardized_time_distances)
print("Log Transformed:", log_time_distances)
print("Robust Scaled:", robust_scaled_time_distances)
print("Clipped:", clipped_time_distances)

# %%
