import numpy as np

# Values from the table
results = np.array([
    [0.439, 0.506, 0.87, 0.317, 0.388, 0.698],
    [0.502, 0.59, 0.878, 0.322, 0.448, 0.795],
    [0.531, 0.62, 0.895, 0.345, 0.451, 0.795],
    [0.49, 0.61, 0.909, 0.358, 0.435, 0.761],
    [0.495, 0.608, 0.918, 0.368, 0.477, 0.77]
])

ct_mean = np.sum(results[:, :3])/15
mri_mean = np.sum(results[:, 3:])/15
ct_mri_comp = ct_mean/mri_mean
ct_1_mean = np.sum(results[:, 0])/5
ct_2_mean = np.sum(results[:, 1])/5
ct_8_mean = np.sum(results[:, 2])/5
mri_1_mean = np.sum(results[:, 3])/5
mri_2_mean = np.sum(results[:, 4])/5
mri_8_mean = np.sum(results[:, 5])/5
global_domain_mean = np.sum(results[2, :])/6
combined_mean = np.sum(results[4, :])/6
local_mean = np.sum(results[3, :])/6
global_threshold_mean = np.sum(results[1, :])/6
random_mean = np.sum(results[0, :])/6
not_random_mean = np.sum(results[1:, :])/24

print(global_domain_mean, combined_mean, local_mean, global_threshold_mean, random_mean)
print(global_domain_mean/local_mean, global_domain_mean/global_threshold_mean)