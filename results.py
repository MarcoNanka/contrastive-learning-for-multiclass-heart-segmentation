import numpy as np

# Values from the table
results = np.array([
    [0.362, 0.464, 0.734, 0.226, 0.285, 0.395],
    [0.477, 0.5, 0.758, 0.269, 0.319, 0.439],
    [0.492, 0.565, 0.761, 0.298, 0.329, 0.434],
    [0.45, 0.523, 0.752, 0.285, 0.354, 0.445],
    [0.543, 0.574, 0.741, 0.3, 0.351, 0.455]
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

print(combined_mean, global_threshold_mean, global_domain_mean, random_mean, local_mean)
print(combined_mean/global_domain_mean, combined_mean/local_mean)