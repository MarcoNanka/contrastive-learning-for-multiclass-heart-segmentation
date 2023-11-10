from data_loading import DataProcessor
import matplotlib.pyplot as plt
import numpy as np

# TODO: label names for segmentation mask
_, _, original_image_data_CT, original_label_data_CT = \
    DataProcessor.create_training_data_array(
        path_list=["../../data/entire_dataset_split_ts_tr/ct_pre/ct_train_1001_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 66, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="CT")

_, _, original_image_data_MRI, original_label_data_MRI = \
    DataProcessor.create_training_data_array(
        path_list=["../../data/entire_dataset_split_ts_tr/mr_pre/mr_train_1001_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 66, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="MRI")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.5]})
plt.subplots_adjust(top=0.8)

ct_img = original_image_data_CT[:, original_image_data_CT.shape[1] // 2, :]
ct_img = np.rot90(ct_img, k=1)
axes[0, 0].set_title('CT Scan', y=1.08, fontweight='bold')
axes[0, 0].imshow(ct_img, cmap='gray')

label_margin = 20  #

axes[0, 0].text(-label_margin, ct_img.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] + label_margin, ct_img.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] // 2, ct_img.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

ct_seg_mask = original_label_data_CT[:, original_label_data_CT.shape[1] // 2, :]
ct_seg_mask = np.rot90(ct_seg_mask, k=1)
axes[0, 1].set_title('CT Segmentation Mask', y=1.08, fontweight='bold')  # Add margin to the title
axes[0, 1].imshow(ct_seg_mask, cmap='viridis')

mri_img = original_image_data_MRI[:, :, original_image_data_MRI.shape[2] // 2]
mri_img = np.rot90(mri_img, k=1)
axes[1, 0].set_title('MRI Image', y=1.06, fontweight='bold')  # Add margin to the title
axes[1, 0].imshow(mri_img, cmap='gray')

mri_seg_mask = original_label_data_MRI[:, :, original_label_data_MRI.shape[2] // 2]
mri_seg_mask = np.rot90(mri_seg_mask, k=1)
axes[1, 1].set_title('MRI Segmentation Mask', y=1.06, fontweight='bold')
axes[1, 1].imshow(mri_seg_mask, cmap='viridis')

axes[0, 1].text(-label_margin, ct_seg_mask.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[0, 1].text(ct_seg_mask.shape[1] + label_margin, ct_seg_mask.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[0, 1].text(ct_seg_mask.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[0, 1].text(ct_seg_mask.shape[1] // 2, ct_seg_mask.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

axes[1, 0].text(-label_margin, mri_img.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] + label_margin, mri_img.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] // 2, mri_img.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

axes[1, 1].text(-label_margin, mri_seg_mask.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[1, 1].text(mri_seg_mask.shape[1] + label_margin, mri_seg_mask.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[1, 1].text(mri_seg_mask.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[1, 1].text(mri_seg_mask.shape[1] // 2, mri_seg_mask.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# plt.savefig('output_plot.png')

plt.show()
