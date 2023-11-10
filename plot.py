from data_loading import DataProcessor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

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

label_values = np.array([0., 205., 420., 500., 550., 600., 820., 850.])
original_label_data_CT = np.digitize(original_label_data_CT, label_values) - 1
original_label_data_MRI = np.digitize(original_label_data_MRI, label_values) - 1

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8), gridspec_kw={'width_ratios': [1.1, 1, 1, 0.8]})
plt.subplots_adjust(top=0.9)

ct_img = original_image_data_CT[:, original_image_data_CT.shape[1] // 2, :]
ct_img = np.rot90(ct_img, k=1)
axes[0, 0].set_title('CT Scan', y=1.08, fontweight='bold')
axes[0, 0].imshow(ct_img, cmap='gray')

label_margin = 20

axes[0, 0].text(-label_margin, ct_img.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] + label_margin, ct_img.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[0, 0].text(ct_img.shape[1] // 2, ct_img.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

ct_seg_mask = original_label_data_CT[:, original_label_data_CT.shape[1] // 2, :]
ct_seg_mask = np.rot90(ct_seg_mask, k=1)
axes[0, 1].set_title('CT Segmentation Mask', y=1.08, fontweight='bold')
ct_seg_img = axes[0, 1].imshow(ct_seg_mask, cmap='hsv')

mri_img = original_image_data_MRI[:, :, original_image_data_MRI.shape[2] // 2]
mri_img = np.rot90(mri_img, k=1)
axes[1, 0].set_title('MRI Image', y=1.06, fontweight='bold')
axes[1, 0].imshow(mri_img, cmap='gray')

mri_seg_mask = original_label_data_MRI[:, :, original_label_data_MRI.shape[2] // 2]
mri_seg_mask = np.rot90(mri_seg_mask, k=1)
axes[1, 1].set_title('MRI Segmentation Mask', y=1.06, fontweight='bold')
mri_seg_img = axes[1, 1].imshow(mri_seg_mask, cmap='hsv')

axes[1, 0].text(-label_margin, mri_img.shape[0] // 2, 'R', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] + label_margin, mri_img.shape[0] // 2, 'L', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] // 2, -label_margin, 'S', fontsize=12, ha='center', va='center')
axes[1, 0].text(mri_img.shape[1] // 2, mri_img.shape[0] + label_margin, 'I', fontsize=12, ha='center', va='center')

cmap = plt.cm.hsv
new_cmap = cmap(np.arange(cmap.N))
new_cmap[:, -1] = np.array([0.0 if i == 0 else 1 for i in range(cmap.N)])  # Set alpha values

# Create a ListedColormap object
custom_cmap = ListedColormap(new_cmap)

# Normalize the label values to the range of the colormap
norm = Normalize(vmin=0, vmax=np.max([np.max(ct_seg_mask), np.max(mri_seg_mask)]))

# Plotting for CT
axes[0, 2].set_title('CT Image with Segmentation Mask', y=1.08, fontweight='bold')
combined_ct_plot = axes[0, 2].imshow(np.zeros_like(ct_img), cmap='gray')
ct_img_plot = axes[0, 2].imshow(ct_img, cmap='gray', alpha=1.0)
ct_seg_mask_plot = axes[0, 2].imshow(ct_seg_mask, cmap=custom_cmap, norm=norm)

# Plotting for MRI
axes[1, 2].set_title('MRI Image with Segmentation Mask', y=1.08, fontweight='bold')
combined_mri_plot = axes[1, 2].imshow(np.zeros_like(mri_img), cmap='gray')
mri_img_plot = axes[1, 2].imshow(mri_img, cmap='gray', alpha=1.0)
mri_seg_mask_plot = axes[1, 2].imshow(mri_seg_mask, cmap=custom_cmap, norm=norm)


unique_colors_mri = np.unique(mri_seg_mask)

legend_patches_mri = [Patch(color=mri_seg_img.cmap(mri_seg_img.norm(color)), label=f'{color_label}') for
                      color, color_label in zip(unique_colors_mri, ["Background", "Myocardium of the left ventricle",
                                                                    "Left atrium blood cavity",
                                                                    "Left ventricle blood cavity",
                                                                    "Right atrium blood cavity",
                                                                    "Right ventricle blood cavity", "Ascending aorta",
                                                                    "Pulmonary artery"])]

axes[1, 3] = plt.subplot2grid((2, 4), (1, 3), rowspan=1)
legend_mri = axes[1, 3].legend(handles=legend_patches_mri, loc='lower right', title='Legend')
legend_mri.get_title().set_fontweight('bold')
axes[1, 3].axis('off')

axes[0, 3].set_visible(False)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
