from data_loading import DataProcessor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.gridspec as gridspec

# Load your data
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

# Create a custom gridspec
gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 0.25], wspace=0.1, hspace=0.2)

fig = plt.figure(figsize=(16, 12), dpi=600)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

# First row
ax00 = plt.subplot(gs[0, 0])
ax01 = plt.subplot(gs[0, 1])
ax02 = plt.subplot(gs[0, 2])

# Second row
ax10 = plt.subplot(gs[1, 0])
ax11 = plt.subplot(gs[1, 1])
ax16 = plt.subplot(gs[1, 2])

# Third row
ax2 = plt.subplot(gs[2, :])

# First row
ct_img = original_image_data_CT[:, original_image_data_CT.shape[1] // 2, :]
ct_img = np.rot90(ct_img, k=1)
ax00.set_title('CT Scan', y=1.08, fontweight='bold')
ax00.imshow(ct_img, cmap='gray')

label_margin = 20

ax00.text(-label_margin, ct_img.shape[0] // 2, 'R', fontsize=16, ha='center', va='center')
ax00.text(ct_img.shape[1] + label_margin, ct_img.shape[0] // 2, 'L', fontsize=16, ha='center', va='center')
ax00.text(ct_img.shape[1] // 2, -label_margin, 'S', fontsize=16, ha='center', va='center')
ax00.text(ct_img.shape[1] // 2, ct_img.shape[0] + label_margin, 'I', fontsize=16, ha='center', va='center')

ct_seg_mask = original_label_data_CT[:, original_label_data_CT.shape[1] // 2, :]
ct_seg_mask = np.rot90(ct_seg_mask, k=1)
ax01.set_title('CT Segmentation Mask', y=1.08, fontweight='bold')
ct_seg_img = ax01.imshow(ct_seg_mask, cmap='viridis')

cmap = plt.cm.viridis
new_cmap = cmap(np.arange(cmap.N))
new_cmap[:, -1] = np.array([0.0 if i == 0 else 1 for i in range(cmap.N)])
custom_cmap = ListedColormap(new_cmap)
norm = Normalize(vmin=0, vmax=np.max(ct_seg_mask))

ax02.set_title('CT Scan + Segmentation Mask', y=1.08, fontweight='bold')
combined_ct_plot = ax02.imshow(np.zeros_like(ct_img), cmap='gray')
ct_img_plot = ax02.imshow(ct_img, cmap='gray', alpha=1.0)
ct_seg_mask_plot = ax02.imshow(ct_seg_mask, cmap=custom_cmap, norm=norm)

# Second row
mri_img = original_image_data_MRI[:, :, original_image_data_MRI.shape[2] // 2]
mri_img = np.rot90(mri_img, k=1)
ax10.set_title('MRI Image', y=1.06, fontweight='bold')
ax10.imshow(mri_img, cmap='gray')

ax10.text(-label_margin, mri_img.shape[0] // 2, 'R', fontsize=16, ha='center', va='center')
ax10.text(mri_img.shape[1] + label_margin, mri_img.shape[0] // 2, 'L', fontsize=16, ha='center', va='center')
ax10.text(mri_img.shape[1] // 2, -label_margin, 'S', fontsize=16, ha='center', va='center')
ax10.text(mri_img.shape[1] // 2, mri_img.shape[0] + label_margin, 'I', fontsize=16, ha='center', va='center')

mri_seg_mask = original_label_data_MRI[:, :, original_label_data_MRI.shape[2] // 2]
mri_seg_mask = np.rot90(mri_seg_mask, k=1)
ax11.set_title('MRI Segmentation Mask', y=1.06, fontweight='bold')
mri_seg_img = ax11.imshow(mri_seg_mask, cmap='viridis')

ax16.set_title('MRI Scan + Segmentation Mask', y=1.08, fontweight='bold')
combined_mri_plot = ax16.imshow(np.zeros_like(mri_img), cmap='gray')
mri_img_plot = ax16.imshow(mri_img, cmap='gray', alpha=1.0)
mri_seg_mask_plot = ax16.imshow(mri_seg_mask, cmap=custom_cmap, norm=norm)

for ax in [ax00, ax01, ax02, ax10, ax11, ax16]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_fontsize(20)

# Third row
ax2.axis('off')

unique_colors_mri = np.unique(mri_seg_mask)

legend_patches_mri = [Patch(color=mri_seg_img.cmap(mri_seg_img.norm(color)), label=f'{color_label}') for
                      color, color_label in zip(unique_colors_mri, ["Background", "Myocardium of the left ventricle",
                                                                    "Left atrium blood cavity",
                                                                    "Left ventricle blood cavity",
                                                                    "Right atrium blood cavity",
                                                                    "Right ventricle blood cavity", "Ascending aorta",
                                                                    "Pulmonary artery"])]

# Use ncol to specify the number of columns for the legend
legend_mri = ax2.legend(handles=legend_patches_mri, title='Legend', ncol=3, bbox_to_anchor=(0.5, 0),
                        loc='lower center', frameon=False, numpoints=1)
legend_mri.get_title().set_fontweight('bold')
legend_mri.get_title().set_fontsize(20)
for i in range(7):
    legend_mri.get_texts()[i].set_fontsize(20)

plt.tight_layout()

plt.savefig('your_plot_filename.pdf')
