from data_loading import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
import matplotlib as mpl
from matplotlib import cm


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


_, _, original_image_data_CT, original_label_data_CT = \
    DataProcessor.create_training_data_array(
        path_list=["../../data/entire_dataset_split_ts_tr/ct_pre/ct_train_1001_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 96, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="MRI")

_, _, original_image_data_MRI, original_label_data_MRI = \
    DataProcessor.create_training_data_array(
        path_list=["../../data/entire_dataset_split_ts_tr/mr_pre/mr_train_1003_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 96, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="MRI")

label_values = np.array([0., 205., 420., 500., 550., 600., 820., 850.])
original_label_data_CT = np.digitize(original_label_data_CT, label_values) - 1
original_label_data_MRI = np.digitize(original_label_data_MRI, label_values) - 1


def plot_high_quality_image(image_data, mask_data=None, output_filename='output_image.png', dpi=600, cmap='gray',
                            is_combined=False, custom_cmap=None, norm=None):
    fig, ax = plt.subplots(figsize=(16, 16), dpi=dpi)
    if not is_combined:
        ax.imshow(image_data, cmap=cmap)
    else:
        ax.imshow(np.zeros_like(image_data), cmap='gray')
        ax.imshow(image_data, cmap='gray', alpha=1.0)
        ax.imshow(mask_data, cmap=custom_cmap, norm=norm)
    ax.axis('off')
    plt.savefig(output_filename, dpi=dpi, bbox_inches='tight', pad_inches=0)


# TODO: Consider that Axes are not the same for CT and MRI (CT = [:, AP, :], MRI = [:, : AP]
ct_img = original_image_data_CT[:, 57, :]
ct_img = np.rot90(ct_img, k=1)
ct_seg_mask = original_label_data_CT[:, 57, :]
ct_seg_mask = np.rot90(ct_seg_mask, k=1)

cmap = plt.cm.viridis
new_cmap = cmap(np.arange(cmap.N))
new_cmap[:, -1] = np.array([0.0 if i == 0 else 1 for i in range(cmap.N)])
custom_cmap = ListedColormap(new_cmap)
norm = Normalize(vmin=0, vmax=np.max(ct_seg_mask))

mri_img = original_image_data_MRI[original_image_data_MRI.shape[0] // 2, :, :]
# mri_img = np.rot90(mri_img, k=1)
mri_seg_mask = original_label_data_MRI[original_label_data_MRI.shape[0] // 2, :, :]
# mri_seg_mask = np.rot90(mri_seg_mask, k=1)

plot_high_quality_image(ct_img, output_filename='ct_57.png')
plot_high_quality_image(ct_seg_mask, output_filename='ct_57_seg_mask.png', cmap='viridis')
plot_high_quality_image(ct_img, mask_data=ct_seg_mask, is_combined=True, output_filename='ct_57_img_with_seg_mask.png',
                        custom_cmap=custom_cmap, norm=norm)
# plot_high_quality_image(mri_img, output_filename='mri_img_second_category.png')
# plot_high_quality_image(mri_seg_mask, output_filename='mri_seg_mask.png', cmap='viridis')
# plot_high_quality_image(mri_img, mask_data=mri_seg_mask, is_combined=True, output_filename='mri_img_with_seg_mask.png',
#                         custom_cmap=custom_cmap, norm=norm)

# mplColorHelper = MplColorHelper('viridis', 0, 7)
# for i in range(0, 8):
#     print(tuple(value * 255 for value in mplColorHelper.get_rgb(i)))
