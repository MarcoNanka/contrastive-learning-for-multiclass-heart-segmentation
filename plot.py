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
    plt.savefig("../../prediction_plots/"+output_filename, dpi=dpi, bbox_inches='tight', pad_inches=0)


def main():
    _, _, original_image_data_mri, _ = \
        DataProcessor.create_training_data_array(
            path_list=["../../data/entire_dataset_split_ts_tr/mr_ts/mr_train_1011_image.nii.gz"],
            is_validation_dataset=True, patch_size=(96, 96, 96),
            patches_filter=0, is_contrastive_dataset=False, image_type="MRI")
    mask_names = [
        "mr_combined1_nn3_0.nii",
        "mr_combined2_n5_3.nii",
        "mr_combined8_n4_6.nii",
        "mr_global_da1_n2_4.nii",
        "mr_global_da2_3_0.nii",
        "mr_global_da8_3_2.nii",
        "mr_global_ft1_3_2.nii",
        "mr_global_ft2_3_5.nii",
        "mr_global_ft8_2_3.nii",
        "mr_local1_2_1.nii",
        "mr_local2_1_7.nii",
        "mr_local8_3_2.nii",
        "mr_random1_1_1.nii",
        "mr_random2_3_2.nii",
        "mr_random8_2_7.nii"
    ]
    for mask_name in mask_names:
        _, _, _, original_label_data_mri = \
            DataProcessor.create_training_data_array(
                path_list=["../../prediction_masks/"+mask_name],
                is_validation_dataset=True, patch_size=(96, 96, 96),
                patches_filter=0, is_contrastive_dataset=False, image_type="MRI")
        label_values = np.array([0., 205., 420., 500., 550., 600., 820., 850.])
        original_label_data_mri = np.digitize(original_label_data_mri, label_values) - 1

        # TODO: Consider that Axes are not the same for CT and MRI (CT = [:, AP, :], MRI = [:, : AP], rot 90 for ct
        mri_img = original_image_data_mri[135, :, :]
        mri_img = np.rot90(mri_img, k=1)
        mri_img = np.rot90(mri_img, k=1)
        mri_seg_mask = original_label_data_mri[135, :, :]
        mri_seg_mask = np.rot90(mri_seg_mask, k=1)
        mri_seg_mask = np.rot90(mri_seg_mask, k=1)

        cmap = plt.cm.viridis
        new_cmap = cmap(np.arange(cmap.N))
        new_cmap[:, -1] = np.array([0.0 if i == 0 else 1 for i in range(cmap.N)])
        custom_cmap = ListedColormap(new_cmap)
        norm = Normalize(vmin=0, vmax=7)
        print(np.max(mri_seg_mask), cmap.N, norm, custom_cmap)
        # plot_high_quality_image(mri_img, output_filename=mask_name+"volume.png")
        # plot_high_quality_image(mri_seg_mask, output_filename=mask_name+"seg_mask.png", cmap='viridis')
        plot_high_quality_image(mri_img, mask_data=mri_seg_mask, is_combined=True,
                                output_filename=mask_name+"seg_mask_with_volume.png",
                                custom_cmap=custom_cmap, norm=norm)

    # mplColorHelper = MplColorHelper('viridis', 0, 7)
    # for i in range(0, 8):
    #     print(tuple(value * 255 for value in mplColorHelper.get_rgb(i)))


if __name__ == "__main__":
    main()
