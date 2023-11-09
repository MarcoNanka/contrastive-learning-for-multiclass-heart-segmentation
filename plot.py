from data_loading import DataProcessor
import matplotlib.pyplot as plt


_, _, original_image_data_CT, original_label_data_CT = \
    DataProcessor.create_training_data_array(
        path_list=["../data/entire_dataset_split_ts_tr/ct_pre/ct_train_1001_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 66, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="CT")

_, _, original_image_data_MRI, original_label_data_MRI = \
    DataProcessor.create_training_data_array(
        path_list=["../data/entire_dataset_split_ts_tr/mr_pre/mr_train_1001_image.nii.gz"],
        is_validation_dataset=True, patch_size=(96, 66, 96),
        patches_filter=0, is_contrastive_dataset=False, image_type="MRI")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

axes[0, 0].set_title('CT Scan')
axes[0, 0].imshow(original_image_data_CT[:, :, original_image_data_CT.shape[2] // 2], cmap='gray')

axes[0, 1].set_title('CT Segmentation Mask')
axes[0, 1].imshow(original_label_data_CT[:, :, original_label_data_CT.shape[2] // 2], cmap='viridis')

axes[1, 0].set_title('MRI Image')
axes[1, 0].imshow(original_image_data_MRI[:, :, original_image_data_MRI.shape[2] // 2], cmap='gray')

axes[1, 1].set_title('MRI Segmentation Mask')
axes[1, 1].imshow(original_label_data_MRI[:, :, original_label_data_MRI.shape[2] // 2], cmap='viridis')

plt.tight_layout()

plt.savefig('output_plot.png')

plt.show()
