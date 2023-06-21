# standard library imports
import os
from typing import Union
import glob
# third party imports
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


# local application imports
# TODO: Save CT converted to pytorch tensor to local system and Google colab


class MMWHSDataset(Dataset):

    def __init__(
            self,
            raw_data_dir: str, subfolders: Union[tuple, list, str], patch_size: tuple
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.subfolders = subfolders
        self.patch_size = patch_size
        self.x, self.y = self.load_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def extract_patches(self, image_data):
        patches = []
        dim_x, dim_y, dim_z = image_data.shape
        mod_x = dim_x % self.patch_size[0]
        mod_y = dim_y % self.patch_size[1]
        mod_z = dim_z % self.patch_size[2]
        pad_x = 0 if mod_x == 0 else self.patch_size[0] - mod_x
        pad_y = 0 if mod_y == 0 else self.patch_size[1] - mod_y
        pad_z = 0 if mod_z == 0 else self.patch_size[2] - mod_z
        image_data = np.pad(image_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')

        for x in range(0, image_data.shape[0], self.patch_size[0]):
            for y in range(0, image_data.shape[1], self.patch_size[1]):
                for z in range(0, image_data.shape[2], self.patch_size[2]):
                    patch = image_data[x: x + self.patch_size[0], y: y + self.patch_size[1], z: z + self.patch_size[2]]
                    patch = np.expand_dims(patch, axis=0)
                    patches.append(patch)

        return np.array(patches)

    def normalize_minmax_data(self, raw_data, min_val=1, max_val=99, is_label=False):
        """
        # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
        Here, the minimum and maximum values are used as 1st and 99th percentiles respectively from the 3D MRI scan.
        We expect the outliers to be away from the range of [0,1].
        input params :
            image_data : 3D MRI scan to be normalized using min-max normalization
            min_val : minimum value percentile
            max_val : maximum value percentile
        returns:
            final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
        """
        if is_label:
            label_values = np.sort(np.unique(raw_data))
            for ind, val in enumerate(label_values):
                raw_data[raw_data == val] = ind
            normalized_data = raw_data
        elif not is_label:
            min_val_low_p = np.percentile(raw_data, min_val)
            max_val_high_p = np.percentile(raw_data, max_val)
            normalized_data = (raw_data - min_val_low_p) / (max_val_high_p - min_val_low_p)
        return normalized_data

    def load_data(self):
        """
        # MRI and CT scans are loaded and stored into 4D torch tensor (width x height x
        slices x number of scans).
        input params:
        returns:
            ret_img_tensor: Normalized 4D MRI/CT scans as pytorch tensor
        """

        def create_training_data_array(path_list: list):
            ret_array = []
            for path in path_list:
                array = np.array(nib.load(path).get_fdata())
                array = self.extract_patches(array)
                ret_array.append(array)

            ret_array = np.concatenate(ret_array, axis=0)
            return ret_array

        def get_training_data(dire, subf):
            # Create lists with image/label paths
            image_path_names = []
            for idx, filename in enumerate(glob.glob(os.path.join(dire + subf, "*image.nii*"))):
                with open(filename, 'r'):
                    image_path_names.append(filename)
            if not image_path_names:
                raise ValueError("Empty list! Check if folder path & subfolder is correct.")
            label_path_names = [i.replace('image', 'label') for i in image_path_names]

            # Create arrays which contain the training data (images tensor + corresponding labels tensor)
            ret_imgs = create_training_data_array(image_path_names)
            ret_labels = create_training_data_array(label_path_names)
            return ret_imgs, ret_labels

        if isinstance(self.subfolders, tuple) or isinstance(self.subfolders, list):
            for subfolder in self.subfolders:
                img_data, label_data = get_training_data(self.raw_data_dir, subfolder)
        elif isinstance(self.subfolders, str):
            img_data, label_data = get_training_data(self.raw_data_dir, self.subfolders)
        else:
            raise ValueError("Subfolder variable must be of type list, tuple or string.")

        img_data = self.normalize_minmax_data(img_data, 0, 100)
        label_data = self.normalize_minmax_data(label_data, 0, 100, is_label=True)

        num_classes = 8
        label_data_one_hot_encoding = np.eye(num_classes)[label_data.astype(int)]
        label_data = []
        label_data_one_hot_encoding = np.transpose(np.squeeze(label_data_one_hot_encoding), (0, 4, 1, 2, 3))

        return torch.from_numpy(img_data), torch.from_numpy(label_data_one_hot_encoding)


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/"
    subfolder = "mr_train"
    patch_size = (24, 24, 24)
    dataset = MMWHSDataset(main_dir, subfolder, patch_size)
    print(f"image data: {dataset.x.shape}")
    print(f"labels: {dataset.y.shape}")
    print(f"example image data: {dataset.x[1, 0, 2:4, 2:4, 2:4]}")
    print(f"corresponding labels: {dataset.y[1, :, 2:4, 2:4, 2:4]}")
    print(f"unique labels: {np.unique(dataset.y)}")
