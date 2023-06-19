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


# WARNING: MR scans in training set do not all have the same width/height (CT scans: all 512x512)

class MMWHSDataset(Dataset):

    def __init__(
            self,
            raw_data_dir: str, subfolders: Union[tuple, list, str]
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.subfolders = subfolders
        self.x, self.y = self.load_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
            normalized_data = np.zeros((raw_data.shape[0], raw_data.shape[1],
                                        raw_data.shape[2], raw_data.shape[3]), dtype=np.float64)
            # min-max norm on total 3D volume
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

        def create_training_data_array(path_list: list, depth: int):
            for i, path in enumerate(path_list):
                if i == 0:
                    ret_array = np.array(nib.load(path).get_fdata())
                    ret_array.resize([ret_array.shape[0], ret_array.shape[1], depth])
                    # ret array with shape (samples, width, height, depth)
                    ret_array = np.expand_dims(ret_array, axis=0)
                else:
                    buf = np.array(nib.load(path).get_fdata())
                    buf.resize([buf.shape[0], buf.shape[1], depth])
                    buf = np.expand_dims(buf, axis=0)
                    # ret_array with shape (i+1 samples, width, height, depth)
                    ret_array = np.concatenate((ret_array, buf), 0)
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
            # Determine third dimension of array for resizing by finding max. third dimension
            # max_num_of_slices = nib.load(image_path_names[0]).get_fdata().shape[-1]
            max_num_of_slices = 4
            for index, path in enumerate(image_path_names):
                if nib.load(path).get_fdata().shape[-1] > max_num_of_slices:
                    max_num_of_slices = nib.load(path).get_fdata().shape[-1]
            # Create arrays which contain the training data (images tensor + corresponding labels tensor)
            ret_imgs = create_training_data_array(image_path_names, max_num_of_slices)
            ret_labels = create_training_data_array(label_path_names, max_num_of_slices)
            return ret_imgs, ret_labels

        directory = self.raw_data_dir
        if isinstance(self.subfolders, tuple) or isinstance(self.subfolders, list):
            for subfolder in self.subfolders:
                img_data, label_data = get_training_data(directory, subfolder)
        elif isinstance(self.subfolders, str):
            img_data, label_data = get_training_data(directory, self.subfolders)
        else:
            raise ValueError("Subfolder variable must be of type list, tuple or string.")
        img_data = self.normalize_minmax_data(img_data, 0, 100)
        label_data = self.normalize_minmax_data(label_data, 0, 100, is_label=True)
        # Add dimension for number of channels: (samples, channels, width, height, depth)
        img_data = np.expand_dims(img_data, axis=1)
        label_data = np.expand_dims(label_data, axis=1)
        return torch.from_numpy(img_data), torch.from_numpy(label_data)


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/"
    subfolder = "ct_train"
    dataset = MMWHSDataset(main_dir, subfolder)
    print(f"image data: {dataset.x.shape}")
    print(f"labels: {dataset.y.shape}")
    print(f"example image data: {dataset.x[0, 0, 200:202, 200:202, 300]}")
    print(f"corresponding labels: {dataset.y[0, 0, 200:202, 200:202, 300]}")
    print(f"unique labels: {np.unique(dataset.y)}")
