import os
import glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
from typing import Tuple


class MMWHSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MM-WHS dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int]) -> None:
        """
        Initialize the MMWHSDataset.

        Args:
            folder_path (str): The path to the folder containing the dataset.
            patch_size (tuple): The size of the patches to extract from the data.
        """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.x, self.y = self.load_data()

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
       Get a specific sample from the dataset.

       Args:
           idx (int): The index of the sample.

       Returns:
           tuple: The input and target data for the sample.
       """
        return self.x[idx], self.y[idx]

    def extract_patches(self, image_data: np.ndarray) -> np.ndarray:
        """
        Extract patches from the given image data.

        Args:
            image_data (np.ndarray): The input image data.

        Returns:
            np.ndarray: An array of extracted patches.
        """
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

    def normalize_minmax_data(self, raw_data: np.ndarray, min_val: float = 1, max_val: float = 99) -> np.ndarray:
        """
        Normalize the given raw data using min-max scaling.

        Args:
            raw_data (np.ndarray): The raw input data.
            min_val (float): The minimum percentile for normalization (default: 1).
            max_val (float): The maximum percentile for normalization (default: 99).

        Returns:
            np.ndarray: The normalized data.
        """
        min_val_low_p = np.percentile(raw_data, min_val)
        max_val_high_p = np.percentile(raw_data, max_val)
        normalized_data = (raw_data - min_val_low_p) / (max_val_high_p - min_val_low_p)
        return normalized_data

    def preprocess_label_data(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Prepare the label data for training.

        Args:
            raw_data (np.ndarray): The raw label data.

        Returns:
            np.ndarray: The prepared label data.
        """
        label_values = np.sort(np.unique(raw_data))
        num_classes = len(label_values)
        for ind, val in enumerate(label_values):
            raw_data[raw_data == val] = ind

        # raw_data = np.eye(num_classes)[raw_data.astype(int)]
        raw_data = np.transpose(np.squeeze(raw_data), (0, 3, 1, 2))
        return raw_data

    def create_training_data_array(self, path_list: list) -> np.ndarray:
        """
        Create the training data array from the given list of paths.

        Args:
            path_list (list): The list of file paths.

        Returns:
            np.ndarray: The training data array.
        """
        ret_array = []
        for path in path_list:
            array = np.array(nib.load(path).get_fdata()).astype(np.float16)
            array = self.extract_patches(array)
            ret_array.append(array)

        ret_array = np.concatenate(ret_array, axis=0, dtype=np.float16)
        return ret_array

    def get_training_data_from_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the training data from the file system.

        Returns:
            tuple: The input and target training data.
        """
        image_path_names = glob.glob(os.path.join(self.folder_path, "*image.nii*"))
        if not image_path_names:
            raise ValueError("Empty list! Check if folder path contains images.")
        label_path_names = [path.replace('image', 'label') for path in image_path_names]
        ret_imgs = self.create_training_data_array(image_path_names)
        ret_labels = self.create_training_data_array(label_path_names)
        return ret_imgs, ret_labels

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the dataset.

        Returns:
            tuple: The preprocessed input and target data tensors.
        """
        img_data, label_data = self.get_training_data_from_system()
        img_data = self.normalize_minmax_data(img_data, 0, 100).astype(np.float16)
        label_data = self.preprocess_label_data(label_data).astype(np.intc)
        return torch.from_numpy(img_data), torch.from_numpy(label_data)


if __name__ == "__main__":
    folder_path = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/ct_train"
    patch_size = (24, 24, 24)
    dataset = MMWHSDataset(folder_path=folder_path, patch_size=patch_size)
    print(f"image data: {dataset.x.shape}")
    print(f"labels: {dataset.y.shape}")
    print(f"example image data: {dataset.x[3, 0, 2:4, 2:4, 2:4]}")
    print(f"corresponding labels: {dataset.y[3, :, 2:4, 2:4, 2:4]}")
