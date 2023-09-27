import os
import glob
import numpy as np
import torch
import nibabel as nib
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Tuple, Optional


class MMWHSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MM-WHS dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], is_validation_dataset: bool,
                 patches_filter: int, mean: Optional[float] = None,
                 std_dev: Optional[float] = None) -> None:
        """
        Initialize the MMWHSDataset.

        Args:
            folder_path (str): The path to the folder containing the dataset.
            patch_size (tuple): The size of the patches to extract from the data.
            is_validation_dataset (bool): True if this is a validation dataset, False for training.
            patches_filter (int): The filter value for patches.
            mean (float, optional): The mean value for normalization (default: None).
            std_dev (float, optional): The standard deviation value for normalization (default: None).
        """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.is_validation_dataset = is_validation_dataset
        self.patches_filter = patches_filter
        self.mean = mean
        self.std_dev = std_dev
        self.x, self.y, self.num_classes, self.label_values, self.original_image_data, self.original_label_data, \
            self.mean, self.std_dev = self.load_data()

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

    def extract_patches(self, image_data: np.ndarray, label_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from the given image data.

        Args:
            image_data (np.ndarray): The input image data, dimensions: (width, height, depth).
            label_data (np.ndarray): The input label data.

        Returns:
            np.ndarray: An array of extracted image patches.
            np.ndarray: An array of extracted label patches.
        """
        image_patches = []
        label_patches = []
        dim_x, dim_y, dim_z = image_data.shape
        mod_x = dim_x % self.patch_size[0]
        mod_y = dim_y % self.patch_size[1]
        mod_z = dim_z % self.patch_size[2]
        pad_x = 0 if mod_x == 0 else self.patch_size[0] - mod_x
        pad_y = 0 if mod_y == 0 else self.patch_size[1] - mod_y
        pad_z = 0 if mod_z == 0 else self.patch_size[2] - mod_z
        image_data = np.pad(image_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        label_data = np.pad(label_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')

        for x in range(0, image_data.shape[0], self.patch_size[0]):
            for y in range(0, image_data.shape[1], self.patch_size[1]):
                for z in range(0, image_data.shape[2], self.patch_size[2]):
                    img_patch = image_data[x: x + self.patch_size[0], y: y + self.patch_size[1],
                                           z: z + self.patch_size[2]]
                    img_patch = np.expand_dims(img_patch, axis=0)
                    label_patch = label_data[x: x + self.patch_size[0], y: y + self.patch_size[1],
                                             z: z + self.patch_size[2]]
                    label_patch = np.expand_dims(label_patch, axis=0)
                    unique, counts = np.unique(label_patch, return_counts=True)
                    counts_descending = -np.sort(-counts)
                    if self.is_validation_dataset or (len(unique) > 1 and counts_descending[1] >= self.patches_filter) \
                            or unique[0] != 0:
                        image_patches.append(img_patch)
                        label_patches.append(label_patch)
        print(f"np.array(image_patches).shape: {np.array(image_patches).shape}, np.array(label_patches).shape: {np.array(label_patches).shape}")
        return np.array(image_patches), np.array(label_patches)

    def normalize_z_score_data(self, raw_data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Normalize the given raw data using z-score normalization.

        Args:
            raw_data (np.ndarray): The raw input data.

        Returns:
            np.ndarray: The normalized data.
        """
        if self.is_validation_dataset and self.mean is not None and self.std_dev is not None:
            mean = self.mean
            std_dev = self.std_dev
        else:
            mean = float(np.mean(raw_data))
            std_dev = float(np.std(raw_data))

        normalized_data = (raw_data - mean) / std_dev
        return normalized_data, mean, std_dev

    @staticmethod
    def preprocess_label_data(raw_data: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Prepare the label data for training.

        Args:
            raw_data (np.ndarray): The raw label data.

        Returns:
            np.ndarray: The prepared label data.
        """
        label_values = np.sort(np.unique(raw_data))
        for ind, val in enumerate(label_values):
            raw_data[raw_data == val] = ind

        raw_data = np.squeeze(raw_data)
        return raw_data, len(label_values), label_values

    def create_training_data_array(self, path_list: list) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Create the training data array from the given list of paths.

        Args:
            path_list (list): The list of file paths.

        Returns:
            np.ndarray: The patched image data.
            np.ndarray: The patched label data.
        """
        patches_images = []
        patches_labels = []
        for path in path_list:
            image_data = np.array(nib.load(path).get_fdata())
            label_data = np.array(nib.load(path.replace('image', 'label')).get_fdata())
            original_image_data = image_data
            original_label_data = label_data
            image_data, label_data = self.extract_patches(image_data, label_data)
            patches_images.append(image_data)
            patches_labels.append(label_data)

        patches_images = np.concatenate(patches_images, axis=0)
        patches_labels = np.concatenate(patches_labels, axis=0)
        print(f"is validation: {self.is_validation_dataset} -> shape of patches array: {patches_images.shape}")
        return patches_images, patches_labels, original_image_data, original_label_data

    def get_training_data_from_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the training data from the file system.

        Returns:
            tuple: The input and target training data.
        """
        image_path_names = glob.glob(os.path.join(self.folder_path, "*image.nii*"))
        if not image_path_names:
            raise ValueError("Empty list! Check if folder path contains images.")
        ret_imgs, ret_labels, original_image_data, original_label_data = \
            self.create_training_data_array(image_path_names)
        return ret_imgs, ret_labels, original_image_data, original_label_data

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, int, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Load and preprocess the dataset.

        Returns:
            tuple: The preprocessed input and target data tensors.
        """
        img_data, label_data, original_image_data, original_label_data = self.get_training_data_from_system()
        img_data, mean, std_dev = self.normalize_z_score_data(img_data)
        label_data, num_classes, label_values = self.preprocess_label_data(label_data)
        return torch.from_numpy(img_data), torch.from_numpy(label_data), num_classes, label_values, \
            original_image_data, original_label_data, mean, std_dev
