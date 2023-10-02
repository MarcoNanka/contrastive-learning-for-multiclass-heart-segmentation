import os
import glob
import numpy as np
import torch
import nibabel as nib
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union
import random
from monai.transforms import Compose, RandFlip, ToTensor, RandZoom, RandGaussianNoise, RandGaussianSmooth


class DataProcessor:
    @staticmethod
    def preprocess_label_data(raw_data: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Prepare the label data for training.
        """
        label_values = np.sort(np.unique(raw_data))
        for ind, val in enumerate(label_values):
            raw_data[raw_data == val] = ind

        raw_data = np.squeeze(raw_data)
        return raw_data, len(label_values), label_values

    @staticmethod
    def normalize_z_score_data(raw_data: np.ndarray, is_validation_dataset: bool = False, mean: float = None,
                               std_dev: float = None, is_contrastive_dataset: bool = False) -> \
            Union[Tuple[np.ndarray, float, float], list]:
        """
        Normalize the given raw data using z-score normalization.
        """
        if is_contrastive_dataset:
            buf_raw_data = np.concatenate(raw_data, axis=0)
            mean = float(np.mean(buf_raw_data))
            std_dev = float(np.std(buf_raw_data))
            print(f"contrastive dataset --- mean: {mean}, std_dev: {std_dev}")
            ret_array = []
            for idx, _ in enumerate(raw_data):
                normalized_data = (raw_data[idx] - mean) / std_dev
                ret_array.append(normalized_data)
            return ret_array

        if not is_validation_dataset:
            mean = float(np.mean(raw_data))
            std_dev = float(np.std(raw_data))

        normalized_data = (raw_data - mean) / std_dev
        return normalized_data, mean, std_dev

    @staticmethod
    def undo_extract_patches_label_only(label_patches: np.ndarray, patch_size: Tuple[int, int, int],
                                        original_label_data: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original label data from extracted label patches of the validation dataset.
        """
        original_shape = original_label_data.shape
        dim_x, dim_y, dim_z = original_shape
        label_data = np.zeros(original_shape, dtype=label_patches.dtype)

        patch_index = 0

        for x in range(0, dim_x, patch_size[0]):
            for y in range(0, dim_y, patch_size[1]):
                for z in range(0, dim_z, patch_size[2]):
                    label_patch = label_patches[patch_index]
                    x_end = min(x + patch_size[0], dim_x)
                    y_end = min(y + patch_size[1], dim_y)
                    z_end = min(z + patch_size[2], dim_z)
                    label_data[x:x_end, y:y_end, z:z_end] = label_patch[:x_end - x, :y_end - y, :z_end - z]
                    patch_index += 1

        return label_data

    @staticmethod
    def extract_patches(image_data: np.ndarray, label_data: np.ndarray, patch_size: Tuple[int, int, int],
                        is_validation_dataset: bool, patches_filter: int, is_contrastive_dataset: bool) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from the given image data.
        """
        image_patches = []
        label_patches = []
        dim_x, dim_y, dim_z = image_data.shape
        mod_x = dim_x % patch_size[0]
        mod_y = dim_y % patch_size[1]
        mod_z = dim_z % patch_size[2]
        pad_x = 0 if mod_x == 0 else patch_size[0] - mod_x
        pad_y = 0 if mod_y == 0 else patch_size[1] - mod_y
        pad_z = 0 if mod_z == 0 else patch_size[2] - mod_z
        image_data = np.pad(image_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        label_data = np.pad(label_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')

        for x in range(0, image_data.shape[0], patch_size[0]):
            for y in range(0, image_data.shape[1], patch_size[1]):
                for z in range(0, image_data.shape[2], patch_size[2]):
                    img_patch = image_data[x: x + patch_size[0], y: y + patch_size[1],
                                           z: z + patch_size[2]]
                    img_patch = np.expand_dims(img_patch, axis=0)
                    if not is_contrastive_dataset:
                        label_patch = label_data[x: x + patch_size[0], y: y + patch_size[1],
                                                 z: z + patch_size[2]]
                        label_patch = np.expand_dims(label_patch, axis=0)
                    else:
                        label_patch = np.empty((0, 0, 0))
                    unique, counts = np.unique(label_patch, return_counts=True)
                    counts_descending = -np.sort(-counts)
                    if is_validation_dataset or is_contrastive_dataset or unique[0] != 0 or \
                            (len(unique) > 1 and counts_descending[1] >= patches_filter):
                        image_patches.append(img_patch)
                        label_patches.append(label_patch)

        return np.array(image_patches), np.array(label_patches)

    @staticmethod
    def create_training_data_array(path_list: list, is_validation_dataset: bool, patch_size: Tuple[int, int, int],
                                   patches_filter: int, is_contrastive_dataset: bool) -> \
            tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Create the training data array from the given list of paths.
        """
        patches_images = []
        patches_labels = []
        for path in path_list:
            image_data = np.array(nib.load(path).get_fdata())
            original_image_data = image_data
            if not is_contrastive_dataset:
                label_data = np.array(nib.load(path.replace('image', 'label')).get_fdata())
                original_label_data = label_data
            else:
                label_data = original_label_data = np.empty((0, 0, 0))
            if is_validation_dataset:
                unique, counts = np.unique(label_data, return_counts=True)
                print(f"COUNTER LABELS (without background padding): {unique, counts}")
            image_data, label_data = DataProcessor.extract_patches(image_data=image_data, label_data=label_data,
                                                                   patch_size=patch_size,
                                                                   is_validation_dataset=is_validation_dataset,
                                                                   patches_filter=patches_filter,
                                                                   is_contrastive_dataset=is_contrastive_dataset)
            patches_images.append(image_data)
            patches_labels.append(label_data)

        if not is_contrastive_dataset:
            patches_images = np.concatenate(patches_images, axis=0)
            patches_labels = np.concatenate(patches_labels, axis=0)
            print(f"is validation: {is_validation_dataset} -> shape of patches array: {patches_images.shape}")

        else:
            print(f"contrastive dataset length: {len(patches_images)}, shape of first image: {patches_images[0].shape}")

        return patches_images, patches_labels, original_image_data, original_label_data

    @staticmethod
    def get_training_data_from_system(folder_path: str, is_validation_dataset: bool, patch_size: Tuple[int, int, int],
                                      patches_filter: int, is_contrastive_dataset: bool = False) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the training data from the file system.
        """
        image_path_names = glob.glob(os.path.join(folder_path, "*image.nii*"))
        if not image_path_names:
            raise ValueError("Empty list! Check if folder path contains images.")

        return DataProcessor.create_training_data_array(path_list=image_path_names,
                                                        is_validation_dataset=is_validation_dataset,
                                                        patch_size=patch_size,
                                                        patches_filter=patches_filter,
                                                        is_contrastive_dataset=is_contrastive_dataset)


class MMWHSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MM-WHS dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], is_validation_dataset: bool,
                 patches_filter: int, mean: Optional[float] = None,
                 std_dev: Optional[float] = None) -> None:
        """
        Initialize the MMWHSDataset for supervised learning.
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
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
       Get a specific sample from the dataset.
       """
        return self.x[idx], self.y[idx]

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, int, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Load and preprocess the dataset.
        """
        img_data, label_data, original_image_data, original_label_data = DataProcessor. \
            get_training_data_from_system(folder_path=self.folder_path,
                                          is_validation_dataset=self.is_validation_dataset, patch_size=self.patch_size,
                                          patches_filter=self.patches_filter)
        img_data, mean, std_dev = DataProcessor.normalize_z_score_data(raw_data=img_data, is_validation_dataset=self.
                                                                       is_validation_dataset, mean=self.mean,
                                                                       std_dev=self.std_dev)
        label_data, num_classes, label_values = DataProcessor.preprocess_label_data(raw_data=label_data)
        return torch.from_numpy(img_data), torch.from_numpy(label_data), num_classes, label_values, \
            original_image_data, original_label_data, mean, std_dev


class MMWHSContrastiveDataset(Dataset):
    """
        Custom PyTorch Dataset for loading MM-WHS contrastive learning dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], patches_filter: int):
        """
            Initialize the MMWHSDataset for contrastive learning.
            """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.patches_filter = patches_filter
        self.transform = Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandZoom(min_zoom=0.8, max_zoom=1.2, prob=0.5),
            RandGaussianNoise(prob=0.5),
            RandGaussianSmooth(prob=0.5),
            ToTensor()
        ])
        self.x, self.original_image_data = self.load_data()

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
       Get a specific sample from the dataset.
       """
        sample = self.x[idx]
        positive_pair = self.transform(sample)
        negative_pair = self.transform(random.choice(self.x))
        return positive_pair, negative_pair

    def load_data(self):
        """
        Load and preprocess the dataset.
        """
        img_data, _, original_image_data, _ = DataProcessor. \
            get_training_data_from_system(folder_path=self.folder_path,
                                          is_validation_dataset=False, patch_size=self.patch_size,
                                          patches_filter=self.patches_filter,
                                          is_contrastive_dataset=True)
        img_data = DataProcessor.normalize_z_score_data(raw_data=img_data, is_contrastive_dataset=True)
        for idx, _ in enumerate(img_data):
            torch.from_numpy(img_data[idx])
        return img_data, original_image_data
