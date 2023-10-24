import os
import glob
import numpy as np
import torch
import nibabel as nib
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Tuple, Optional
from monai.transforms import Compose, RandFlip, ToTensor, RandGaussianNoise, RandGaussianSmooth


class DataProcessor:
    @staticmethod
    def preprocess_label_data(raw_data: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        label_values = np.array([0., 205., 420., 500., 550., 600., 820., 850.])
        raw_data = np.digitize(raw_data, label_values) - 1
        raw_data = np.squeeze(raw_data)
        return raw_data, len(label_values), label_values

    @staticmethod
    def normalize_z_score_data(raw_data: np.ndarray, is_validation_dataset: bool = False, mean: float = None,
                               std_dev: float = None) -> Tuple[np.ndarray, float, float]:
        """
        Normalize the given raw data using z-score normalization.
        """
        if not is_validation_dataset:
            mean, std_dev = float(np.mean(raw_data)), float(np.std(raw_data))
        normalized_data = (raw_data - mean) / std_dev
        return normalized_data, mean, std_dev

    @staticmethod
    def undo_extract_patches_label_only(label_patches: np.ndarray, patch_size: Tuple[int, int, int],
                                        original_label_data: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original label data from extracted label patches of the validation dataset.
        """
        original_shape = original_label_data.shape
        label_data = np.zeros(original_shape, dtype=label_patches.dtype)
        patch_index = 0

        for x in range(0, original_shape[0], patch_size[0]):
            for y in range(0, original_shape[1], patch_size[1]):
                for z in range(0, original_shape[2], patch_size[2]):
                    label_patch = label_patches[patch_index]
                    x_end = min(x + patch_size[0], original_shape[0])
                    y_end = min(y + patch_size[1], original_shape[1])
                    z_end = min(z + patch_size[2], original_shape[2])
                    label_data[x:x_end, y:y_end, z:z_end] = label_patch[:x_end - x, :y_end - y, :z_end - z]
                    patch_index += 1

        return label_data

    @staticmethod
    def extract_patches(image_data: np.ndarray, label_data: np.ndarray, patch_size: Tuple[int, int, int],
                        is_validation_dataset: bool, patches_filter: int, is_contrastive_dataset: bool,
                        image_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from the given image data.
        """
        if patch_size[0] > 0:
            image_patches = []
            label_patches = []
            dim_x, dim_y, dim_z = image_data.shape
            mod_x, mod_y, mod_z = dim_x % patch_size[0], dim_y % patch_size[1], dim_z % patch_size[2]
            pad_x = 0 if mod_x == 0 else patch_size[0] - mod_x
            pad_y = 0 if mod_y == 0 else patch_size[1] - mod_y
            pad_z = 0 if mod_z == 0 else patch_size[2] - mod_z
            label_data = np.pad(label_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant', constant_values=0)
            image_data = np.pad(image_data, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant',
                                constant_values=(0 if image_type == "MRI" else -3022))

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

        else:
            image_patches = []
            label_patches = []
            posterior_anterior_axis = image_data.shape[0] if image_type == "MRI" else image_data.shape[1]
            remainder = posterior_anterior_axis % patch_size[2]

            for i in range(remainder // 2, posterior_anterior_axis - (remainder + 1) // 2, patch_size[2]):
                img_patch = (image_data[i:i + patch_size[2], :, :] if image_type == "MRI" and image_data.shape[1] == 512
                             else image_data[:, i:i + patch_size[2], :])
                img_patch = np.expand_dims(img_patch, axis=0)
                label_patch = np.empty((0, 0, 0))
                image_patches.append(img_patch)
                label_patches.append(label_patch)

        return np.array(image_patches), np.array(label_patches)

    @staticmethod
    def create_training_data_array(path_list: list, is_validation_dataset: bool, patch_size: Tuple[int, int, int],
                                   patches_filter: int, is_contrastive_dataset: bool, image_type: str) -> \
            tuple[list[ndarray], list[ndarray], ndarray, ndarray]:
        """
        Create the training data array from the given list of paths.
        """
        patches_images = []
        patches_labels = []
        for path in path_list:
            image_data = np.array(nib.load(path).get_fdata())
            original_image_data = image_data
            original_label_data = np.array(nib.load(path.replace('image', 'label')).get_fdata()) \
                if not is_contrastive_dataset else np.empty((0, 0, 0))
            label_data = original_label_data.copy() if not is_contrastive_dataset else original_label_data
            image_data, label_data = DataProcessor.extract_patches(image_data=image_data, label_data=label_data,
                                                                   patch_size=patch_size,
                                                                   is_validation_dataset=is_validation_dataset,
                                                                   patches_filter=patches_filter,
                                                                   is_contrastive_dataset=is_contrastive_dataset,
                                                                   image_type=image_type)
            patches_images.append(image_data)
            patches_labels.append(label_data)

        return patches_images, patches_labels, original_image_data, original_label_data

    @staticmethod
    def get_training_data_from_system(folder_path: str, is_validation_dataset: bool, patch_size: Tuple[int, int, int],
                                      patches_filter: int, image_type: str, is_contrastive_dataset: bool = False) -> \
            Tuple[list[ndarray], list[ndarray], np.ndarray, np.ndarray]:
        """
        Load the training data from the file system.
        """
        image_path_names = glob.glob(os.path.join(folder_path, "*image.nii*"))
        if not image_path_names:
            raise ValueError("Empty list! Check if folder path contains images.")

        return DataProcessor.create_training_data_array(path_list=image_path_names,
                                                        is_validation_dataset=is_validation_dataset,
                                                        patch_size=patch_size, patches_filter=patches_filter,
                                                        is_contrastive_dataset=is_contrastive_dataset,
                                                        image_type=image_type)


class MMWHSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MM-WHS dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], is_validation_dataset: bool,
                 patches_filter: int, image_type: str, mean: Optional[float] = None,
                 std_dev: Optional[float] = None) -> None:
        """
        Initialize the MMWHSDataset for supervised learning.
        """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.is_validation_dataset = is_validation_dataset
        self.patches_filter = patches_filter
        self.image_type = image_type
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
                                          patches_filter=self.patches_filter, image_type=self.image_type)
        img_data, label_data = np.concatenate(img_data, axis=0), np.concatenate(label_data, axis=0)
        print(f"is validation: {self.is_validation_dataset} -> shape of patches array: {img_data.shape}")
        img_data, mean, std_dev = DataProcessor.normalize_z_score_data(raw_data=img_data, is_validation_dataset=self.
                                                                       is_validation_dataset, mean=self.mean,
                                                                       std_dev=self.std_dev)
        label_data, num_classes, label_values = DataProcessor.preprocess_label_data(raw_data=label_data)
        return torch.from_numpy(img_data), torch.from_numpy(label_data), num_classes, label_values, \
            original_image_data, original_label_data, mean, std_dev


class MMWHSLocalContrastiveDataset(Dataset):
    """
        Custom PyTorch Dataset for loading MM-WHS local contrastive learning dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], removal_percentage: float, image_type: str):
        """
            Initialize the MMWHSDataset for contrastive learning.
            """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.removal_percentage = removal_percentage
        self.image_type = image_type
        self.transform = Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            RandGaussianNoise(prob=0.5),
            RandGaussianSmooth(prob=0.5),
            ToTensor()
        ])
        self.x, self.original_image_data, self.mean, self.std_dev = self.load_data()

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        positive_pair = self.transform(self.x[idx]), self.transform(self.x[idx])  # self.x[idx].shape = (1,96^3)
        positive_label = torch.tensor(1.0)

        negative_idx = torch.randint(0, len(self.x), (1,)).item()
        while negative_idx == idx:
            negative_idx = torch.randint(0, len(self.x), (1,)).item()
        negative_pair = self.transform(self.x[negative_idx]), self.transform(self.x[idx])
        negative_label = torch.tensor(0.0)

        if torch.rand(1).item() > 0.5:
            return positive_pair, positive_label
        else:
            return negative_pair, negative_label

    def load_data(self):
        """
        Load and preprocess the dataset.
        """
        img_data, _, original_image_data, _ = DataProcessor. \
            get_training_data_from_system(folder_path=self.folder_path, is_validation_dataset=False,
                                          patch_size=self.patch_size, patches_filter=0, is_contrastive_dataset=True,
                                          image_type=self.image_type)
        img_data = np.concatenate(img_data, axis=0)

        print(f"SHAPE OF UNFILTERED IMG_DATA: {img_data.shape}, MEAN: {np.mean(img_data)}")
        mean_intensity_per_patch = np.mean(img_data, axis=(1, 2, 3, 4))
        num_patches_to_remove = int(self.removal_percentage * len(mean_intensity_per_patch))
        ascending_sorted_patch_indices = np.argsort(mean_intensity_per_patch)
        img_data = img_data[ascending_sorted_patch_indices]
        img_data = img_data[num_patches_to_remove:]
        img_data, mean, std_dev = DataProcessor.normalize_z_score_data(raw_data=img_data)
        print(f"SHAPE OF FILTERED IMG_DATA: {img_data.shape}, MEAN: {mean}")

        for idx, _ in enumerate(img_data):
            torch.from_numpy(img_data[idx])
        return img_data, original_image_data, mean, std_dev


class MMWHSDomainContrastiveDataset(Dataset):
    """
        Custom PyTorch Dataset for loading MM-WHS domain-specific contrastive learning dataset.
    """

    def __init__(self, folder_path: str, patch_size: Tuple[int, int, int], image_type: str, is_distance_adjusted: bool):
        """
            Initialize the MMWHSDataset for contrastive learning.
            """
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.image_type = image_type
        self.is_distance_adjusted = is_distance_adjusted
        self.transform = Compose([
            RandFlip(spatial_axis=0, prob=0.5),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            RandGaussianNoise(prob=0.5),
            RandGaussianSmooth(prob=0.5),
            ToTensor()
        ])
        self.x, self.original_image_data, self.num_of_partitions, self.number_of_imgs = self.load_data()

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        img_idx = idx // self.num_of_partitions
        idx_position = idx
        while idx_position >= self.num_of_partitions:
            idx_position -= self.num_of_partitions
        rand_other_img_idx = torch.randint(0, self.number_of_imgs, (1,)).item()
        while rand_other_img_idx == img_idx:
            rand_other_img_idx = torch.randint(0, self.number_of_imgs, (1,)).item()

        if self.is_distance_adjusted:
            second_img_idx_position = torch.randint(0, self.num_of_partitions, (1,)).item()
            pair = self.transform(self.x[idx]), self.transform(self.x[second_img_idx_position +
                                                                      self.num_of_partitions * rand_other_img_idx])
            distance = abs(second_img_idx_position - idx_position)
            return pair, distance

        else:
            positive_pair = self.transform(self.x[idx]), \
                self.transform(self.x[idx_position + self.num_of_partitions * rand_other_img_idx])
            positive_label = torch.tensor(1.0)

            negative_idx_position = torch.randint(0, self.num_of_partitions, (1,)).item()
            while abs(negative_idx_position - idx_position) <= 25:
                negative_idx_position = torch.randint(0, self.num_of_partitions, (1,)).item()
            negative_pair = self.transform(self.x[idx]), \
                self.transform(self.x[negative_idx_position + self.num_of_partitions * rand_other_img_idx])
            negative_label = torch.tensor(0.0)

            if torch.rand(1).item() > 0.5:
                return positive_pair, positive_label
            else:
                return negative_pair, negative_label

    def load_data(self):
        """
        Load and preprocess the dataset.
        """
        img_data, _, original_image_data, _ = DataProcessor. \
            get_training_data_from_system(folder_path=self.folder_path, is_validation_dataset=False,
                                          patch_size=self.patch_size, patches_filter=0, is_contrastive_dataset=True,
                                          image_type=self.image_type)
        if self.image_type == "MRI":
            target_val = 120
            img_data = [arr for arr in img_data if (arr.shape[3] == 512 and arr.shape[4] >= target_val)]
        else:
            target_val = 230
            img_data = [arr for arr in img_data if (arr.shape[4] >= target_val)]
        for idx, arr in enumerate(img_data):
            remove_from_start = (arr.shape[4] - target_val) // 2
            remove_from_end = arr.shape[4] - target_val - remove_from_start
            img_data[idx] = arr[:, :, :, :, remove_from_start:arr.shape[4] - remove_from_end]
        img_data = np.concatenate(img_data, axis=0)
        print(f"DOMAIN CONTRASTIVE, SHAPE OF DATA: {img_data.shape}")
        num_of_partitions = 512 // self.patch_size[2]
        number_of_imgs = img_data.shape[0] // num_of_partitions

        for idx, _ in enumerate(img_data):
            torch.from_numpy(img_data[idx])
        return img_data, original_image_data, num_of_partitions, number_of_imgs
