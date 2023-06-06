"""
This code defines a class for the Multi-Modality Whole Heart Segmentation (MMWHS) dataset,
which provides data loading functionality.

def load_data - loads images into one 4D pytorch tensor (512: width x 512: height x
n_of_slices_per_img: user input x total_number_of_slices/n_of_slices_per_img);
"""

# standard library imports
import os
from typing import Union
import glob
import math
# third party imports
import numpy as np
import torch
import nibabel as nib
# local application imports
from config import args


class MMWHSDataset:
    def __init__(
            self,
            raw_data_dir: str, subfolders: Union[tuple, list, str],
            n_of_slices_per_img
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.subfolders = subfolders
        self.data = self.load_data(n_of_slices_per_img)

    def load_data(self, n_of_slices_per_img: int):
        directory = self.raw_data_dir
        file_path_names = []
        for subfolder in self.subfolders:
            for idx, filename in enumerate(glob.glob(os.path.join(directory + subfolder, "*.nii*"))):
                with open(filename, 'r'):
                    file_path_names.append(filename)

        img_data = np.array([])
        for i, path in enumerate(file_path_names):
            buf = np.array(nib.load(path).get_fdata())
            n_of_slices = buf.shape[-1]
            n_of_batches = math.ceil(n_of_slices / n_of_slices_per_img)
            buf.resize([buf.shape[0], buf.shape[1], n_of_slices_per_img, n_of_batches])
            for j in range(n_of_batches):
                if i == 0 and j == 0:
                    img_data = np.array(buf[:, :, :, j])
                    img_data = img_data[:, :, :, np.newaxis]
                else:
                    img_data = np.append(img_data, buf[:, :, :, j, np.newaxis], 3)

        return torch.from_numpy(img_data)


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/even more reduced MM-WHS 2017 Dataset/"
    dataset = MMWHSDataset(main_dir, ("ct_train", "ct_test"), args.n_of_slices_per_image)
    print(dataset.data.shape)
