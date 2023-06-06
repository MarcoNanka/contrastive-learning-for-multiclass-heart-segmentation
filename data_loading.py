import os
from typing import Union, List, Tuple, Callable, Dict
import numpy as np
import torch
import nibabel as nib
import glob
import math


class MMWHSDataset():
    def __init__(
            self,
            raw_data_dir: str, subfolders: Union[tuple, list, str],
            n_of_slices_per_image
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.subfolders = subfolders
        self.data = self.load_data(n_of_slices_per_image)

    def load_data(self, n_of_slices_per_image: int):
        directory = self.raw_data_dir
        file_path_names = []
        for subfolder in self.subfolders:
            for idx, filename in enumerate(glob.glob(os.path.join(directory + subfolder, "*.nii*"))):
                with open(filename, 'r') as f:
                    file_path_names.append(filename)

        img_data = np.array([])
        for i, path in enumerate(file_path_names):
            buf = np.array(nib.load(path).get_fdata())
            n_of_slices = buf.shape[-1]
            n_of_batches = math.ceil(n_of_slices / n_of_slices_per_image)
            buf.resize([buf.shape[0], buf.shape[1], n_of_slices_per_image, n_of_batches])
            for j in range(n_of_batches):
                if i == 0 and j == 0:
                    img_data = np.array(buf[:, :, :, j])
                    img_data = img_data[:, :, :, np.newaxis]
                else:
                    img_data = np.append(img_data, buf[:, :, :, j, np.newaxis], 3)

        return torch.from_numpy(img_data)


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/even more reduced MM-WHS 2017 Dataset/"
    dataset = MMWHSDataset(main_dir, ("ct_train", "ct_test"), 10)
