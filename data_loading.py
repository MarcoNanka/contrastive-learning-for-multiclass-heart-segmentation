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


# TODO: 3D instead of 4D?? should be better!
# TODO: seperate data from labels by keeping relationship at the same time

class MMWHSDataset:

    def __init__(
            self,
            raw_data_dir: str, subfolders: Union[tuple, list, str],
            n_of_slices_per_img
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.subfolders = subfolders
        self.data = self.load_data(n_of_slices_per_img)

    def normalize_minmax_data(self, image_data, min_val=1, max_val=99):
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
        min_val_1p = np.percentile(image_data, min_val)
        max_val_99p = np.percentile(image_data, max_val)
        final_image_data = np.zeros((image_data.shape[0], image_data.shape[1],
                                     image_data.shape[2], image_data.shape[3]), dtype=np.float64)
        # min-max norm on total 3D volume
        final_image_data = (image_data - min_val_1p) / (max_val_99p - min_val_1p)
        return final_image_data

    def load_data(self, n_of_slices_per_img: int):
        """
        # MRI and CT scans are loaded and stored into 4D torch tensor (width x height x
        n_of_slices_per_image x math.ceil(n_of_slices / n_of_slices_per_img)).
        input params:
            n_of_slices_per_img: third dimension of tensor, determined by user
        returns:
            img_data: Normalized 4D MRI/CT scans
        """

        def store_file_path_name(dire, subf):
            buf_file_path_names = []
            for idx, filename in enumerate(glob.glob(os.path.join(dire + subf, "*.nii*"))):
                with open(filename, 'r'):
                    buf_file_path_names.append(filename)
            return buf_file_path_names

        directory = self.raw_data_dir
        if isinstance(self.subfolders, tuple) or isinstance(self.subfolders, list):
            for subfolder in self.subfolders:
                file_path_names = store_file_path_name(directory, subfolder)
        else:
            file_path_names = store_file_path_name(directory, self.subfolders)
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
        # print(f"img_data.shape: {img_data.shape}")
        # print(f"img_data[250:260, 0, 0, 0]: {img_data[250:260, 0, 0, 0]}")
        # print(f"index, value of max element in 1.D: {np.argmax(img_data[:, 0, 0, 0])}, "
        #       f"{np.max(img_data[:, 0, 0, 0])}, "
        #       f"min/max value of array: {np.min(img_data), np.max(img_data)}")
        img_data = self.normalize_minmax_data(torch.from_numpy(img_data), 0, 100)
        return img_data


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/even more reduced MM-WHS 2017 Dataset/"
    dataset = MMWHSDataset(main_dir, "mr_test", args.n_of_slices_per_image)
    # print(f"dataset.data.shape: {dataset.data.shape}")
    # print(f"dataset.data[250:260, 0, 0, 0]: {dataset.data[250:260, 0, 0, 0]}")

