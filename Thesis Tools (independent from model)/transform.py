import os
from monai.transforms import Compose, RandFlip, RandGaussianNoise, RandGaussianSmooth, ToTensor
from torchvision import transforms
from torchvision.transforms import GaussianBlur, RandomVerticalFlip, RandomHorizontalFlip, RandomPerspective, \
    RandomGrayscale, ColorJitter, RandomAffine, RandomErasing
from PIL import Image
import numpy as np
import torch
import random


def apply_transform(input_file, output_dir, input_path):
    print(input_file.shape)

    transform = transforms.Compose([
        RandomHorizontalFlip(p=1),
        RandomVerticalFlip(p=1),
        ColorJitter(),
        RandomErasing()
    ])

    transformed_image = transform(input_file)
    print(transformed_image.shape)

    transformed_image_np = transformed_image.numpy()
    print(transformed_image_np.shape)

    output_image = Image.fromarray(transformed_image_np, 'RGB')
    rand_int = random.randint(0, 99)
    output_file = os.path.join(output_dir, os.path.basename(input_path))
    file_name, file_extension = os.path.splitext(output_file)
    output_file_with_rand = f"{file_name}_{rand_int}{file_extension}"
    output_image.save(output_file_with_rand)


if __name__ == "__main__":
    input_directory = "/Users/marconanka/BioMedia/Thesis/figures/contr_ex"
    output_directory = "/Users/marconanka/BioMedia/Thesis/figures/contr_ex_transformed"
    os.makedirs(output_directory, exist_ok=True)
    input_files = [f for f in os.listdir(input_directory) if f.endswith(".jpeg")]
    for input_file in input_files:
        input_path = os.path.join(input_directory, input_file)
        image = Image.open(input_path)
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array)
        apply_transform(image_tensor, output_directory, input_file)
