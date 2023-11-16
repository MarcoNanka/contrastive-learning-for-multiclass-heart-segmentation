import os
from monai.transforms import Compose, RandFlip, RandGaussianNoise, RandGaussianSmooth, ToTensor
from torchvision import transforms
from torchvision.transforms import GaussianBlur, RandomVerticalFlip, RandomHorizontalFlip, RandomPerspective, \
    RandomGrayscale
from PIL import Image
import numpy as np
import torch


def apply_transform(input_file, output_dir, input_path):
    print(input_file.shape)

    # Define the transformation
    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomPerspective(),
        GaussianBlur(kernel_size=3),
        RandomGrayscale()
    ])

    # Apply the transformation
    transformed_image = transform(transform(transform(input_file)))
    print(transformed_image.shape)

    # Convert the PyTorch tensor to a NumPy array before saving
    transformed_image_np = transformed_image.numpy()
    print(transformed_image_np.shape)

    output_image = Image.fromarray(transformed_image_np, 'RGB')

    # Save the transformed image
    output_file = os.path.join(output_dir, os.path.basename(input_path))
    output_image.save(output_file)


if __name__ == "__main__":
    # Specify the input directory
    input_directory = "/Users/marconanka/BioMedia/Thesis/figures/contr_ex"

    # Specify the output directory
    output_directory = "/Users/marconanka/BioMedia/Thesis/figures/contr_ex_transformed"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all files in the input directory
    input_files = [f for f in os.listdir(input_directory) if f.endswith(".jpeg")]

    # Apply transformations to each file
    for input_file in input_files:
        input_path = os.path.join(input_directory, input_file)

        # Read the JPEG file and convert it to a NumPy array
        image = Image.open(input_path)
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array)

        # Apply transformations to the image array
        apply_transform(image_tensor, output_directory, input_file)
