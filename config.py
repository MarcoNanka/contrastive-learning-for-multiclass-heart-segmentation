import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net model for semantic segmentation")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the training data folder")
    parser.add_argument("--val_folder_path", type=str, required=True, help="Path to the validation data folder")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(24, 24, 24), help=
                        "Patch size (depth, height, width)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--validation_interval", type=int, default=2, help="Validation interval in epochs")
    parser.add_argument("--patches_filter", type=int, default=100, help="Number of relevant voxels")
    args = parser.parse_args()
    return args
