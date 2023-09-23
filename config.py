import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net model for semantic segmentation")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the training data folder")
    parser.add_argument("--val_folder_path", type=str, required=True, help="Path to the validation data folder")
    parser.add_argument("--patch_size", type=int, nargs=3, help="Patch size (depth, height, width)")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--validation_interval", type=int, help="Validation interval in epochs")
    parser.add_argument("--patches_filter", type=int, help="Number of relevant voxels")
    parser.add_argument("--training_shuffle", type=float, help="Percentage of training patches used for each epoch")
    parser.add_argument("--normalization_percentiles", type=int, nargs=2, help="Min and max percentile values for "
                                                                               "normalization of data")
    args = parser.parse_args()
    return args
