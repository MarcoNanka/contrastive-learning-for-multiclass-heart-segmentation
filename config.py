import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net model for semantic segmentation")
    parser.add_argument("--folder_path", type=str, help="Path to the training data folder")
    parser.add_argument("--test_folder_path", type=str, help="Path to the test data folder")
    parser.add_argument("--training_dataset_size", type=int, help="Number of images used for fine-tuning, MAX: 8")
    parser.add_argument("--patch_size", type=int, nargs=3, help="Patch size (depth, height, width)")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--validation_interval", type=int, help="Validation interval in epochs")
    parser.add_argument("--patches_filter", type=int, help="Number of relevant voxels")
    parser.add_argument("--training_shuffle", type=float, help="Percentage of training patches used for each epoch")
    parser.add_argument("--model_name", type=str, help="Name under which encoder/UNET will be saved")
    parser.add_argument("--encoder_file_name", type=str, help="Name of pre-trained encoder file which shall be used for"
                                                              " supervised training")
    parser.add_argument("--image_path", type=str, help="Path to image for predictor")
    parser.add_argument("--output_mask_name", type=str, help="Name of predicted output mask")
    parser.add_argument("--mean", type=float, help="Mean from loaded UNET for predictor")
    parser.add_argument("--std_dev", type=float, help="Standard_deviation from loaded UNET for predictor")
    parser.add_argument("--patience", type=int, help="Threshold which determines when to early-stop training")
    parser.add_argument("--removal_percentage", type=float, help="Determines what percentage of patches (with lowest "
                                                                 "mean) should be removed from contrastive dataset")
    parser.add_argument("--contrastive_type", type=str, help="Determines if domain-specific or local")
    parser.add_argument("--temperature", type=float, help="Adjusts softness of contrastive loss function")
    args = parser.parse_args()
    return args
