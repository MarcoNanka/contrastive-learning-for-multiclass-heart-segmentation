import torch
import nibabel as nib
from model import UNet
from data_loading import DataProcessor
import numpy as np
from config import parse_args


class Predictor:
    def __init__(self, model_name, image_path, patch_size, output_mask_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.image_path = image_path
        self.output_mask_name = output_mask_name
        self.patch_size = patch_size

    def predict(self):
        # Load model
        model = UNet()
        model.load_state_dict(torch.load("trained_unet/" + self.model_name))
        model.eval()
        print(f"FINISHED LOAD MODEL")

        # Load and preprocess the input image
        img_data, label_data, original_image_data, original_label_data = DataProcessor. \
            get_training_data_from_system(folder_path=self.image_path, is_validation_dataset=True,
                                          patch_size=self.patch_size, patches_filter=0)
        img_data, _, _ = DataProcessor.normalize_z_score_data(raw_data=img_data)
        label_data, _, _ = DataProcessor.preprocess_label_data(raw_data=label_data)
        img_data = torch.from_numpy(img_data)
        label_data = torch.from_numpy(label_data)
        print(f"FINISHED LOAD & PREPROCESS INPUT IMAGE, img_data.shape: {img_data.shape}")

        # Perform prediction
        model.to(device=self.device, dtype=torch.float)
        with torch.no_grad():
            predicted_output = model(img_data)
            _, predicted = torch.max(predicted_output, dim=1)

        prediction_mask = DataProcessor.undo_extract_patches_label_only(label_patches=predicted,
                                                                        patch_size=self.patch_size,
                                                                        original_label_data=original_label_data)
        print(f"FINISHED PERFORM PREDICTION")

        # Save the predicted mask as a NIfTI file
        output_nifti = nib.Nifti1Image(prediction_mask, affine=np.eye(4))
        nib.save(output_nifti, "prediction_masks/" + self.output_mask_name)

        print(f"Predicted mask saved at: \"prediction_masks/\"{self.output_mask_name}")


def main(args):
    predictor = Predictor(model_name=args.model_name, image_path=args.image_path, patch_size=args.patch_size,
                          output_mask_name=args.output_mask_name)
    predictor.predict()


if __name__ == "__main__":
    args = parse_args()
    main(args)
