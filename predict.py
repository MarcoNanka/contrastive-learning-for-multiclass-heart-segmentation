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
        print(f"UNIQUE LABEL VALUES (ground truth): {np.unique(label_data), label_data.shape}")
        img_data = img_data.to(device=self.device, dtype=torch.float)
        label_data = label_data.to(device=self.device, dtype=torch.long)
        model.to(device=self.device, dtype=torch.float)
        predicted_arrays_list = []
        true_positives = np.zeros(8)
        false_positives = np.zeros(8)
        false_negatives = np.zeros(8)
        true_negatives = np.zeros(8)

        with torch.no_grad():
            step_size = 10
            for i in range(0, img_data.shape[0], step_size):
                predicted_output = model(img_data[i:i + step_size])
                _, predicted = torch.max(predicted_output, dim=1)
                predicted_arrays_list.append(predicted.cpu().numpy())
                for class_idx in range(8):
                    true_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                   torch.eq(label_data[i:i + step_size],
                                                                            class_idx)).sum().item()
                    false_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                    torch.ne(label_data[i:i + step_size],
                                                                             class_idx)).sum().item()
                    false_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                    torch.eq(label_data[i:i + step_size],
                                                                             class_idx)).sum().item()
                    true_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                   torch.ne(label_data[i:i + step_size],
                                                                            class_idx)).sum().item()

        dice_score = (2 * true_positives) / (2 * true_positives + false_negatives + false_positives)
        dice_score_macro = np.mean(dice_score)
        combined_predicted_array = np.concatenate(predicted_arrays_list, axis=0)
        prediction_mask = DataProcessor.undo_extract_patches_label_only(label_patches=combined_predicted_array,
                                                                        patch_size=self.patch_size,
                                                                        original_label_data=original_label_data)
        print(f"UNIQUE LABEL VALUES (prediction mask): {np.unique(prediction_mask), combined_predicted_array.shape}")
        print(f"DICE SCORE MACRO: {dice_score_macro}")
        print(f"FINISHED PERFORM PREDICTION")

        # Save the predicted mask as a NIfTI file
        label_values = np.array([0., 205., 420., 500., 550., 600., 620., 850.])
        prediction_mask = np.where(prediction_mask == 0, label_values[0], prediction_mask)
        prediction_mask = np.where(prediction_mask == 1, label_values[1], prediction_mask)
        prediction_mask = np.where(prediction_mask == 2, label_values[2], prediction_mask)
        prediction_mask = np.where(prediction_mask == 3, label_values[3], prediction_mask)
        prediction_mask = np.where(prediction_mask == 4, label_values[4], prediction_mask)
        prediction_mask = np.where(prediction_mask == 5, label_values[5], prediction_mask)
        prediction_mask = np.where(prediction_mask == 6, label_values[6], prediction_mask)
        prediction_mask = np.where(prediction_mask == 7, label_values[7], prediction_mask)
        output_nifti = nib.Nifti1Image(prediction_mask, affine=np.eye(4))
        nib.save(output_nifti, "prediction_masks/" + self.output_mask_name)

        print(f"Predicted mask saved at: prediction_masks/{self.output_mask_name}")


def main(args):
    predictor = Predictor(model_name=args.model_name, image_path=args.image_path, patch_size=args.patch_size,
                          output_mask_name=args.output_mask_name)
    predictor.predict()


if __name__ == "__main__":
    args = parse_args()
    main(args)
