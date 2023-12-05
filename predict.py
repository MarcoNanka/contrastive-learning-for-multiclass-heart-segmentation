import torch
import nibabel as nib
from model import UNet
from data_loading import DataProcessor
import numpy as np
from config import parse_args
from torch.utils.data import DataLoader
from data_loading import MMWHSDataset
import glob
import os


class Predictor:
    def __init__(self, model_name, patch_size, output_mask_name, mean, std_dev, image_type, batch_size,
                 folder_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.output_mask_name = output_mask_name
        self.patch_size = patch_size
        self.mean = mean
        self.std_dev = std_dev
        self.image_type = image_type
        self.batch_size = batch_size
        self.folder_path = folder_path

    def predict(self):
        # Load model
        model = UNet()
        model.load_state_dict(torch.load("trained_unet/" + self.model_name))
        model.eval()
        print(f"FINISHED LOAD MODEL")

        # Load and preprocess the input image
        image_path_names = sorted(glob.glob(os.path.join(self.folder_path, "*image.nii*")))
        dataset = MMWHSDataset(patch_size=self.patch_size, image_type=self.image_type,
                               is_validation_dataset=True, img_path_names=image_path_names, patches_filter=0,
                               mean=self.mean, std_dev=self.std_dev)
        print(f"FINISHED LOAD & PREPROCESS INPUT IMAGE")

        # Perform prediction
        model.to(device=self.device, dtype=torch.float)
        predicted_arrays_list = []
        true_positives = np.zeros(8)
        false_positives = np.zeros(8)
        false_negatives = np.zeros(8)
        true_negatives = np.zeros(8)

        predict_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for predict_batch_x, predict_batch_y in predict_dataloader:
                predict_batch_x = predict_batch_x.to(device=self.device, dtype=torch.float)
                predict_batch_y = predict_batch_y.to(device=self.device, dtype=torch.long)
                predicted_output = model(predict_batch_x)
                _, predicted = torch.max(predicted_output, dim=1)
                predicted_arrays_list.append(predicted.cpu().numpy())
                for class_idx in range(8):
                    true_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                   torch.eq(predict_batch_y, class_idx)).sum().item()
                    false_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                    torch.ne(predict_batch_y, class_idx)).sum().item()
                    false_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                    torch.eq(predict_batch_y, class_idx)).sum().item()
                    true_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                   torch.ne(predict_batch_y, class_idx)).sum().item()

        dice_score = (2 * true_positives) / (2 * true_positives + false_negatives + false_positives)
        dice_score_macro = np.mean(dice_score)
        combined_predicted_array = np.concatenate(predicted_arrays_list, axis=0)
        prediction_mask = DataProcessor.undo_extract_patches_label_only(label_patches=combined_predicted_array,
                                                                        patch_size=self.patch_size,
                                                                        original_label_data=dataset.original_label_data)
        print(f"DICE SCORE MACRO: {dice_score_macro}")
        print(f"FINISHED PERFORM PREDICTION")

        # Save the predicted mask as a NIfTI file
        label_values = np.array([0., 205., 420., 500., 550., 600., 820., 850.])
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
    image_type = "CT"
    if "mr" in args.folder_path:
        image_type = "MRI"
    predictor = Predictor(model_name=args.model_name, patch_size=args.patch_size,
                          output_mask_name=args.output_mask_name, mean=args.mean, std_dev=args.std_dev,
                          image_type=image_type, batch_size=args.batch_size, folder_path=args.folder_path)
    predictor.predict()


if __name__ == "__main__":
    args = parse_args()
    main(args)
