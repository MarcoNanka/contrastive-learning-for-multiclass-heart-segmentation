import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset, DataProcessor
import numpy as np
from typing import Tuple, Dict
from config import parse_args
import wandb
import os
import random
import glob

os.environ['WANDB_CACHE_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_CONFIG_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_TEMP'] = "$HOME/wandb_tmp"


class Trainer:
    def __init__(self, model, dataset, num_epochs, batch_size, learning_rate, validation_dataset,
                 validation_interval, training_shuffle, patch_size, model_name, patience):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_dataset = validation_dataset
        self.validation_interval = validation_interval
        self.training_shuffle = training_shuffle
        self.patch_size = patch_size
        self.model_name = model_name
        self.patience = patience
        self.class_labels = {
            0: "background",
            1: "myocardium of the left ventricle",  # 205
            2: "left atrium blood cavity",  # 420
            3: "left ventricle blood cavity",  # 500
            4: "right atrium blood cavity",  # 550
            5: "right ventricle blood cavity",  # 600
            6: "ascending aorta",  # 820
            7: "pulmonary artery"  # 850
        }

    def evaluate(self, dataset, batch_size, best_model_state=None) \
            -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                     np.ndarray, np.ndarray, np.ndarray]:
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.model.eval()

        val_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        val_criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_classes = self.dataset.num_classes

        true_positives = np.zeros(num_classes)
        false_positives = np.zeros(num_classes)
        false_negatives = np.zeros(num_classes)
        true_negatives = np.zeros(num_classes)

        predicted_arrays_list = []

        with torch.no_grad():
            for val_batch_x, val_batch_y in val_dataloader:
                val_batch_x = val_batch_x.to(device=self.device, dtype=torch.float)
                val_batch_y = val_batch_y.to(device=self.device, dtype=torch.long)
                val_outputs = self.model(val_batch_x)
                val_loss = val_criterion(input=val_outputs, target=val_batch_y)
                total_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, dim=1)
                predicted_arrays_list.append(predicted.cpu().numpy())

                for class_idx in range(num_classes):
                    true_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                   torch.eq(val_batch_y, class_idx)).sum().item()
                    false_positives[class_idx] += torch.logical_and(torch.eq(predicted, class_idx),
                                                                    torch.ne(val_batch_y, class_idx)).sum().item()
                    false_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                    torch.eq(val_batch_y, class_idx)).sum().item()
                    true_negatives[class_idx] += torch.logical_and(torch.ne(predicted, class_idx),
                                                                   torch.ne(val_batch_y, class_idx)).sum().item()

            average_loss = total_loss / len(val_dataloader)
            accuracy = (true_positives + true_negatives) / \
                       (true_positives + true_negatives + false_negatives + false_positives)
            dice_score = (2 * true_positives) / (2 * true_positives + false_negatives + false_positives)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            precision_macro = np.mean(precision)
            recall_macro = np.mean(recall)
            accuracy_macro = np.mean(accuracy)
            dice_score_macro = np.mean(dice_score)

        combined_predicted_array = np.concatenate(predicted_arrays_list, axis=0)
        prediction_mask = DataProcessor.\
            undo_extract_patches_label_only(label_patches=combined_predicted_array,
                                            patch_size=self.patch_size, original_label_data=dataset.original_label_data)

        return true_positives, average_loss, accuracy_macro, precision_macro, recall_macro, dice_score_macro, \
            accuracy, precision, recall, dice_score, prediction_mask

    def train(self) -> Dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.model.to(device=self.device, dtype=torch.float)
        num_patches = len(self.dataset)
        num_patches_to_use = int(self.training_shuffle * num_patches)
        og_labels_int, _, _ = DataProcessor.preprocess_label_data(self.validation_dataset.original_label_data)
        no_improvement_counter = 0
        best_dice_score = 0.0
        best_model_state = self.model.state_dict()

        for epoch in range(self.num_epochs):
            indices = list(range(num_patches))
            random.shuffle(indices)
            selected_indices = indices[:num_patches_to_use]
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False,
                                    sampler=torch.utils.data.SubsetRandomSampler(selected_indices))
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float)
                batch_y = batch_y.to(device=self.device, dtype=torch.long)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(input=outputs, target=batch_y)
                loss.backward()
                optimizer.step()

            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": loss.item()
            })
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.5f}')

            if (epoch + 1) % self.validation_interval == 0 and self.validation_dataset is not None:
                tp, validation_loss, _, _, _, dice_score_macro, _, _, _, dice_score, prediction_mask = \
                    self.evaluate(self.validation_dataset, batch_size=self.batch_size)
                if dice_score_macro > best_dice_score:
                    best_dice_score = dice_score_macro
                    best_model_state = self.model.state_dict()
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                if no_improvement_counter > self.patience:
                    print(f'Early stopping at epoch {epoch + 1}. Best Validation Dice Score: {best_dice_score}')
                    break
                wandb.log({
                    "Epoch": epoch + 1,
                    "Validation Loss": validation_loss,
                    "Validation Dice": dice_score_macro,
                    "Best (baseline: dice, contrastive: loss)": best_dice_score,
                    "slice50": wandb.Image(data_or_path=self.validation_dataset.original_image_data[:, :, 49],
                                           masks={
                                                    "predictions": {
                                                        "mask_data": prediction_mask[:, :, 49],
                                                        "class_labels": self.class_labels
                                                    },
                                                    "ground_truth": {
                                                        "mask_data": og_labels_int[:, :, 49],
                                                        "class_labels": self.class_labels
                                                    }
                                                }),
                    "slice100": wandb.Image(data_or_path=self.validation_dataset.original_image_data[:, :, 99],
                                            masks={
                                                    "predictions": {
                                                        "mask_data": prediction_mask[:, :, 99],
                                                        "class_labels": self.class_labels
                                                    },
                                                    "ground_truth": {
                                                        "mask_data": og_labels_int[:, :, 99],
                                                        "class_labels": self.class_labels
                                                    }
                                                }),
                })
                print(f'Dice score macro: {dice_score_macro}')
                print(f'Dice score by class: {dice_score}')
                print(f'True positives: {tp}')
                print()

        self.model.load_state_dict(best_model_state)
        torch.save(self.model.state_dict(), "trained_unet/" + self.model_name)
        return best_model_state


def main(args):
    # LOAD DATASETS
    image_type = "CT"
    if "mr" in args.folder_path:
        image_type = "MRI"

    image_path_names = sorted(glob.glob(os.path.join(args.folder_path, "*image.nii*")))
    validation_image_path_names = random.sample(image_path_names, 2)
    training_image_path_names = random.sample([item for item in image_path_names if item not in
                                               validation_image_path_names], args.training_dataset_size)
    dataset = MMWHSDataset(img_path_names=training_image_path_names, is_validation_dataset=False,
                           patches_filter=args.patches_filter, patch_size=args.patch_size, image_type=image_type)
    validation_dataset = MMWHSDataset(img_path_names=validation_image_path_names, is_validation_dataset=True,
                                      patches_filter=args.patches_filter, mean=dataset.mean, std_dev=dataset.std_dev,
                                      patch_size=args.patch_size, image_type=image_type)

    # SET UP WEIGHTS & BIASES
    wandb.login(key="ef43996df858440ef6e65e9f7562a84ad0c407ea")
    wandb.init(
        entity="marco-n",
        project="contrastive-learning-for-multiclass-heart-segmentation",
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patch_size": args.patch_size,
            "validation_interval": args.validation_interval,
            "training_shuffle": args.training_shuffle,
            "filter": args.patches_filter,
            "mean": dataset.mean,
            "std_dev": dataset.std_dev,
            "patience": args.patience,
            "training type": "BASELINE",
            "image_type": image_type,
            "model_name": args.model_name,
            "folder_path": args.folder_path,
            "training_dataset_size": args.training_dataset_size,
            "encoder_file_name": args.encoder_file_name
        }
    )

    # DO SUPERVISED LEARNING
    if os.path.isfile("pretrained_encoder/" + args.encoder_file_name):
        pretrained_encoder = torch.load("pretrained_encoder/" + args.encoder_file_name)
        encoder_weights, encoder_biases = pretrained_encoder['encoder_weights'], pretrained_encoder['encoder_biases']
        print("Pre-trained encoder is LOADED")
    else:
        encoder_weights, encoder_biases = None, None
        print("Pre-trained encoder does NOT EXIST")
    model = UNet(encoder_weights=encoder_weights, encoder_biases=encoder_biases)
    trainer = Trainer(model=model, dataset=dataset, num_epochs=args.num_epochs, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, validation_dataset=validation_dataset,
                      validation_interval=args.validation_interval, training_shuffle=args.training_shuffle,
                      patch_size=args.patch_size, model_name=args.model_name, patience=args.patience)
    best_model_state = trainer.train()

    # EVALUATE MODEL
    test_image_path_names = sorted(glob.glob(os.path.join(args.folder_path, "*image.nii*")))
    test_dataset = MMWHSDataset(img_path_names=test_image_path_names, is_validation_dataset=True,
                                patches_filter=args.patches_filter, mean=dataset.mean, std_dev=dataset.std_dev,
                                patch_size=args.patch_size, image_type=image_type)
    tp_test, _, _, _, _, dice_score_macro_test, _, _, _, dice_score_test, prediction_mask_test = \
        trainer.evaluate(dataset=test_dataset, best_model_state=best_model_state, batch_size=1)
    og_labels_int, _, _ = DataProcessor.preprocess_label_data(test_dataset.original_label_data)

    print(f"---FINAL EVALUATION ON TEST SET--- (training is finished)")
    print(f"True Positives Test Dataset: {tp_test}")
    print(f"Dice Scores Test Dataset: {dice_score_test}")
    print(f"Macro Dice Test Dataset: {dice_score_macro_test}")
    wandb.log({
        "Test Macro Dice": dice_score_macro_test,
        "Test slice50": wandb.Image(data_or_path=test_dataset.original_image_data[:, :, 49],
                                    masks={
                                        "predictions": {
                                            "mask_data": prediction_mask_test[:, :, 49],
                                            "class_labels": trainer.class_labels
                                        },
                                        "ground_truth": {
                                            "mask_data": og_labels_int[:, :, 49],
                                            "class_labels": trainer.class_labels
                                        }
                                    }),
        "Test slice100": wandb.Image(data_or_path=test_dataset.original_image_data[:, :, 99],
                                     masks={
                                         "predictions": {
                                             "mask_data": prediction_mask_test[:, :, 99],
                                             "class_labels": trainer.class_labels
                                         },
                                         "ground_truth": {
                                             "mask_data": og_labels_int[:, :, 99],
                                             "class_labels": trainer.class_labels
                                         }
                                     }),
    })


if __name__ == "__main__":
    args = parse_args()
    main(args)
