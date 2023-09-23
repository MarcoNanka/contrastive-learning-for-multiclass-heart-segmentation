import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset
import time
import numpy as np
from typing import Tuple
from config import parse_args
import wandb
import os
import random

os.environ['WANDB_CACHE_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_CONFIG_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_DIR'] = "$HOME/wandb_tmp"
os.environ['WANDB_TEMP'] = "$HOME/wandb_tmp"


class Trainer:
    def __init__(self, model, dataset, num_epochs, batch_size, learning_rate, validation_dataset,
                 validation_interval, training_shuffle, patch_size):
        """
        Trainer class for training a model.

        Args:
            model (torch.nn.Module): The model to train.
            dataset (torch.utils.data.Dataset): The training dataset.
            num_epochs (int): The number of training epochs.
            batch_size (int, optional): The batch size for training. Default is 4.
            learning_rate (float, optional): The learning rate for the optimizer. Default is 0.001.
            validation_dataset (torch.utils.data.Dataset, optional): The validation dataset. Default is None.
            validation_interval (int, optional): The number of epochs between each validation evaluation. Default is 5.
        """
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

    def reconstruct_labels(self, predicted: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original label data from the extracted label patches.

        Args:
            predicted (np.ndarray): An array of extracted label patches.

        Returns:
            np.ndarray: The reconstructed original label data.
        """
        num_patches, _, patch_dim_x, patch_dim_y, patch_dim_z = predicted.shape
        og_shape = self.validation_dataset.original_image_data.shape
        reconstructed_label = np.zeros(og_shape)

        patch_idx = 0
        for x in range(0, og_shape[0], self.patch_size[0]):
            for y in range(0, og_shape[1], self.patch_size[1]):
                for z in range(0, og_shape[2], self.patch_size[2]):
                    label_patch = predicted[patch_idx]
                    reconstructed_label[x: x + patch_dim_x, y: y + patch_dim_y,
                                        z: z + patch_dim_z] = label_patch[0]
                    patch_idx += 1

        return reconstructed_label

    def evaluate_validation(self) -> Tuple[np.ndarray, float, np.ndarray,
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            tuple: Validation loss, validation accuracy, validation precision, and validation recall.
        """
        self.model.eval()

        val_dataloader = DataLoader(dataset=self.validation_dataset, batch_size=self.batch_size, shuffle=False)

        val_criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_classes = self.dataset.num_classes

        true_positives = np.zeros(num_classes)
        false_positives = np.zeros(num_classes)
        false_negatives = np.zeros(num_classes)
        true_negatives = np.zeros(num_classes)

        with torch.no_grad():
            for val_batch_x, val_batch_y in val_dataloader:
                val_batch_x = val_batch_x.to(device=self.device, dtype=torch.float)  # shape: batch_size, 1, patch_dims
                val_batch_y = val_batch_y.to(device=self.device, dtype=torch.long)  # shape: batch_size, patch_dims
                val_outputs = self.model(val_batch_x)  # shape: batch_size, num_labels, patch_dims
                val_loss = val_criterion(input=val_outputs, target=val_batch_y)
                total_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, dim=1)

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

        # prediction_mask = self.reconstruct_labels(predicted)
        prediction_mask = np.zeros(self.validation_dataset.original_image_data.shape)

        return true_positives, average_loss, accuracy_macro, precision_macro, recall_macro, dice_score_macro, \
            accuracy, precision, recall, dice_score, prediction_mask

    def train(self):
        """
        Trains the model using the provided dataset.

        Prints the loss and validation performance at the end of each epoch.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.model.to(device=self.device, dtype=torch.float)
        # Calculate the number of training patches to use for this epoch (80% of the total)
        num_patches = len(self.dataset)
        print(f"Number of patches: {num_patches}")
        num_patches_to_use = int(self.training_shuffle * num_patches)
        class_labels = {
            0: "background",
            205: "myocardium of the left ventricle",
            420: "left atrium blood cavity",
            500: "left ventricle blood cavity",
            550: "right atrium blood cavity",
            600: "right ventricle blood cavity",
            820: "ascending aorta",
            850: "pulmonary artery"
        }

        for epoch in range(self.num_epochs):
            # Shuffle the dataset to ensure random patch selection
            indices = list(range(num_patches))
            random.shuffle(indices)
            # Create a dataloader for this epoch with the selected patches
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
                    "Epoch": epoch,
                    "Training Loss": loss.item()
                })

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.5f}')

            if (epoch + 1) % self.validation_interval == 0 and self.validation_dataset is not None:
                tp, validation_loss, accuracy_macro, precision_macro, recall_macro, dice_score_macro, \
                    accuracy, precision, recall, dice_score, prediction_mask = self.evaluate_validation()
                wandb.log({
                    "Epoch": epoch,
                    "Validation Loss": validation_loss,
                    "Validation Dice": dice_score_macro,
                    "my_image_key": wandb.Image(self.validation_dataset.original_image_data[:][:][100], masks={
                        "predictions": {
                            "mask_data": prediction_mask[:][:][100],
                            "class_labels": class_labels
                        },
                        "ground_truth": {
                            "mask_data": self.validation_dataset.original_label_data[:][:][100],
                            "class_labels": class_labels
                        }
                    })
                    })
                print(f'Dice score macro: {dice_score_macro}')
                print(f'Dice score by class: {dice_score}')
                print(f'True positives: {tp}')
                print()

        torch.save(self.model.state_dict(), 'trained_model.pth')


def main(args):
    dataset = MMWHSDataset(folder_path=args.folder_path, patch_size=args.patch_size, is_validation_dataset=False,
                           patches_filter=args.patches_filter, normalization_percentiles=args.normalization_percentiles)
    validation_dataset = MMWHSDataset(folder_path=args.val_folder_path, patch_size=args.patch_size,
                                      is_validation_dataset=True, patches_filter=args.patches_filter,
                                      normalization_percentiles=args.normalization_percentiles)
    number_of_channels = dataset.x.shape[1]
    model = UNet(in_channels=number_of_channels, num_classes=dataset.num_classes)
    start_train = time.process_time()
    wandb.login(key="ef43996df858440ef6e65e9f7562a84ad0c407ea")
    wandb.init(
        entity="marco-n",
        project="local-contrastive-learning",
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patch_size": args.patch_size,
            "validation_interval": args.validation_interval,
            "training_shuffle": args.training_shuffle,
            "normalization_percentiles": args.normalization_percentiles
        }
    )
    config = wandb.config

    trainer = Trainer(model=model, dataset=dataset, num_epochs=config.num_epochs, batch_size=config.batch_size,
                      learning_rate=config.learning_rate, validation_dataset=validation_dataset,
                      validation_interval=config.validation_interval, training_shuffle=config.training_shuffle,
                      patch_size=config.patch_size)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
