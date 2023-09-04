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

WANDB_CACHE_DIR = "$HOME/wandb_tmp"
WANDB_CONFIG_DIR = "$HOME/wandb_tmp"
WANDB_DIR = "$HOME/wandb_tmp"


class Trainer:
    def __init__(self, model, dataset, num_epochs, batch_size=4, learning_rate=0.001, validation_dataset=None,
                 validation_interval=5):
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

    def evaluate_validation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray,
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray]:
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
                val_batch_x = val_batch_x.to(device=self.device, dtype=torch.float)
                val_batch_y = val_batch_y.to(device=self.device, dtype=torch.long)
                val_outputs = self.model(val_batch_x)
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

        return true_positives, false_positives, true_negatives, false_negatives, average_loss, accuracy_macro, \
            precision_macro, recall_macro, dice_score_macro, accuracy, precision, recall, dice_score

    def train(self):
        """
        Trains the model using the provided dataset.

        Prints the loss and validation performance at the end of each epoch.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        self.model.to(device=self.device, dtype=torch.float)

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float)
                batch_y = batch_y.to(device=self.device, dtype=torch.long)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(input=outputs, target=batch_y)
                loss.backward()
                optimizer.step()
                wandb.log({"Training Loss": loss.item()})

            if (epoch + 1) % self.validation_interval == 0 and self.validation_dataset is not None:
                tp, fp, tn, fn, validation_loss, accuracy_macro, precision_macro, recall_macro, dice_score_macro, \
                    accuracy, precision, recall, dice_score = self.evaluate_validation()
                wandb.log({
                    "Validation Loss": validation_loss,
                    # "Validation Accuracy": accuracy_macro,
                    # "Validation Precision": precision_macro,
                    # "Validation Recall": recall_macro,
                    "Validation Dice": dice_score_macro,
                    # "True Positives label 1": tp[1],
                    # "True Positives label 2": tp[2],
                    # "True Positives label 3": tp[3],
                    # "True Positives label 4": tp[4],
                    # "True Positives label 5": tp[5],
                    # "True Positives label 6": tp[6],
                    # "True Positives label 7": tp[7],
                })
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.5f},')
                # print(f'Validation Loss: {validation_loss:.5f},')
                # print(f'Precision macro: {precision_macro:.5f},')
                # print(f'Recall macro: {recall_macro:.5f},')
                # print(f'Accuracy macro: {accuracy_macro:.5f}')
                print(f'Dice score macro: {dice_score_macro}')
                # print(f'Precision by class: {precision}')
                # print(f'Recall by class: {recall}')
                # print(f'Accuracy by class: {accuracy}')
                print(f'Dice score by class: {dice_score}')
                # print(f'TP: {tp}')
                # print(f'FP: {fp}')
                # print(f'TN: {tn}')
                # print(f'FN: {fn}')
                print()
            else:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.5f}')

        torch.save(self.model.state_dict(), 'trained_model.pth')


def main(args):
    dataset = MMWHSDataset(folder_path=args.folder_path, patch_size=args.patch_size, is_validation_dataset=False)
    validation_dataset = MMWHSDataset(folder_path=args.val_folder_path, patch_size=args.patch_size,
                                      is_validation_dataset=True)
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
            "validation_interval": args.validation_interval,
        }
    )
    config = wandb.config

    trainer = Trainer(model=model, dataset=dataset, num_epochs=config.num_epochs, batch_size=config.batch_size,
                      learning_rate=config.learning_rate, validation_dataset=validation_dataset,
                      validation_interval=config.validation_interval)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
