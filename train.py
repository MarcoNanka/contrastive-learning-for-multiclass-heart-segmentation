import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset
import time
import numpy as np
from typing import Tuple
from config import parse_args


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
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            tuple: Validation loss, validation accuracy, validation precision, and validation recall.
        """
        self.model.eval()

        val_dataloader = DataLoader(dataset=self.validation_dataset, batch_size=self.batch_size, shuffle=False)

        val_criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
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

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            precision_macro = np.mean(precision)
            recall_macro = np.mean(recall)
            accuracy_macro = np.mean(accuracy)

        return true_positives, false_positives, true_negatives, false_negatives, average_loss, accuracy_macro, \
               precision_macro, recall_macro, accuracy, precision, recall

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

            if (epoch + 1) % self.validation_interval == 0 and self.validation_dataset is not None:
                tp, fp, tn, fn, validation_loss, accuracy_macro, precision_macro, recall_macro, accuracy, precision, \
                recall = self.evaluate_validation()
                print(
                    f'Epoch {epoch + 1}/{self.num_epochs}, '
                    f'Loss: {loss.item():.5f}, '
                    f'Validation Loss: {validation_loss:.5f}, '
                    f'Precision macro: {precision_macro:.5f}, '
                    f'Recall macro: {recall_macro:.5f}, '
                    f'Accuracy macro: {accuracy_macro:.5f}'
                    f'Precision by class: {precision}'
                    f'Recall by class: {recall}'
                    f'Accuracy by class: {accuracy}'
                    f'TP: {tp}'
                    f'FP: {fp}'
                    f'TN: {tn}'
                    f'FN: {fn}')
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
    trainer = Trainer(model=model, dataset=dataset, num_epochs=args.num_epochs, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, validation_dataset=validation_dataset,
                      validation_interval=args.validation_interval)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
