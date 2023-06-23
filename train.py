import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset
import time


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
                validation_loss, validation_accuracy = self.evaluate_validation()
                print(
                    f'Epoch {epoch + 1}/{self.num_epochs}, '
                    f'Loss: {loss.item():.5f}, Validation Loss: {validation_loss:.5f}, '
                    f'Validation Accuracy: {validation_accuracy:.5f}')

            else:
                print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.5f}')

        torch.save(self.model.state_dict(), 'trained_model.pth')

    def evaluate_validation(self):
        """
        Evaluate the model on the validation dataset.

        Returns:
            tuple: Validation loss and validation accuracy.
        """
        self.model.eval()

        val_dataloader = DataLoader(dataset=self.validation_dataset, batch_size=self.batch_size, shuffle=False)

        val_criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            i = 0
            for val_batch_x, val_batch_y in val_dataloader:
                val_batch_x = val_batch_x.to(device=self.device, dtype=torch.float)
                val_batch_y = val_batch_y.to(device=self.device, dtype=torch.long)
                val_outputs = self.model(val_batch_x)  # (batch_size, number_classes, width, height, depth)
                val_loss = val_criterion(input=val_outputs, target=val_batch_y)
                total_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, dim=1)  # predicted: (batch_size, width, height, depth)
                total_correct += torch.eq(predicted, val_batch_y).sum().item()
                if i == 0:
                    print(f"torch.eq(predicted, val_batch_y).sum().item(): "
                          f"{torch.eq(predicted, val_batch_y).sum().item()}")
                    print(f"{total_correct}, {total_correct / torch.numel(predicted)}")
                    print(f"val_outputs: {val_outputs[6, :,4:6, 4:6, 4], val_outputs.shape}")
                    print(f"predicted: {predicted[6, 4:6, 4:6, 4], predicted.shape}")
                i = i + 1

            average_loss = total_loss / len(val_dataloader)
            accuracy = total_correct / torch.numel(self.validation_dataset.y)
            print(f"total_correct: {total_correct}")

            return average_loss, accuracy


if __name__ == "__main__":
    folder_path = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/ct_train"
    val_folder_path = "/Users/marconanka/BioMedia/data/quarter reduced MM-WHS 2017 Dataset/val_ct_train"
    patch_size = (24, 24, 24)
    dataset = MMWHSDataset(folder_path=folder_path, patch_size=patch_size)
    validation_dataset = MMWHSDataset(folder_path=val_folder_path, patch_size=patch_size)
    num_epochs = 10
    batch_size = 3
    learning_rate = 0.001
    validation_interval = 5
    number_of_channels = dataset.x.shape[1]
    model = UNet(in_channels=number_of_channels, num_classes=dataset.num_classes)
    start_train = time.process_time()
    trainer = Trainer(model=model, dataset=dataset, num_epochs=num_epochs, batch_size=batch_size,
                      learning_rate=learning_rate, validation_dataset=validation_dataset,
                      validation_interval=validation_interval)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")
