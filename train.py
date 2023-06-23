import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset
import time


class Trainer:
    def __init__(self, model, dataset, num_epochs, batch_size=4, learning_rate=0.001):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def force_cudnn_initialization(self):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=device, dtype=torch.float)

        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device=device, dtype=torch.float)
                batch_y = batch_y.to(device=device, dtype=torch.long)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(input=outputs, target=batch_y)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

        torch.save(self.model.state_dict(), 'trained_model.pth')


if __name__ == "__main__":
    folder_path = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/ct_train"
    patch_size = (24, 24, 24)
    start_dataset = time.process_time()
    dataset = MMWHSDataset(folder_path=folder_path, patch_size=patch_size)
    print(f"time for dataset: {time.process_time() - start_dataset}")
    num_epochs = 10
    batch_size = 3
    learning_rate = 0.001
    print(f"image data: {dataset.x.shape}")
    print(f"labels: {dataset.y.shape}")
    start_model = time.process_time()
    model = UNet(in_channels=1, num_classes=8)
    print(f"time for model: {time.process_time() - start_model}")
    start_train = time.process_time()
    trainer = Trainer(model=model, dataset=dataset, num_epochs=num_epochs, batch_size=batch_size,
                      learning_rate=learning_rate)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")
