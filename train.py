import torch
from torch import nn
from torch.utils.data import DataLoader
from model import UNet
from data_loading import MMWHSDataset
import time


class Trainer:
    def __init__(self, model, dataset, num_epochs, batch_size=4):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    # def force_cudnn_initialization(self):
    #     s = 32
    #     dev = torch.device('cuda')
    #     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    def train(self):
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Train the model
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to(device=device, dtype=torch.float32)
        # self.force_cudnn_initialization()

        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in dataloader:
                # batch_x = batch_x.to(device=device, dtype=torch.float32)
                # batch_y = batch_y.to(device=device, dtype=torch.float32)
                batch_x = batch_x.float()
                print(f"batch_x.shape = {batch_x.shape}")
                print(f"batch_y.shape = {batch_y.shape}")
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                print(f"outputs.shape = {outputs.shape}")
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

        # Save the trained model
        torch.save(self.model.state_dict(), 'trained_model.pth')


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/"
    subfolder = "ct_train"
    start_dataset = time.process_time()
    dataset = MMWHSDataset(main_dir, subfolder)
    print(f"time for dataset: {time.process_time() - start_dataset}")
    start_model = time.process_time()
    model = UNet(in_channels=1, num_classes=8)
    print(f"time for model: {time.process_time() - start_model}")
    start_train = time.process_time()
    trainer = Trainer(model=model, dataset=dataset, num_epochs=2, batch_size=4)
    trainer.train()
    print(f"time for training: {time.process_time() - start_train}")
