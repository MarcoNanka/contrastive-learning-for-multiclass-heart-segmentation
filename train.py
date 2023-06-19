import torch
from torch import nn
from torch.utils.data import DataLoader
from model import BasicCNN
from data_loading import MMWHSDataset
from sys import stdout


class Trainer:
    def __init__(self, model, dataset, num_epochs):
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs

    def force_cudnn_initialization(self):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    def train(self):
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Train the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=device, dtype=torch.float32)
        self.force_cudnn_initialization()

        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device=device, dtype=torch.float32)
                batch_y = batch_y.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs['out'], batch_y)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')
            stdout.write(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

        # Save the trained model
        torch.save(self.model.state_dict(), 'trained_model.pth')


if __name__ == "__main__":
    main_dir = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/"
    subfolder = "ct_train"
    dataset = MMWHSDataset(main_dir, subfolder)
    model = BasicCNN(num_classes=8)
    trainer = Trainer(model=model, dataset=dataset, num_epochs=2)
    trainer.train()
