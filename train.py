import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import BasicCNN
from data_loading import MMWHSDataset


def train(model, num_epochs):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device, dtype=torch.float32)

    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs['out'], batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')


main_dir = "/gdrive/MyDrive/bachelor's thesis code/data/reduced MM-WHS 2017 Dataset/"
subfolder = "ct_train"
dataset = MMWHSDataset(main_dir, subfolder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = BasicCNN(num_classes=8)
train(model=model, num_epochs=2)
