import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import BasicCNN
from data_loading import MMWHSDataset

main_dir = "/Users/marconanka/BioMedia/data/reduced MM-WHS 2017 Dataset/"
subfolder = "ct_train"
dataset = MMWHSDataset(main_dir, subfolder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model
model = BasicCNN(num_classes=8)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs['out'], batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
