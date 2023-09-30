import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        x1 = nn.functional.normalize(x1, dim=-1, p=2)
        x2 = nn.functional.normalize(x2, dim=-1, p=2)
        similarity_matrix = torch.matmul(x1, x2.T) / self.temperature
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss


class PreTrainer:
    def __init__(self, encoder, contrastive_dataset, num_epochs, batch_size, learning_rate, patch_size):
        """
        Trainer class for training a model.

        Args:
            encoder (torch.nn.Module)
            contrastive_dataset (torch.utils.data.Dataset)
            num_epochs (int)
            batch_size (int)
            learning_rate (float)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        self.contrastive_dataset = contrastive_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patch_size = patch_size

    def pre_train(self):
        contrastive_dataloader = DataLoader(self.contrastive_dataset, batch_size=32, shuffle=True)
        contrastive_loss = ContrastiveLoss()
        optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)

        num_epochs = 10
        for epoch in range(num_epochs):
            for batch in contrastive_dataloader:
                # Split the batch into two augmented views and apply the model
                view1, view2 = batch
                output1 = self.encoder(view1)
                output2 = self.encoder(view2)

                loss = contrastive_loss(output1, output2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        encoder_weights = (self.encoder.encoder_conv1.weight.data, self.encoder.encoder_conv2.weight.data,
                           self.encoder.encoder_conv3.weight.data)
        encoder_biases = (self.encoder.encoder_conv1.bias.data, self.encoder.encoder_conv2.bias.data,
                          self.encoder.encoder_conv3.bias.data)

        return encoder_weights, encoder_biases
