import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super(UNet, self).__init__()

        # Contracting path: Increasing features, reducing spatial dimensions
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Expanding path: Decreasing features, increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Contracting path
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(self.pool1(x1)))

        # Expanding path
        x3 = self.upconv1(x2)
        x3 = torch.cat((x1, x3), dim=1)
        x3 = self.relu(self.conv3(x3))
        output = self.conv4(x3)

        return output
