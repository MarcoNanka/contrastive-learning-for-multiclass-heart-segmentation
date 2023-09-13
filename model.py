import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        """
        U-Net model for semantic segmentation.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super(UNet, self).__init__()

        # Contracting path: Increasing features, reducing spatial dimensions
        self.encoder_conv1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Expanding path: Decreasing features, increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
       Forward pass of the U-Net model.

       Args:
           x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

       Returns:
           torch.Tensor: Output tensor of shape (batch_size, num_classes, depth, height, width).
       """
        # Contracting path: Finding High-level patterns
        x1 = self.relu(self.encoder_conv1(x))  # 16,16,24,24,24
        x2 = self.pool(x1)  # 16,16,12,12,12
        x2 = self.relu(self.encoder_conv2(x2))  # 16, 32, 12, 12, 12
        x3 = self.pool(x2)  # 16, 32, 6, 6, 6
        x3 = self.relu(self.encoder_conv3(x3))  # 16, 64, 6, 6, 6

        # Expanding path: Refining features
        x4 = self.upconv1(x3)  # 16, 32, 12, 12, 12
        x4 = torch.cat((x2, x4), dim=1)  # 16, 64, 12, 12, 12
        x4 = self.relu(self.decoder_conv1(x4))  # 16, 32, 12, 12, 12
        x5 = self.upconv2(x4)  # 16, 16, 24, 24, 24
        x5 = torch.cat((x1, x5), dim=1)  # 16, 32, 24, 24, 24
        x5 = self.relu(self.decoder_conv2(x5))  # 16, 16, 24, 24, 24
        output = self.relu(self.decoder_conv3(x5))  # 16, 8, 24, 24, 24

        return output
