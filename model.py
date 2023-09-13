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
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Expanding path: Decreasing features, increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

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
        x1 = self.relu(self.encoder_conv1(x))
        print(x1.shape)
        x2 = self.pool1(x1)
        print(x2.shape)
        x2 = self.relu(self.encoder_conv2(x2))
        print(x2.shape)

        # Expanding path: Refining features
        x3 = self.upconv1(x2)
        print(x3.shape)
        x3 = torch.cat((x1, x3), dim=1)
        print(x3.shape)
        x3 = self.relu(self.decoder_conv1(x3))
        print(x3.shape)
        output = self.decoder_conv2(x3)
        print(output.shape)

        return output
