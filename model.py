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

        # Contracting path
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv7 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        # Expanding path
        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv9 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv13 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.conv15 = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, depth, height, width).
        """
        # Contracting path
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        x2 = self.pool1(x1)

        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x3 = self.pool2(x2)

        x3 = self.relu(self.conv5(x3))
        x3 = self.relu(self.conv6(x3))
        x4 = self.pool3(x3)

        # Bottleneck
        x4 = self.relu(self.conv7(x4))
        x4 = self.relu(self.conv8(x4))

        # Expanding path
        x = self.upconv1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))

        x = self.upconv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))

        x = self.conv15(x)

        return x
