import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=8, encoder_weights: tuple = None, encoder_biases: tuple = None):
        """
        U-Net model for semantic segmentation.
        """
        super(UNet, self).__init__()

        # Contracting path: Increasing features, reducing spatial dimensions
        self.encoder_conv1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        if encoder_weights is not None and encoder_biases is not None:
            encoder_conv1_weights, encoder_conv2_weights, encoder_conv3_weights, encoder_conv4_weights, \
                encoder_conv5_weights = encoder_weights
            encoder_conv1_biases, encoder_conv2_biases, encoder_conv3_biases, encoder_conv4_biases, \
                encoder_conv5_biases = encoder_biases

            self.encoder_conv1.weight = nn.Parameter(encoder_conv1_weights)
            self.encoder_conv1.bias = nn.Parameter(encoder_conv1_biases)
            self.encoder_conv2.weight = nn.Parameter(encoder_conv2_weights)
            self.encoder_conv2.bias = nn.Parameter(encoder_conv2_biases)
            self.encoder_conv3.weight = nn.Parameter(encoder_conv3_weights)
            self.encoder_conv3.bias = nn.Parameter(encoder_conv3_biases)
            self.encoder_conv4.weight = nn.Parameter(encoder_conv4_weights)
            self.encoder_conv4.bias = nn.Parameter(encoder_conv4_biases)
            self.encoder_conv5.weight = nn.Parameter(encoder_conv5_weights)
            self.encoder_conv5.bias = nn.Parameter(encoder_conv5_biases)

        # Expanding path: Decreasing features, increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.decoder_conv5 = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
       Forward pass of the U-Net model. Comments are based on example values: 96^3 patch size, 16 batch size

       Args:
           x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

       Returns:
           torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width, depth).
       """
        # Contracting path: Finding High-level patterns
        x1 = self.relu(self.encoder_conv1(x))  # 16, 16, 96, 96, 96
        x2 = self.pool(x1)  # 16, 16, 48, 48, 48
        x2 = self.relu(self.encoder_conv2(x2))  # 16, 32, 48, 48, 48
        x3 = self.pool(x2)  # 16, 32, 24, 24, 24
        x3 = self.relu(self.encoder_conv3(x3))  # 16, 64, 24, 24, 24
        x4 = self.pool(x3)  # 16, 64, 12, 12, 12
        x4 = self.relu(self.encoder_conv4(x4))  # 16, 128, 12, 12, 12
        x5 = self.pool(x4)  # 16, 128, 6, 6, 6
        x5 = self.relu(self.encoder_conv5(x5))  # 16, 256, 6, 6, 6

        # Expanding path: Refining features
        x6 = self.upconv1(x5)  # 16, 128, 12, 12, 12
        x6 = torch.cat((x4, x6), dim=1)  # 16, 256, 12, 12, 12
        x6 = self.relu(self.decoder_conv1(x6))  # 16, 128, 12, 12, 12
        x7 = self.upconv2(x6)  # 16, 64, 24, 24, 24
        x7 = torch.cat((x3, x7), dim=1)  # 16, 128, 24, 24, 24
        x7 = self.relu(self.decoder_conv2(x7))  # 16, 64, 24, 24, 24
        x8 = self.upconv3(x7)  # 16, 32, 48, 48, 48
        x8 = torch.cat((x2, x8), dim=1)  # 16, 64, 48, 48, 48
        x8 = self.relu(self.decoder_conv3(x8))  # 16, 32, 48, 48, 48
        x9 = self.upconv4(x8)  # 16, 16, 96, 96, 96
        x9 = torch.cat((x1, x9), dim=1)  # 16, 32, 96, 96, 96
        x9 = self.relu(self.decoder_conv4(x9))  # 16, 16, 96, 96, 96
        output = self.relu(self.decoder_conv5(x9))  # 16, 8, 96, 96, 96

        return output


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        self.encoder_conv1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.encoder_conv1(x))
        x2 = self.pool(x1)
        x2 = self.relu(self.encoder_conv2(x2))
        x3 = self.pool(x2)
        x3 = self.relu(self.encoder_conv3(x3))
        x4 = self.pool(x3)
        x4 = self.relu(self.encoder_conv4(x4))
        x5 = self.pool(x4)
        x5 = self.relu(self.encoder_conv5(x5))
        return x5
