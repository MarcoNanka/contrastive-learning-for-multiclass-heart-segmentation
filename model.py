import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, encoder_weights: tuple = None, encoder_biases: tuple = None):
        """
        U-Net model for semantic segmentation.
        """
        super(UNet, self).__init__()

        # Contracting path: Increasing features, reducing spatial dimensions
        self.encoder_conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        if encoder_weights is not None and encoder_biases is not None:
            encoder_layers = [self.encoder_conv1, self.encoder_conv2, self.encoder_conv3, self.encoder_conv4,
                              self.encoder_conv5]
            for layer, (weights, biases) in zip(encoder_layers, zip(encoder_weights, encoder_biases)):
                layer.weight = nn.Parameter(weights)
                layer.bias = nn.Parameter(biases)

        # Expanding path: Decreasing features, increasing spatial dimensions
        self.upconv1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.decoder_conv5 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=1)

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


class LocalEncoder(nn.Module):
    def __init__(self, encoder_weights: tuple = None, encoder_biases: tuple = None):
        super(LocalEncoder, self).__init__()
        self.encoder_conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        if encoder_weights is not None and encoder_biases is not None:
            encoder_layers = [self.encoder_conv1, self.encoder_conv2, self.encoder_conv3, self.encoder_conv4]
            for idx, (weights, biases) in enumerate(zip(encoder_weights, encoder_biases)):
                print(f"idx: {idx}")
                encoder_layers[idx].weight = nn.Parameter(weights)
                encoder_layers[idx].bias = nn.Parameter(biases)

    def forward(self, x):
        x1 = self.relu(self.encoder_conv1(x))  # 2, 16, 96, 96, 96
        x2 = self.pool(x1)  # 2, 16, 48, 48, 48
        x2 = self.relu(self.encoder_conv2(x2))  # 2, 32, 48, 48, 48
        x3 = self.pool(x2)  # 2, 32, 24, 24, 24
        x3 = self.relu(self.encoder_conv3(x3))  # 2, 64, 24, 24, 24
        x4 = self.pool(x3)  # 2, 64, 12, 12, 12
        x4 = self.relu(self.encoder_conv4(x4))  # 2, 128, 12, 12, 12
        x5 = self.pool(x4)  # 2, 128, 6, 6, 6
        x5 = self.relu(self.encoder_conv5(x5))  # 2, 256, 6, 6, 6
        return x5


class DomainEncoder(nn.Module):
    def __init__(self):
        super(DomainEncoder, self).__init__()
        self.encoder_conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.encoder_conv1(x))  # 2, 16, 96, 96, 96
        x2 = self.pool(x1)  # 2, 16, 48, 48, 48
        x2 = self.relu(self.encoder_conv2(x2))  # 2, 32, 48, 48, 48
        x3 = self.pool(x2)  # 2, 32, 24, 24, 24
        x3 = self.relu(self.encoder_conv3(x3))  # 2, 64, 24, 24, 24
        x4 = self.pool(x3)  # 2, 64, 12, 12, 12
        x4 = self.relu(self.encoder_conv4(x4))  # 2, 128, 12, 12, 12
        return x4
