import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super(UNet, self).__init__()

        # Contracting path
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Expanding path
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, num_classes, kernel_size=1)

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


# class BasicCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(BasicCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=3, stride=2),
#             nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool3d(kernel_size=3, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=169136, out_features=32),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=32, out_features=num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


