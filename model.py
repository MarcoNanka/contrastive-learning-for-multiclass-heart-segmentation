import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.calculate_num_features(), out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        self.input_size = 0

    def calculate_num_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size[2:])
            features_output = self.features(dummy_input)
        return features_output.view(dummy_input.size(0), -1).size(1)

    def forward(self, x):
        self.input_size = x.size()
        x = self.features(x)
        x = x.view(self.input_size[0], -1)
        x = self.classifier(x)
        out = {'out': x}
        return out
