from torch import nn
import torch


class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            # in_channels: number of channels, for CT/MRI = 1, RGB = 3; out_channels: how many filters to apply
            # both independent of image dimensions (width, height, depth)
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.features = self.features.to(torch.float32)
        self.classifier = nn.Linear(in_features=32 * 128 * 128 * 80, out_features=num_classes)

    def forward(self, x):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        x = self.features(x)
        # changing shape of x to (batch_size, rest)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        out = {'out': x}
        return out
