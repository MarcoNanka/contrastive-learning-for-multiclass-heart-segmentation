from torch import nn


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv = nn.Sequential(
            # in_channels: number of channels, for CT/MRI = 1, RGB = 3; out_channels: how many filters to apply
            # both independent of image dimensions (width, height, depth)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True)
        )
        # self.fc = nn.Linear(in_features=512 * 512 * 1, out_features=8)

    def forward(self, x):
        x = self.conv(x)
        # x = self.fc(x)
        return x
