import torch
import torch.nn as nn


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels=256):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        size = x.shape[2:]

        feat1 = self.conv1(x)
        feat6 = self.conv6(x)
        feat12 = self.conv12(x)
        feat18 = self.conv18(x)

        pool = self.pool(x)
        pool = nn.functional.interpolate(pool, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([feat1, feat6, feat12, feat18, pool], dim=1)

        return self.final(x)