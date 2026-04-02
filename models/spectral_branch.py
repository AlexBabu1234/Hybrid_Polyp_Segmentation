import torch
import torch.nn as nn
import pywt
import numpy as np


class SpectralDecomposition(nn.Module):
    def __init__(self):
        super(SpectralDecomposition, self).__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Convert entire batch at once (faster than per-item loop)
        x_cpu = x[:, 0, :, :].detach().cpu().numpy()  # use first channel

        spectral_list = []

        for i in range(B):
            LL, (LH, HL, HH) = pywt.dwt2(x_cpu[i], 'haar')
            stacked = np.stack([LL, LH, HL, HH], axis=0)
            spectral_list.append(stacked)

        spectral_array = np.stack(spectral_list, axis=0)
        spectral_tensor = torch.from_numpy(spectral_array).float().to(x.device)

        return spectral_tensor
    
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y
    
class SpectralConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)