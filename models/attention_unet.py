import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_branch import SpectralDecomposition, SpectralConvBlock, ChannelAttention
from models.transformer_block import TransformerBlock
from models.aspp import ASPP


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class CrossScaleAttention(nn.Module):

    def __init__(self, low_c, high_c):
        super().__init__()

        self.conv = nn.Conv2d(low_c + high_c, low_c, 1)

        self.att = nn.Sequential(
            nn.Conv2d(low_c, low_c, 1),
            nn.Sigmoid()
        )

    def forward(self, low, high):

        high_up = F.interpolate(
            high,
            size=low.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        fusion = torch.cat([low, high_up], dim=1)

        fusion = self.conv(fusion)

        attention = self.att(fusion)

        return low * attention


class AttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.e1 = conv_block(3, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)

        # Spectral Branch
        self.spectral = SpectralDecomposition()
        self.spectral_conv = SpectralConvBlock(4, 128)

        self.fusion_conv = conv_block(256, 128)

        self.channel_att = ChannelAttention(128)

        # Transformer
        self.transformer = TransformerBlock(512)

        # ASPP Multi-scale Context
        self.aspp = ASPP(512, 512)

        # Cross-scale Attention
        self.cross_scale = CrossScaleAttention(128, 512)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        self.mc_dropout = nn.Dropout2d(p=0.3)

        # Decoder
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.d4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = conv_block(128, 64)

        # Final Segmentation Layer
        self.final = nn.Conv2d(64, 1, 1)

        # GradCAM storage
        self.gradients = None
        self.activations = None


    def save_gradient(self, grad):
        self.gradients = grad


    def get_gradients(self):
        return self.gradients


    def get_activations(self):
        return self.activations


    def forward(self, x):

        # Encoder
        e1 = self.e1(x)

        # Spectral Branch
        spectral_feat = self.spectral(x)
        spectral_feat = spectral_feat.to(x.device)
        spectral_feat = self.spectral_conv(spectral_feat)

        e2 = self.e2(self.pool(e1))

        fusion = torch.cat([e2, spectral_feat], dim=1)

        fusion = self.fusion_conv(fusion)

        fusion = self.channel_att(fusion)

        e2 = e2 + fusion

        e3 = self.e3(self.pool(e2))

        e4 = self.e4(self.pool(e3))

        # Transformer
        e4 = self.transformer(e4)

        # ASPP
        e4 = self.aspp(e4)

        # Cross-scale attention
        e2 = self.cross_scale(e2, e4)


        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        b = self.mc_dropout(b)

        # Decoder
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.d4(d4)

        # Store activations for GradCAM from the first decoder block
        self.activations = d4

        if torch.is_grad_enabled():
            d4.register_hook(self.save_gradient)

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.d3(d3)

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.d2(d2)

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.d1(d1)

        seg = self.final(d1)

        return seg