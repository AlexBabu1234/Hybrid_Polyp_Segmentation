import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, ff_dim=512):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, channels)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten spatial
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Self Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.dropout(attn_out)

        # Feed Forward
        x_norm = self.norm2(x_flat)
        ff_out = self.ff(x_norm)
        x_flat = x_flat + self.dropout(ff_out)

        # Reshape back
        x_out = x_flat.permute(0, 2, 1).view(B, C, H, W)

        return x_out