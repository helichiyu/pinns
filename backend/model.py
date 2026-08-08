import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(1, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.enc3 = DoubleConv(2 * b, 4 * b)
        self.enc4 = DoubleConv(4 * b, 8 * b)
        self.bottleneck = DoubleConv(8 * b, 8 * b)
        self.dec4 = DoubleConv(16 * b, 4 * b)
        self.dec3 = DoubleConv(8 * b, 2 * b)
        self.dec2 = DoubleConv(4 * b, b)
        self.dec1 = DoubleConv(2 * b, b)
        self.output = nn.Sequential(nn.Conv2d(b, 1, 1), nn.Sigmoid())

    @staticmethod
    def upsample(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        x = self.bottleneck(F.max_pool2d(enc4, 2))
        x = self.dec4(torch.cat((self.upsample(x, enc4), enc4), dim=1))
        x = self.dec3(torch.cat((self.upsample(x, enc3), enc3), dim=1))
        x = self.dec2(torch.cat((self.upsample(x, enc2), enc2), dim=1))
        x = self.dec1(torch.cat((self.upsample(x, enc1), enc1), dim=1))
        return self.output(x)
