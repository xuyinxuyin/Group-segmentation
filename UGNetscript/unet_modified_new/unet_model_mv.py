""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *


class UNet_mv(nn.Module):
    def __init__(self, n_channels, n_classes,layer_mul,bilinear=True):
        super(UNet_mv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32*layer_mul)
        self.down1 = Down(32*layer_mul, 64*layer_mul)
        self.down2 = Down(64*layer_mul, 128*layer_mul)
        self.down3 = Down(128*layer_mul, 256*layer_mul)
        self.down4 = Down(256*layer_mul, 256*layer_mul)
        
        
        self.up1 = Up(512*layer_mul, 128*layer_mul, bilinear)
        self.up2 = Up(256*layer_mul, 64*layer_mul, bilinear)
        self.up3 = Up(128*layer_mul, 32*layer_mul, bilinear)
        self.up4 = Up(64*layer_mul, 32*layer_mul, bilinear)
        self.outc = OutConv(32*layer_mul, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
