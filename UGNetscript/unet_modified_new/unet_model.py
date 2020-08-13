""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *
import pdb

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,gc_layer=[512,512],solid_name='ico',n_element=60,layer_mul=1,bilinear=True):
        ###maybe change the middle layers in gc_block, adding parameters like [512,512,512]
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gc_layer=gc_layer
        self.n_element=n_element
        
        
        self.inc = DoubleConv(n_channels, 32*layer_mul) ###original 64
        #224
        self.down1 = Down(32*layer_mul, 64*layer_mul)
        #112
        self.down2 = Down(64*layer_mul, 128*layer_mul)
        #56
        self.down3 = Down(128*layer_mul, 256*layer_mul)
        #28
        self.down4 = Down(256*layer_mul, 256*layer_mul)
        #14
        
        
        gc_block=[]
        for (i, layer_n) in enumerate(gc_layer):
            if i==0:
                layer=Groupmulticonv(256*layer_mul,layer_n*layer_mul,solid_name,n_element)
            else:
                layer=Groupmulticonv(gc_layer[i-1]*layer_mul,layer_n*layer_mul,solid_name,n_element)
            gc_block.append(layer)
        self.gc_block=nn.Sequential(*gc_block)
        
        self.up1 = Up(512*layer_mul, 128*layer_mul, bilinear)
        self.up2 = Up(256*layer_mul, 64*layer_mul, bilinear)
        self.up3 = Up(128*layer_mul, 32*layer_mul, bilinear)
        self.up4 = Up(64*layer_mul, 32*layer_mul, bilinear)
        self.outc = OutConv(32*layer_mul, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #[n*60,64,224,224]
        x2 = self.down1(x1)
        #[n*60,128,112,112]
        x3 = self.down2(x2)
        #[n*60,256,56,56]
        x4 = self.down3(x3)
        #[n*60,512,28,28]
        x5 = self.down4(x4)
        # [n*60,512,14,14]
        x5= x5.view(-1, self.n_element, x5.shape[1], x5.shape[2], x5.shape[3]) 
        ## not sure the order [n,60, 512, 14,14]
        x5=x5.permute(0,2,1,3,4)  
        
        # [n,512,60,14,14]
        x5 = self.gc_block(x5)    
        
        #[n,512,60, 14,14]
        x5=x5.permute(0,2,1,3,4)
        #[n,60,512,14,14]
        
        
        x5=x5.view(-1,x5.shape[2],x5.shape[3],x5.shape[4]) #not sure the order
        #[n*60,512,14,14]
        
        x = self.up1(x5, x4)
        #28
        #del x5, x4
        x = self.up2(x, x3)
        #56
        #del x3
        x = self.up3(x, x2)
        #112
        #del x2
        x = self.up4(x, x1)
        #224
        #del x1
        logits = self.outc(x)
        
        
        
        return logits
