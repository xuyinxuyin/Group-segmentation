""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import pdb

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)





        

def IndoGCir(n_of_group):
    inde=np.zeros([n_of_group,n_of_group])
    for i in range(n_of_group):
        for j in range(n_of_group):
            inde[i,j]=(i+j)%n_of_group
    return inde   


#def IndofGr12():
#    inde = np.zeros([12, 12])
#    for i in range(12):
#        for j in range(12):
#            inde[i, j] = (i + j) % 12
#    return inde


def IndofGr60():
    matt = sio.loadmat(os.path.join(os.path.dirname(__file__),
                                    'malater.mat'))
    inde = matt['multi'] - 1
    # local=np.array([0,1,2,3,4,13,24,10,9,6,16])
    # inde2=inde[:,local]
    return inde


#class Groupconv(nn.Module):
#    "group and 1by1 convolution"
#    def __init__(self, in_channels, out_channels, n_elements=12):
#        super().__init__()
#        self.n_elements=n_elements
#        if n_elements==12:
#            self.inde=IndofGr12()
#        else:
#            self.inde=IndofGr60()
#        
#        self.gc=nn.Conv3d(in_channels,
#                            out_channels,kernel_size=[n_elements,1,1])
#        
#        # here is 60x1x1 kernel
#        
#        
#    def forward(self,x):
#        x=torch.stack([x[:,:,self.inde[i,:],:] for i in range(self.n_elements)],axis=0)
#        # before x nxcx60x64 after 60[1]xnxcx60x4
#        x=x.permute(1,2,3,0,4)
#        # nxcx60x60[1]x49
#        x=self.gc(x)
#        #nxcx60x1x49
#        x=x.squeeze()
#        #nxcx60x49
#        return x


class Groupmulticonv(nn.Module):
    "group and 1by1 convolution"
    def __init__(self, in_channels, out_channels, solid_name,n_elements,gc_filter):
        super().__init__()
        self.n_elements=n_elements
        if solid_name=='cir':
            self.inde=IndoGCir(n_elements)
        else:
            self.inde=IndofGr60() 
        self.inde=self.inde.astype(int)
        if gc_filter==3:
            self.gc=nn.Conv3d(in_channels,out_channels,kernel_size=[self.n_elements,3,3],padding=[0,1,1])
        if gc_filter==1:
            self.gc=nn.Conv3d(in_channels,out_channels,kernel_size=[self.n_elements,1,1],padding=[0,0,0])
        #self.gc=nn.Conv3d(in_channels,out_channels,kernel_size=[self.n_elements,1,1],padding=[0,0,0])
        
        
        
    def forward(self,x):
        x=torch.cat([self.gc(x[:,:,self.inde[i,:],:,:]) for i in range(self.n_elements)],axis=2)
        return x
        ###check me, not sure to expand memory or do for loop



#class Groupmulticonv(nn.Module):
#    "group and 1by1 convolution"
#    def __init__(self, in_channels, out_channels, n_elements=12):
#        super().__init__()
#        self.n_elements=n_elements
#        if n_elements==12:
#            self.inde=IndofGr12()
#        else:
#            self.inde=IndofGr60() 
#        self.inde=self.inde.astype(int)
#        self.gc=nn.Conv3d(in_channels,out_channels,kernel_size=[self.n_elements,3,3],padding=[0,1,1])
#        
#        
#        
#    def forward(self,x):
#        x=torch.cat([self.gc(x[:,:,self.inde[i,:],:,:]) for i in range(self.n_elements)],axis=2)
#        
#        
#        #before x nxcx60x7x7  after nxcx60x8x8 after nxcx60x7x7
#        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class Cat_pool(nn.Module):
    def __init__(self,pool_way,n_element):
        super(Cat_pool,self).__init__()
        self.n_element=n_element
        self.pool_way=pool_way
    def forward(self,x1):
        x1=x1.view(-1,self.n_element,x1.shape[1],x1.shape[2],x1.shape[3])
        if self.pool_way=='max':
            x2=x1.max(dim=1, keepdim=True)[0]
        else:
            x2=x1.mean(dim=1,keepdim=True)
        x2=x2.expand(-1,self.n_element,-1,-1,-1)
        x2=torch.cat([x1,x2],2)          ##why 2 not 1?
        x2=x2.view(-1,x2.shape[2],x2.shape[3],x2.shape[4])
        return x2
        
        
    

