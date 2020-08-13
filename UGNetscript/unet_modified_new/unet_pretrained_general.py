#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:54:57 2020

@author: xuyin
"""


import torch.nn.functional as F
import torch.nn as nn
import torch
from .unet_parts import *
import segmentation_models_pytorch as smp
from .change_decoder import *  
import pdb


class Unet_mv(nn.Module):
      def __init__(self, n_classes,pr_model,pretrain):
          super(Unet_mv,self).__init__()
          if pretrain:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='imagenet',in_channels=4)   
          else:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='None',in_channels=4)   
          self.n_classes=n_classes
          self.encoder=model.encoder
          self.decoder=model.decoder
          self.segmentation_head=model.segmentation_head   ###final conv +softmax (not sure whether we should change center to conv)
      def forward(self,x):
          features=self.encoder(x)
          decoder_output=self.decoder(*features)
          masks=self.segmentation_head(decoder_output)
          return masks
          
      
    
class Unet_group(nn.Module):
      def __init__(self, n_classes,solid_name,n_element,gc_filter,gc_layer,pr_model,pretrain):
          super(Unet_group,self).__init__()
          self.n_classes=n_classes
          self.n_element=n_element
          if pretrain:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='imagenet',in_channels=4)  
          else:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='None',in_channels=4)
        
          self.encoder=model.encoder
          self.segmentation_head=model.segmentation_head
          gc_block=[]
          if pr_model=='mobilenet_v2':
             for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(1280,layer_n,solid_name,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,solid_name, n_element,gc_filter[i])
                 gc_block.append(layer)
             self.gc_model=nn.Sequential(*gc_block)
             if not gc_layer[-1]==1280:
                 change_channel_decoder(model=model.decoder, in_channels=gc_layer[-1]+96)
          elif pr_model=='resnet18' or pr_model=='resnet34':  
              for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(512,layer_n,solid_name,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,solid_name,n_element,gc_filter[i])
                 gc_block.append(layer)
              self.gc_model=nn.Sequential(*gc_block)
              if not gc_layer[-1]==512:
                  change_channel_decoder(model=model.decoder, in_channels=gc_layer[-1]+96)
          elif pr_model=='resnet50' or pr_model=='resnet101':
              for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(2048,layer_n,solid_name,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,solid_name,n_element,gc_filter[i])
                 gc_block.append(layer)
              self.gc_model=nn.Sequential(*gc_block)
              if not gc_layer[-1]==2048:
                 change_channel_decoder(model=model.decoder, in_channels=gc_layer[-1]+96)
          self.decoder=model.decoder
      def forward(self,x):
          features=self.encoder(x)
          features[5]= features[5].view(-1, self.n_element, features[5].shape[1], features[5].shape[2], features[5].shape[3]) 
           ## [b*nol*n_ele,512,14,14] to [b*nol,n_ele,512,14,14]
          features[5]= features[5].permute(0,2,1,3,4)  
          # [b*nol,512,n_ele,14,14]
          features[5]= self.gc_model(features[5])   
          
          #[b*nol,512,n_ele, 14,14]
          features[5]=features[5].permute(0,2,1,3,4)
          #[b*nol,n_ele,512,14,14]
          features[5]=features[5].contiguous().view(-1,features[5].shape[2],features[5].shape[3],features[5].shape[4]) 
          #[b*60*n_ele,512,14,14]
          decoder_output=self.decoder(*features)
          masks=self.segmentation_head(decoder_output)   
          return masks
                   


class Unet_pool(nn.Module):    ###n_element
      def __init__(self, n_classes,n_view, pool_way,pr_model='mobilenet_v2'):
          super(Unet_pool,self).__init__()
          if pretrain:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='imagenet',in_channels=4)  
          else:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='None',in_channels=4)
          self.n_classes=n_classes
          self.encoder=model.encoder
          if pr_model=='mobilenet_v2':
              #change_channel_decoder(model=model.decoder,in_channels=2656)
              change_channel_decoder(model=model.decoder, in_channels=2*1280+96)
          elif pr_model=='resnet18':
              #change_channel_decoder(model=model.decoder,in_channels=1280)  
              change_channel_decoder(model=model.decoder, in_channels=2*512+256)
          elif pr_model=='resnet34':
              change_channel_decoder(model=model.decoder, in_channels=2*512+256)
          elif pr_model=='resnet50':
              change_channel_decoder(model=model.decoder, in_channels=2*2048+1024)
          elif pr_model=='renet101':
              change_channel_decoder(model=model.decoder, in_channels=2*2048+1024)
              
          self.decoder=model.decoder
          self.pool_way=pool_way
          self.n_view=n_view
          self.segmentation_head=model.segmentation_head
          self.catpool=Cat_pool(self.pool_way,self.n_view)
      def forward(self,x):
          features=self.encoder(x)
          features[5]=self.catpool(features[5])
          decoder_output=self.decoder(*features)
          masks=self.segmentation_head(decoder_output)
          return masks
                   
      
class Unet_groupcat(nn.Module):
      def __init__(self, n_classes,solid_name,n_element,gc_filter,gc_layer,pr_model='mobilenet_v2'):
          super(Unet_groupcat,self).__init__()
          if pretrain:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='imagenet',in_channels=4)  
          else:
              model=smp.Unet(pr_model, classes=15, activation='softmax',encoder_weights='None',in_channels=4) 
          self.n_classes=n_classes
          self.encoder=model.encoder
          self.n_element=n_element
          gc_block=[]
          if pr_model=='mobilenet_v2':
              for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(1280,layer_n,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,n_element,gc_filter[i])
                 gc_block.append(layer)
              self.gc_model=nn.Sequential(*gc_block)
              change_channel_decoder(model=model.decoder, in_channels=1280+gc_layer[-1]+96)
          elif pr_model=='resnet18' or 'resnet34':
              for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(512,layer_n,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,n_element,gc_filter[i])
                 gc_block.append(layer)
              self.gc_model=nn.Sequential(*gc_block)
              change_channel_decoder(model=model.decoder, in_channels=512+gc_layer[-1]+96)
          elif pr_model=='resnet50' or 'resnet101':
              for (i, layer_n) in enumerate(gc_layer):
                 if i==0:
                     layer=Groupmulticonv(2048,layer_n,n_element,gc_filter[i])
                 else: 
                     layer=Groupmulticonv(gc_layer[i-1],layer_n,n_element,gc_filter[i])
                 gc_block.append(layer)
              self.gc_model=nn.Sequential(*gc_block)
              change_channel_decoder(model=model.decoder, in_channels=2048+gc_layer[-1]+96)
          self.decoder=model.decoder
          self.segmentation_head=model.segmentation_head
      def forward(self,x):
          features=self.encoder(x)
          x6=features[5].view(-1, self.n_element, features[5].shape[1], features[5].shape[2], features[5].shape[3]) 
          x6=x6.permute(0,2,1,3,4)  
          #[b,512,60,14,14]
          x6 = self.gc_model(x6)    
         #[b,512,60, 14,14]
          x6=x6.permute(0,2,1,3,4)
        #[b,60,512,14,14]
          x6=x6.contiguous().view(-1,x6.shape[2],x6.shape[3],x6.shape[4]) #not sure the order ###??? contiguous
        #[b*60,512,14,14]
        
          features[5]=torch.cat([features[5],x6],1)
          decoder_output=self.decoder(*features)
          masks=self.segmentation_head(decoder_output)
          return masks


