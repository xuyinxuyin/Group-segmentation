#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 01:43:35 2020

@author: xuyin
"""


import torch
import torch.nn as nn


def change_channel_decoder(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break
        
    origin_in_channels=module.in_channels   
    module.in_channels=in_channels
    # change input channels for first conv
    #module.in_channels = in_channels
    weight = module.weight.detach()

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        weight = torch.cat([weight,torch.zeros([module.out_channels,(module.in_channels-origin_in_channels)//module.groups,*module.kernel_size])],1)
   
    module.weight = nn.parameter.Parameter(weight)
   