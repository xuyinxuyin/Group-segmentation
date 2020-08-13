#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:58:35 2020

@author: xuyin
"""

from yacs.config import CfgNode as CN

_C = CN()
##################################################
###experiments settings
_C.EPOCHS = 200
_C.BATCH_SIZE = 1
_C.LR = 0.01
_C.PRETRAIN = True   ####pretrained
_C.EVAL_EPOCH = True ### need evaluation for every epoch
_C.SAVE_CP = True   ### save checkpoint
_C.SKIP_TRAIN = False ### whether skip training process
_C.DECAY = True    ### learning rate decay

######process of dataset
_C.SOLID = 'cir' ### format : cir, ico, ded, tru
_C.CROSSWISE = 45   ###0.5 angle of x axis
_C.YPROPX = 1
_C.NOV = 12     ###number of views per layer
_C.NOL = 2      ###number of layers
_C.RESI = 384
_C.RESJ = 384
_C.PATH = '/Datasets/xuyin/project'

#######model setting
_C.GC_LAYER = [512]
_C.CHOOSE_MODEL = 'group'    #['group','mv','pool', 'groupcat']
_C.LAYER_MULTIPLY = 1        ### when use custom network, we might use layer_multiply
_C.N_ELEMENT = 12    ### number of elements in group
_C.POOL_FORMAT = 'max'  ### pooling format
_C.PR_MODEL = 'resnet18'
_C.GC_FILTER_SIZE= [1]









 
    #parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        #help='Load model from a .pth file')
    #parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                       # help='Downscaling factor of the images')
    #parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                       # help='Percent of the data that is used as validation (0-100)')
    