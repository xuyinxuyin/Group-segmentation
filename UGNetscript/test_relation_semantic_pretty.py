#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:06:15 2020

@author: xuyin
"""

import cv2
import glob
import numpy as np
import torch

import pdb



semantic_path='/Datasets/xuyin/area_1/pano/semantic/*'
semantic_pretty_path='/Datasets/xuyin/area_1/pano/semantic_pretty/'


labelmap = np.loadtxt("label_maps.txt", dtype='int32')
dir_list1=glob.glob(semantic_path)

pretty_label=np.load('pretty_label.npy')

for se_name in dir_list1:
    semantic=cv2.imread(se_name)
    idx=semantic[...,1]*256+semantic[...,0]
    rgb_channel=semantic[...,2]
    semantic=labelmap[idx]
    semantic[np.nonzero(rgb_channel)]=14
    
    
    coname=se_name.split('/')[-1]
    coname=coname[:-4]
    pretty_name=coname+'_pretty.png'
    pretty_path=semantic_pretty_path+pretty_name
    semantic_pretty=cv2.imread(pretty_path)
    
    
    semantic_pretty_test=np.stack([pretty_label[:,0][semantic],pretty_label[:,1][semantic],pretty_label[:,2][semantic]],2)
    cv2.imwrite('pretty_true.png', semantic_pretty)
    cv2.imwrite('pretty_test.png',semantic_pretty_test)
    break




#pretty_index=np.zeros([15,3])
#for se_name in dir_list1:
#    semantic=cv2.imread(se_name)
#    idx=semantic[...,1]*256+semantic[...,0]
#    rgb_channel=semantic[...,2]
#    semantic=labelmap[idx]
#    semantic[np.nonzero(rgb_channel)]=14
#
#    if len(np.unique(semantic))==15:
#        coname=se_name.split('/')[-1]
#        coname=coname[:-4]
#        pretty_name=coname+'_pretty.png'
#        pretty_path=semantic_pretty_path+pretty_name
#        semantic_pretty=cv2.imread(pretty_path)
#        semantic_pretty_test=semantic_pretty.copy()
#        
#        for i in range(15):
#            ind_=np.nonzero(semantic==i)
#            ind_first=(ind_[0][0],ind_[1][0])
#            #pdb.set_trace()
#            #ind_first=np.unravel_index(ind2_[0],(2048,4096))
#            #print(ind_first)
#            color_1=semantic_pretty[...,0][ind_first]
#            color_2=semantic_pretty[...,1][ind_first]
#            color_3=semantic_pretty[...,2][ind_first]
#            
#            
#            
#            
#            pretty_index[i,0]=color_1
#            pretty_index[i,1]=color_2
#            pretty_index[i,2]=color_3
#            
#            semantic_pretty_test[...,0][ind_]=color_1
#            semantic_pretty_test[...,1][ind_]=color_2
#            semantic_pretty_test[...,2][ind_]=color_3
#        print(pretty_index)
#        pdb.set_trace()
#        
#        cv2.imwrite('pretty_true.png', semantic_pretty)
#        cv2.imwrite('pretty_test.png',semantic_pretty_test)
#        np.save('pretty_label.npy', pretty_index)
#        break
#




