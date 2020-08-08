#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:31:10 2020

@author: xuyin
"""

from __future__ import division
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
import pdb
import torch
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rgb_path='/home/xuyin/UGNet_local/project/*/area_1/rgb/*/*'
dir_list=glob.glob(rgb_path)
rgb_coname=dir_list[0][:-7]
split_path_dir=dir_list[0].split('/')
coname=split_path_dir[-1]
coname=coname[:-7]
semantic_sph_path='/Datasets/xuyin/area_1/pano/rgb/'+coname+'.png'
rgb_list=[]
for i in range(24):
     rgb=cv2.imread(rgb_coname+'{:0>3}.png'.format(i))
     #rgb=cv2.imread(rgb_coname+'000.png')                                          ###not sure, need to check other code
     rgb_list.append(rgb)
rgb=np.stack(rgb_list,axis=0)
rgb=torch.from_numpy(rgb).to(device,dtype=torch.float32)
rgb_sphere_tru=cv2.imread(semantic_sph_path)

cv2.imwrite('test_inverse_grid_true.png',rgb_sphere_tru)

uv_grid=np.load('/home/xuyin/UGNet_local/project_2_12_384_384_45_1uv_grid.npy')

uv_grid=torch.from_numpy(uv_grid).to(device)
pdb.set_trace()
uv_grid=uv_grid.repeat(1,1,1,1,1)
rgb=rgb.permute(3,0,1,2)
#rgb=rgb.permute(3,0,1,2)
rgb=rgb.repeat(1,1,1,1,1).double()
rgb_sphere=F.grid_sample(rgb, uv_grid,mode='nearest',align_corners=True)

rgb_sphere=rgb_sphere.squeeze()
rgb_sphere=rgb_sphere.permute(1,2,0)

cv2.imwrite('test_inverse_grid.png',rgb_sphere.cpu().numpy())

#rgb_sphere_tru=torch.from_numpy(rgb_sphere_tru).to(device)

        
    
