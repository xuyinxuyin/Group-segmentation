#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:26:28 2020

@author: xuyin
"""

from __future__ import division
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
import pdb


class CirDataset(Dataset):
    def __init__(self,resolui,resoluj,angle,path,ypropx,nov,nol,crop):
        self.semantic_path=glob.glob('/Datasets/xuyin'+'/*/*/semantic/*.png')
        self.path_name=path
        if not crop:
            self.grids=np.load('project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'final_grid.npy')
        else:
            self.grids=np.load('crop_project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'final_grid.npy')

    def __len__(self):
        return len(self.semantic_path)

    def __getitem__(self, idx):
        semantic_img=self.semantic_path[idx]
        semantic_name=semantic_img.split('/')[-1]
        area_number=semantic_img.split('/')[3]
        coname=semantic_name[:-12]
        
#        rgb_img=self.path_name+'/'+area_number+'/pano/rgb/'+coname+'rgb.png'
#        depth_img=self.path_name+'/'+area_number+'/pano/depth/'+coname+'depth.png'
#        sem_pretty_img=self.path_name+'/'+area_number+'/pano/semantic_pretty/'+coname+'semantic_pretty.png'
        rgb_img='/Datasets/xuyin'+'/'+area_number+'/pano/rgb/'+coname+'rgb.png'
        depth_img='/Datasets/xuyin'+'/'+area_number+'/pano/depth/'+coname+'depth.png'
        sem_pretty_img='/Datasets/xuyin'+'/'+area_number+'/pano/semantic_pretty/'+coname+'semantic_pretty.png'
        semantic=cv2.imread(semantic_img)
        rgb=cv2.imread(rgb_img)
        depth=cv2.imread(depth_img)
        semantic_pretty=cv2.imread(sem_pretty_img)
        
    
        
#        image = io.imread(img_name)
#        landmarks = self.landmarks_frame.iloc[idx, 1:]
#        landmarks = np.array([landmarks])
#        landmarks = landmarks.astype('float').reshape(-1, 2)
#        sample = {'image': image, 'landmarks': landmarks}
#
#        if self.transform:
#            sample = self.transform(sample)

        return rgb, depth, semantic, semantic_pretty, coname,area_number
    
    
    
    
