#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:12:12 2020

@author: xuyin
"""

from __future__ import division
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
import pdb


class MultiDataset(Dataset):
    def __init__(self, solid_format,angle,path,resolui,resoluj,ypropx,nov,nol,test_mode):
        if test_mode=='train':
            train_area=['area_1','area_2','area_3','area_4','area_6']
        else:
            train_area=['area_5a','area_5b']
           
        self.solid=solid_format
        self.angle=angle
        self.path=path
        self.resolui=resolui
        self.resoluj=resoluj
        self.ypropx=ypropx
        self.nov=nov
        self.nol=nol
        #self.n_element=n_element
        
        if solid_format=='cir':
            path_name=path+'/'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)
            self.n_element=nol*nov
        else:
            path_name=path+'/'+solid_format+'_'+str(resolui)+'_'+str(angle)+'_pano'
            self.n_element=60
    
#        if n_element==60:
#            path_name=path+'/'+solid_format+'_'+str(resolu)+'_'+str(angle)+'_pano'
#        elif n_element==12:
#            path_name=path+'/'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_pano'
#        
        
        
        self.dir_list=[]
        for area_number in train_area:
            self.dir_list+=glob.glob(path_name+'/'+area_number+'/semantic/*/')
            
    
        self.path_name=path
        self.labelmap = np.loadtxt("label_maps.txt", dtype='int32')
        self.test_mode=test_mode
        
    
        
    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        semantic_img=self.dir_list[idx]
        semantic_img=semantic_img[:-1]
        semantic_path=semantic_img.split('/')
        
        semantic_name=semantic_path[-1]
        
        semantic_path.remove(semantic_name)
        
        
        last_item=semantic_path[-1]
        semantic_path.remove(last_item)
        
        area_number=semantic_path[-1]
        
        copath='/'.join(semantic_path)
        
        coname=semantic_name[:-8]
        
        rgb_path=copath+'/rgb/'+coname+'rgb'
        depth_path=copath+'/depth/'+coname+'depth'
        #semantic_pretty_path=copath+'/semantic_pretty/'+coname+'semantic_pretty'
        semantic_sph_path='/Datasets/xuyin/'+area_number+'/pano/semantic/'+coname+'semantic.png'
        
        
        
        rgb_coname=rgb_path.split('/')[-1]
        depth_coname=depth_path.split('/')[-1]
        semantic_coname=semantic_img.split('/')[-1]
        
    
        
        rgb_list=[]
        for k in range(self.n_element):
            rgb=cv2.imread(rgb_path+'/'+rgb_coname+'{:0>3}.png'.format(k))
            rgb=rgb/255                                                    ###not sure, need to check other code
            rgb_list.append(rgb)
        rgb=np.stack(rgb_list,axis=0)
        
        
        
        
        depth_list=[]
        for k in range(self.n_element):
            depth=cv2.imread(depth_path+'/'+depth_coname+'{:0>3}.png'.format(k), cv2.IMREAD_GRAYSCALE)/512.####not sure, need to check other code
            depth=np.clip(depth,0,5)                       
            depth=np.expand_dims(depth, axis=2)
            depth_list.append(depth) 
        depth=np.stack(depth_list,axis=0)
        
        
        semantic_list=[]
        for k in range(self.n_element):
            semantic=cv2.imread(semantic_img+'/'+semantic_coname+'{:0>3}.png'.format(k))
            idx=semantic[...,1]*256+semantic[...,0]
            rgb_channel=semantic[...,2]
            semantic=self.labelmap[idx]
            semantic[np.nonzero(rgb_channel)]=14
            semantic=np.expand_dims(semantic,axis=2)   ####need to check
            
            semantic_list.append(semantic)
        semantic=np.stack(semantic_list,axis=0)

        ## to do list, transfer semantic_label to semantic_pretty
        
        if self.test_mode=='train':
            return rgb, depth, semantic
        else:
            semantic_pano=cv2.imread(semantic_sph_path)
            idx=semantic_pano[...,1]*256+semantic_pano[...,0]
            rgb_channel=semantic_pano[...,2]
            semantic_pano=self.labelmap[idx]
            semantic_pano[np.nonzero(rgb_channel)]=14
            semantic_pano=np.expand_dims(semantic_pano,axis=2)
            ##need more path
            return rgb, depth, semantic, semantic_pano
        
        
        
        
