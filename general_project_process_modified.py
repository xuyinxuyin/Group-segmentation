#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:30:02 2020

@author: xuyin
"""

#from segprocess12 import CirDataset
from general_prepare_pano_modified import CirDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import cv2
import pdb
from pathlib import Path
import glob

parser = argparse.ArgumentParser(description='PyTorch process segmentation')
parser.add_argument('--resi',type=int, default=384, help='resolution of image') #format of homogeneous space
parser.add_argument('--resj',type=int, default=384, help='resolution of image') #format of homogeneous space
parser.add_argument('--crosswise', type=int, default=45, help='resolution of image') #format of homogeneous space
parser.add_argument('--path_name',type=str,default='/Datasets/xuyin/project',help='path to data')
parser.add_argument('--crop',action='store_true', help='whether we should erase black area')
parser.add_argument('--ypropx',type=float, default=1, help='propotion of y to x')
parser.add_argument('--nov',type=int,default=12, help='number of view')
parser.add_argument('--nol',type=int,default=2, help='number of layers')



args = parser.parse_args()

area_list=['area_1','area_2','area_3','area_4','area_5a','area_5b','area_6']

type_list=['rgb','depth','semantic', 'semantic_pretty']

sample_dict={}

resolui=args.resi
resoluj=args.resj
angle=args.crosswise
path_name=args.path_name
crop=args.crop
nov=args.nov
ypropx=args.ypropx
nol=args.nol

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for area_number in area_list:
    for im_type in type_list:
         Path(path_name+'/'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number+'/'+im_type).mkdir(parents=True, exist_ok=True)
    #Path(path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number+'/rgb').mkdir(parents=True, exist_ok=True)
    #Path(path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number+'/depth').mkdir(parents=True, exist_ok=True)
    #Path(path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number+'/semantic').mkdir(parents=True, exist_ok=True)
    #Path(path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number+'/semantic_pretty').mkdir(parents=True, exist_ok=True)
pdb.set_trace()
#pdb.set_trace()
sphere_dataset=CirDataset(resolui,resoluj,angle,path_name,ypropx,nov,nol,crop)
sample_grids=sphere_dataset.grids
sample_grids=torch.from_numpy(sample_grids).to(device).float()
process_loader=DataLoader(dataset=sphere_dataset,batch_size=10, shuffle=True,num_workers=0)


print(len(sphere_dataset))

for step_i, [rgb, depth, semantic, semantic_pretty, coname, area_number] in enumerate(process_loader):
    if not crop:
        rgb=rgb.to(device).permute(0,3,1,2).float()
        depth=depth.to(device).permute(0,3,1,2).float()
        semantic=semantic.to(device).permute(0,3,1,2).float()
        semantic_pretty=semantic_pretty.to(device).permute(0,3,1,2).float()
    else:
        rgb=rgb[:,270:1778,:,:].to(device).permute(0,3,1,2).float()
        depth=depth[:,270:1778,:,:].to(device).permute(0,3,1,2).float()
        semantic=semantic[:,270:1778,:,:].to(device).permute(0,3,1,2).float()
        semantic_pretty=semantic_pretty[:,270:1778,:,:].to(device).permute(0,3,1,2).float()
        
    basi=rgb.shape[0]
    
    for k in range(nov*nol):
        grid=sample_grids[k,:,:]
        multigrid=grid.repeat(basi,1,1,1)
        rgb_samples=F.grid_sample(rgb,multigrid,padding_mode='border')
        depth_samples=F.grid_sample(depth,multigrid,padding_mode='border')
        semantic_samples=F.grid_sample(semantic,multigrid,padding_mode='border',mode='nearest')
        pretty_semantic_samples=F.grid_sample(semantic_pretty,multigrid, padding_mode='border',mode='nearest')
        for t in range(basi):
            coname_t=coname[t]
            rgbb=rgb_samples[t,:,:,:].permute(1,2,0)
            depthh=depth_samples[t,:,:,:].permute(1,2,0)
            semanticc=semantic_samples[t,:,:,:].permute(1,2,0)
            semantic_prettyy=pretty_semantic_samples[t,:,:,:].permute(1,2,0)
            sample_dict={'rgb':rgbb, 'depth':depthh,'semantic':semanticc, 'semantic_pretty': semantic_prettyy}
            for im_type in type_list:
                general_dir_name=path_name+'/'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number[t]+'/'+im_type+'/'+coname_t+im_type
                Path(general_dir_name).mkdir(parents=True, exist_ok=True)
                general_name=general_dir_name+'/'+coname_t+im_type+'{:0>3}'.format(k)+'.png'
                cv2.imwrite(general_name,sample_dict[im_type].cpu().numpy())
            
#            rgb_dir_name=path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number[t]+'/rgb/'+coname_t+'rgb'
#            Path(rgb_dir_name).mkdir(parents=True, exist_ok=True)
#            rgb_name=rgb_dir_name+'/'+coname_t+'rgb'+'{:0>3}'.format(k)+'.png'
#            print(rgb_name)
#            
#            depth_dir_name=path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number[t]+'/depth/'+coname_t+'depth'
#            Path(depth_dir_name).mkdir(parents=True, exist_ok=True)
#            depth_name=depth_dir_name+'/'+coname_t+'depth'+'{:0>3}'.format(k)+'.png'
#            
#            
#            semantic_dir_name=path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number[t]+'/semantic/'+coname_t+'semantic'
#            Path(semantic_dir_name).mkdir(parents=True, exist_ok=True)
#            semantic_name=semantic_dir_name+'/'+coname_t+'semantic'+'{:0>3}'.format(k)+'.png'
#            
#            
#            semantic_pretty_dir_name=path_name+'/'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'/'+area_number[t]+'/semantic_pretty/'+coname_t+'semantic_pretty'
#            Path(semantic_pretty_dir_name).mkdir(parents=True, exist_ok=True)
#            semantic_pretty_name=semantic_pretty_dir_name+'/'+coname_t+'semantic_pretty'+'{:0>3}'.format(k)+'.png'
#            
#            cv2.imwrite(rgb_name,rgbb.cpu().numpy())
#            cv2.imwrite(depth_name,depthh.cpu().numpy())
#            cv2.imwrite(semantic_name,semanticc.cpu().numpy())
#            cv2.imwrite(semantic_pretty_name,semantic_prettyy.cpu().numpy())
            
            
