#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:55:24 2020

@author: xuyin
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision as vision
import numpy as np

import math
import pdb



classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
class_names = ["unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column", 
               "door", "floor", "sofa", "table", "wall", "window", "invalid"]
drop = [0, 14]
keep = np.setdiff1d(classes, drop)
label_ratio = [0.04233976974675504, 0.014504436907968913, 0.017173225930738712, 
               0.048004778186652164, 0.17384037404789865, 0.028626771620973622, 
               0.087541966989014, 0.019508096683310605, 0.08321331842901526, 
               0.17002664771895903, 0.002515611224467519, 0.020731298851232174, 
               0.2625963729249342, 0.016994731594287146, 0.012382599143792165]
# label_weight = 1/np.array(label_ratio)/np.sum((1/np.array(label_ratio))[keep])
label_weight = 1 / np.log(1.02 + np.array(label_ratio))
label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)
#x=np.linspace(0, 4095, 4096)
#y=np.linspace(0, 2047, 2048)
#
#sph_grid=torch.unsqueeze(torch.tensor(np.meshgrid(x,y)).cuda(),0)
#

def eval_net(net, loader, device, writer):
    """Evaluation without the densecrf with the dice coefficient"""
    w = torch.tensor(label_weight).to(device)
    net.eval()
    tot = 0
    
    ints_=np.zeros(len(classes)-len(drop))
    unis_=np.zeros(len(classes)-len(drop))
    per_cls_counts=np.zeros(len(classes)-len(drop))
    accs= np.zeros(len(classes)-len(drop))
    
    ints2_=np.zeros(len(classes)-len(drop))
    unis2_=np.zeros(len(classes)-len(drop))
    per_cls_counts2=np.zeros(len(classes)-len(drop))
    accs2= np.zeros(len(classes)-len(drop))
    
#    ints2_ = np.zeros(len(classes)-len(drop))
#    unis2_ = np.zeros(len(classes)-len(drop))
#    per_cls_counts2 = np.zeros(len(classes)-len(drop))
#    accs2= np.zeros(len(classes)-len(drop))
#    
    
    count = 0
    
    
    solid_format=loader.dataset.solid
    resolui=loader.dataset.resolui
    resoluj=loader.dataset.resoluj
    angle=loader.dataset.angle
    ypropx=loader.dataset.ypropx
    nol=loader.dataset.nol
    nov=loader.dataset.nov
    n_val=len(loader.dataset)
    batch_size=loader.batch_size
    
    if solid_format=='cir':
                 uv_grid=np.load('/home/xuyin/Group-segmentation/project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'uv_grid.npy')
                 #mask_pred=mask_pred.view(-1,nov*nol,mask_pred.shape[1],mask_pred.shape[2])
                 uv_grid=torch.from_numpy(uv_grid).to(device)
    else:
                 uv_grid=np.load('/home/xuyin/Group-segmentation/project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'uv_grid.npy')
                 #mask_pred=mask_pred.view(-1,nov*nol,mask_pred.shape[1],mask_pred.shape[2]) 
                 uv_grid=torch.from_numpy(uv_grid).to(device)                                                                                #### need to change
    uv_grid=uv_grid.repeat(batch_size,1,1,1,1) ###[n,1,2048,4096,3]
    
    
    pretty_label=np.load('pretty_label.npy')
    pretty_label=torch.from_numpy(pretty_label).to(device)
    #pretty_label=torch.from_numpy(pretty_label).to(device) ###not sure whether need to change cuda
    
    #record_grid_idx=torch.zeros(60,2,resolu,resolu)
    #grids=torch.tensor(np.load('../'+solid_format+'_'+str(resolu)+'_'+str(angle)+'final_grid.npy'),device=device)
    
#    for i in range(60):
#        record_grid_idx[i,:,:,:]=F.grid_sample(sph_grid,torch.unsqueeze(grids[i,:,:,:],0),padding_mode='border',mode='nearest')
#        
#    record_grid_idx=record_grid_idx.permute(0,2,3,1)   
#    
    record_steps=math.floor(len(loader)/3)
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
    #with tqdm(total=n_val, desc='Validation round', unit='img') as pbar:   
        for step_i, [input_im,true_masks,semantic2] in enumerate(loader):
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            s=input_im.shape[0]
            input_im=input_im.to(device,dtype=torch.float32).permute(0,1,4,2,3)
            
            
            #depth=depth.to(device,dtype=torch.float32).permute(0,1,4,2,3)
            true_masks=true_masks.to(device,dtype=mask_type).permute(0,1,4,2,3)
            #input_img=torch.cat([rgb,depth],axis=2)
            input_im=input_im.contiguous().view(-1,input_im.shape[2],input_im.shape[3],input_im.shape[4])
            ###[basi*n_views(2*12),3,224,224]
            true_masks=true_masks.contiguous().view(-1,true_masks.shape[2],true_masks.shape[3],true_masks.shape[4]).squeeze()
            #[ba_si*60,224,224]
            
          
            
            true_sphere_masks=torch.squeeze(semantic2.to(device,dtype=mask_type),-1)
        
#            imgs = imgs.to(device=device, dtype=torch.float32)
#            mask_type = torch.float32 if net.n_classes == 1 else torch.long
#            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            
            
            with torch.no_grad():
                output= net(input_im)
            
           
            tot += F.cross_entropy(output, true_masks,weight=w)
            mask_pred= output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability;
            #ouput: (ba_si*60,15,224,224)  mask_pred: (ba_si*60,224,224)
            
            
            #for sphere grid:
            
            
            int_, uni_ = iou_score(mask_pred, true_masks)
            tpos, pcc = accuracy(mask_pred, true_masks)
            ints_ += int_
            unis_ += uni_
            accs += tpos
            per_cls_counts += pcc
            
            mask_pred=mask_pred.view(-1,1,nov*nol,mask_pred.shape[1],mask_pred.shape[2]).double() #[b,c(1),nov,h,w]
            sphere_mask=F.grid_sample(mask_pred, uv_grid,mode='nearest',align_corners=True)
            sphere_mask=torch.squeeze(torch.squeeze(sphere_mask,1),1) ###rduce channel, from [b,c(1),1,2048,4096] to [b.2048,4096]
            mask_pred=mask_pred.reshape(-1,mask_pred.shape[3],mask_pred.shape[4]) #[b*nov,h,w]
            
            
            
            
            
            
            
            int2_, uni2_ = iou_score(sphere_mask, true_sphere_masks)
            tpos2, pcc2 = accuracy(sphere_mask, true_sphere_masks)
            ints2_ += int2_
            unis2_ += uni2_
            accs2 += tpos2
            per_cls_counts2 += pcc2



            
            count += s
            
            
            
            mask_pred=torch.stack([pretty_label[:,0][mask_pred.long()],pretty_label[:,1][mask_pred.long()],pretty_label[:,2][mask_pred.long()]],1)
            true_masks=torch.stack([pretty_label[:,0][true_masks.long()],pretty_label[:,1][true_masks.long()],pretty_label[:,2][true_masks.long()]],1)
            
            sphere_mask=torch.stack([pretty_label[:,0][sphere_mask.long()],pretty_label[:,1][sphere_mask.long()],pretty_label[:,2][sphere_mask.long()]],1)
            true_sphere_masks=torch.stack([pretty_label[:,0][true_sphere_masks.long()],pretty_label[:,1][true_sphere_masks.long()],pretty_label[:,2][true_sphere_masks.long()]],1)
            
            
        
        
            
            if step_i% record_steps==0:
            #if step_i == 0:
                writer.add_image('Tangent_images/test',
                                 vision.utils.make_grid(torch.cat([mask_pred,true_masks],0),4))
                ###[2*b_size*n_views(2*12),1,224,224]
                writer.add_image('Sphere_image/test',
                                 vision.utils.make_grid(torch.cat([sphere_mask,true_sphere_masks],0),2)
                                 )
                ###[2*b_size,1,224,224]
           
           
            del mask_pred,true_sphere_masks
            pbar.update(s)
    
    ious = ints_ / unis_
    accs /= per_cls_counts
    tot /= count
    
    ious2 = ints2_ / unis2_
    accs2 /= per_cls_counts2
    
    

    return ious,accs,tot, ious2, accs2


def iou_score(pred_cls, true_cls, nclass=15, drop=drop):
    
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i).int() + (true_cls == i).int()).eq(2).sum().item()
            union = ((pred_cls == i).int() + (true_cls == i).int()).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return np.array(intersect_), np.array(union_)


def accuracy(pred_cls, true_cls, nclass=15, drop=drop):
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i).int() + (true_cls == i).int()).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])
    return np.array(tpos), np.array(per_cls_counts)






             
                
                
            
            
            
            
            
            
            
            
        
    
    


