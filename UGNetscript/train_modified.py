#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:34:03 2020

@author: xuyin
"""

import argparse
import logging


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval_modified import eval_net
from unet_modified import UNet
from unet_modified import UNet_mv
from unet_modified import Pool_UNet_mv
from unet_modified import Catgroup_UNet
from unet_modified import Unet_mv
from unet_modified import Unet_group
from unet_modified import Unet_pool
from unet_modified import Unet_groupcat



from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from util import AvgMeter
from newdataset_modified import MultiDataset


from config_defaults import _C as cfg
import math

#import segmentation_models_pytorch as smp



import pdb
import os

##############################################
###### BEGIN CUSTOM CLUSTER OVERHEAD
#import os
#import signal
#import sys
#
#
#class ClusterStateManager:
#    def __init__(self, time_to_run=600): ####need to change time to run ??600s
#        self.external_exit = None
#        self.timer_exit = False
#
#        signal.signal(signal.SIGTERM, self.signal_handler)
#        signal.signal(signal.SIGINT, self.signal_handler)
#        signal.signal(signal.SIGALRM, self.timer_handler)
#        signal.alarm(time_to_run)
#
#    def signal_handler(self, signal, frame):
#        print("Received signal [", signal, "]")
#        self.external_exit = signal
#
#    def timer_handler(self, signal, frame):
#        print("Received alarm [", signal, "]")
#        self.timer_exit = True
#
#    def should_exit(self):
#        if self.timer_exit:
#            return True
#
#        if self.external_exit is not None:
#            return True
#
#        return False
#
#    def get_exit_code(self):
#        if self.timer_exit:
#            return 3
#
#        if self.external_exit is not None:
#            return 0
#
#        return 0
#    
##############################################    


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



label_weight = 1 / np.log(1.02 + np.array(label_ratio))
label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)                    ####tangent image has four, Todo: take a careful look at tangent image


# Start the timer on how long you're allowed to run
#csm = ClusterStateManager(3600)


logging.getLogger().setLevel(logging.INFO)   ###to output logging.info

####################################
def check_arguments(cfg, args):
    assert os.path.exists(cfg.PATH), 'Data root folder does not exist.'
    
    assert cfg.SOLID in ['cir', 'ico', 'ded', 'tru'], 'Dataset not supported.'
    
    assert cfg.PR_MODEL in ['resnet18', 'resnet34', 'resnet50','resnet101', 'mobilenet_v2']   ###to do, may add more models
    
    assert cfg.CHOOSE_MODEL in ['group', 'mv', 'pool', 'groupcat']
    
    
    if cfg.CHOOSE_MODEL in ['group', 'groupcat']:
        if cfg.PR_MODEL == 'mobilenet_v2':
            assert cfg.GC_LAYER[-1] == 1280
        
        elif cfg.PR_MODEL in ['resnet18','resnet34']:
            assert cfg.GC_LAYER[-1] == 512
        
        elif cfg.PR_MODEL in ['resnet50', 'resnet101']:
            assert cfg.GC_LAYER[-1] == 2048                   ###to do : may add more constraint on dataset settings
    

################################### need to change 

def train_net(net,
              device,
              solid_name,
              angle,
              path,
              ypropx,
              nov,
              nol,
              eval_epoch,
              epochs=5,
              batch_size=1,
              lr=0.01,
              decay=True,
              save_cp=True,
              skip_train=True,
              choose_model='group',
              layer_multiply=1,
              n_element=60,
              resolui=560,
              resoluj=224,
              pretrain=True):
    
    if solid_name=='cir':
        check_exi_path='ugnetcheckpoint_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'_'+choose_model+'_'+str(layer_multiply)+'_'+str(pretrain)
        log_path='log_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'_'+choose_model+'_'+str(layer_multiply)+'_'+str(pretrain)
    else:
        check_exi_path='ugnetcheckpoint_'+str(solid_name)+'_'+str(resolui)+'_'+str(angle)+'_'+choose_model+'_'+str(layer_multiply)+'_'+str(pretrain)
        log_path='log_'+str(solid_name)+'_'+str(resolui)+'_'+str(angle)+'_'+choose_model+'_'+str(layer_multiply)+'_'+str(pretrain)
        
    
    
    w = torch.tensor(label_weight).to(device)
    train_dataset=MultiDataset(solid_name,angle,path,resolui,resoluj,ypropx,nov,nol,test_mode='train')
    test_dataset=MultiDataset(solid_name,angle,path,resolui,resoluj,ypropx,nov,nol,test_mode='test')

    n_train=len(train_dataset)
    n_val=len(test_dataset)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) #####num_worker=0?  if memory is not enough, the pin_memory can be set as False
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    
    writer = SummaryWriter(log_path, comment=f'LR_{lr}_BS_{batch_size}')
    
    global_step = 0

    start_epoch=0
    best_avg_acc=-1
    
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8)
    
    if decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    #print(check_exi_path+ f'/best.pth')
    if os.path.exists(check_exi_path+ f'/latest.pth'):
        checkpoint = torch.load(check_exi_path+f'/latest.pth')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch=checkpoint['epoch']
        del checkpoint
    if os.path.exists(check_exi_path+ f'/best.pth'):
        checkpoint = torch.load(check_exi_path+f'/best.pth')
        best_avg_acc=checkpoint['mean_Accuracy']
        best_mean_iou=checkpoint['mean_IOU']
        del checkpoint
        
    
    if start_epoch>epochs:
        skip_train=True
        
        
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

  
    
   
    
    
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    
    #record_steps=math.floor(len(train_loader)/3)
    
    
    if not skip_train:
        for epoch in range(start_epoch, epochs):
            train_state=dict()
            net.train()
            epoch_loss = AvgMeter()
            
            
            
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                
                
                for step_i, [input_im,semantic] in enumerate(train_loader):
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    s=input_im.shape[0]
                    input_im=input_im.to(device,dtype=torch.float32).permute(0,1,4,2,3).contiguous()     ###(b,g,4,w,h)
                    semantic=semantic.to(device,dtype=mask_type).permute(0,1,4,2,3).contiguous()   ####(b,g,1,w,h)
                    input_im=input_im.view(-1,input_im.shape[2],input_im.shape[3],input_im.shape[4]) ##(b*g,4,w,h)
                    semantic=semantic.view(-1,semantic.shape[2],semantic.shape[3],semantic.shape[4]).squeeze()  ##(b*g,w,h)
                        
          
                    masks_pred = net(input_im)
                    
                    loss = criterion(masks_pred, semantic)
                    
                    epoch_loss.update(loss.item()) 
                    
                    
                    pbar.set_postfix(**{'loss (batch)': epoch_loss.avg})
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
                    pbar.update(s)          
                    global_step += 1
                    
            
                 
            writer.add_scalar('Loss/train',epoch_loss.avg,epoch)
            

            
            if eval_epoch:
                ious, accs, tot, ious2, accs2=eval_net(net,val_loader,device,writer)
                ious_mean=np.mean(ious)
                accs_mean=np.mean(accs)
                ious2_mean=np.mean(ious2)
                accs2_mean=np.mean(accs2)
                
                #logging.info('Tagent MIoU: {}'.format(ious_mean))
                #logging.info('Mean Tagent Accuracy: {}'.format(accs_mean))
                logging.info('MIoU: {}'.format(ious2_mean))
                logging.info('Mean Accuracy: {}'.format(accs2_mean))
                logging.info('Avg loss: {}'.format(tot))
                
            
            if decay:
                scheduler.step()
                 
            train_state['epoch']=epoch+1
            train_state['state_dict']=net.state_dict()
            train_state['optimizer']=optimizer.state_dict()
            train_state['scheduler']=scheduler.state_dict()
            train_state['IOU']=ious2
            train_state['mean_IOU']=ious2_mean
            train_state['Accuracy']=accs2
            train_state['mean_Accuracy']=accs2_mean
            #train_state['tangent_IOU']=ious
            #train_state['tangent_Accuracy']=accs
            #train_state['tangent_mean_IOU']=ious_mean
            #train_state['tangent_mean_Accuracy']=accs_mean
            #train_state['Avg_loss']=tot
            
            
            writer.add_scalar('Loss/test',tot,epoch)
            writer.add_scalar('Tangent_Acc/test',accs_mean,epoch)
            writer.add_scalar('Tangent_IOU/test',ious_mean,epoch)
            writer.add_scalar('Sphere_Acc/test',accs2_mean,epoch)
            writer.add_scalar('Sphere_IOU/test',ious2_mean,epoch)
            
            if  best_avg_acc < accs2_mean:
                try:
                    os.mkdir(check_exi_path)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                best_avg_acc=accs2_mean
                torch.save(train_state,
                       check_exi_path + f'/best.pth')
                logging.info(f'Best_model saved !')
                
           
            
            
            # Once again check if we should exit
            #if csm.should_exit():
                #break
            
        
        if save_cp:
            try:
                os.mkdir(check_exi_path)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(train_state,
                       check_exi_path + f'/latest.pth')
            logging.info(f'Checkpoint saved !')
                
                
        
    
        writer.close()
    else:
          logging.info('Avg accuracy: {}'.format(best_avg_acc))
          logging.info('Avg IOU: {}'.format(best_mean_iou))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='cir212-group-mobile.yaml', help='Path to configuration file.')
    parser.add_argument('overwrite_args', help='Overwrite args from config_file through the command line', default=None,nargs=argparse.REMAINDER)
    
    return parser.parse_args()


def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  ??? not sure function, check me!!!
    args = get_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.overwrite_args)
    cfg.freeze()
    check_arguments(cfg, args)
    print(cfg.dump()) ### check me, not sure

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    
    if cfg.PR_MODEL=='custom':
        if cfg.CHOOSE_MODEL=='group' and cfg.SOLID=='cir':
            net = UNet(n_channels=4, n_classes=15, gc_layer=cfg.GC_LAYER,layer_mul=cfg.LAYER_MULTIPLY, n_element=cfg.N_ELEMENT)          ###change numb
        elif cfg.CHOOSE_MODEL=='group':
            net = UNet(n_channels=4, n_classes=15, gc_layer=cfg.GC_LAYER,layer_mul=cfg.LAYER_MULTIPLY, n_element=60)
        elif cfg.CHOOSE_MODEL=='mv':
            net=UNet_mv(n_channels=4,n_classes=15,layer_mul=cfg.LAYER_MULTIPLY)
        elif cfg.CHOOSE_MODEL=='pool':
            net=Pool_UNet_mv(n_channels=4,n_classes=15,layer_mul=cfg.LAYER_MULTIPLY,poolw=cfg.POOL_FORMAT,n_element=cfg.N_ELEMENT,n_layer=cfg.NOL)
        elif cfg.CHOOSE_MODEL=='groupcat':
            net=Catgroup_UNet(n_channels=4,n_classes=15,gc_layer=cfg.GC_LAYER,layer_mul=cfg.LAYER_MULTIPLY,n_element=cfg.N_ELEMENT)
    else:
        if cfg.CHOOSE_MODEL=='group':
            net=Unet_group(n_classes=15,solid_name=cfg.SOLID,n_element=cfg.N_ELEMENT,gc_layer=cfg.GC_LAYER,pr_model=cfg.PR_MODEL,pretrain=cfg.PRETRAIN)
        elif cfg.CHOOSE_MODEL=='mv':
            net=Unet_mv(n_classes=15, pr_model=cfg.PR_MODEL, pretrain=cfg.PRETRAIN)
        elif cfg.CHOOSE_MODEL=='pool':
            net=Unet_pool(n_classes=15,n_view=cfg.N_ELEMENT*cfg.NOL, pool_way=cfg.POOL_FORMAT,pr_model=cfg.PR_MODEL,pretrain=cfg.PRETRAIN)
        elif cfg.CHOOSE_MODEL=='groupcat':
            net=Unet_groupcat(n_classes=15,solid_name=cfg.SOLID,n_element=cfg.N_ELEMENT,gc_layer=cfg.GC_LAYER, pr_model=cfg.PR_MODEL,pretrain=cfg.PRETRAIN)
    
    
    
    
    
    num_parameters=count_parameters(net)
    print(num_parameters)
    
    
    
    logging.info(f'Network:\n'
                 f'\t{net.n_classes} output channels (classes)\n')
                 #f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
 
   
        

    net.to(device=device)
   
    
    train_net(net=net,
              device=device,
              solid_name=cfg.SOLID,
              angle=cfg.CROSSWISE,
              path=cfg.PATH,
              ypropx=cfg.YPROPX,
              nov=cfg.NOV,
              nol=cfg.NOL,
              eval_epoch=cfg.EVAL_EPOCH,
              epochs=cfg.EPOCHS,
              batch_size=cfg.BATCH_SIZE,
              lr=cfg.LR,
              decay=cfg.DECAY,
              save_cp=cfg.SAVE_CP,
              skip_train=cfg.SKIP_TRAIN,
              choose_model=cfg.CHOOSE_MODEL,
              layer_multiply=cfg.LAYER_MULTIPLY,
              n_element=cfg.N_ELEMENT,
              resolui=cfg.RESI,
              resoluj=cfg.RESJ,
              pretrain=cfg.PRETRAIN
                  )
    # Exit with the exit code the ClusterStateManager has set for you
    #print("Exiting with exit code", csm.get_exit_code())
    #sys.exit(csm.get_exit_code())
        
        
    #except KeyboardInterrupt:
        #torch.save(net.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        #try:
            #sys.exit(0)
        #except SystemExit:
            #os._exit(0)
