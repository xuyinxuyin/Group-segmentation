#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:43:22 2020

@author: xuyin
"""

from __future__ import division
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
from scipy.io import loadmat
import argparse
import math
import pdb

parser = argparse.ArgumentParser(description='PyTorch process segmentation')
parser.add_argument('--resi',type=int, default=384, help='resolution of image on row') #format of homogeneous space
parser.add_argument('--resj',type=int, default=384, help='resolution of image on column')
parser.add_argument('--crosswise', type=float, default=45, help='resolution of image') #format of homogeneous space
#parser.add_argument('--crop', type=float, default=40, help='resolution of image')
parser.add_argument('--ypropx', type=float, default=1, help='proportion of y and x')
parser.add_argument('--nov', type=int, default=12, help='number of view')
parser.add_argument('--nolayers', type=int, default=2, help='number of layer of view')

args = parser.parse_args()



def get_vectors(point,resolui,resoluj,angle,ypropx,nol):
    r=(resoluj/2)/(np.tan(angle))
    
    res_grid_vector=np.zeros([resolui*nol,resoluj,3])
#   each_anglei=np.pi/(resolui-1)
#   each_anglej=(2*angle)/(resoluj-1)
######
    lat_angle=np.pi/(nol*2)
    axis=np.asarray([0,0,1])
    rotate_axis=np.cross(point[:,0],axis).reshape(3,1)
    rotation1=rotation_matrix(rotate_axis, lat_angle)
    point1=rotation1.dot(point)
    rotation2=rotation_matrix(rotate_axis,-lat_angle)
    point2=rotation2.dot(point)
    point_stack=np.stack([point1,point2])
    #print('point:')
    #print(point1)
    #print(point2)
##### 
    for i_layer in range(nol):
        point=point_stack[i_layer,:,:]
        a=point[0,0]
        b=point[1,0]
        c=point[2,0]
        canon_axis=np.asarray([-a*c,-b*c,a**2+b**2])
        canon_axis=(canon_axis/np.linalg.norm(canon_axis)).reshape(3,1)
        
        for i in range(resolui):
            for j in range(resoluj):
                x=-(resoluj-1)/2+j
                y=((resolui-1)/2-i)*ypropx
                canon_angle=math.atan2(y,x)-np.pi/2
                rot1=rotation_matrix(point/np.linalg.norm(point),canon_angle)
                new_canon=rot1.dot(canon_axis)
                sita=np.sqrt(x**2+y**2)
                pp=r*point+sita*new_canon
                pp=pp/np.linalg.norm(pp)
                res_grid_vector[i_layer*resolui+i,j,0]=pp[0,0]
                res_grid_vector[i_layer*resolui+i,j,1]=pp[1,0]
                res_grid_vector[i_layer*resolui+i,j,2]=pp[2,0]
    return res_grid_vector



def get_latlong(point):
    long=math.atan2(point[1,0],point[0,0])
    if long<0:
        long=long+2*np.pi
    lat=math.atan2(point[2,0],np.sqrt(point[1,0]**2+point[0,0]**2))
    return long,lat
#
def rad2pos(long,lat):
    x=long/(np.pi*2)*2-1
    y=(np.pi/2-lat)/np.pi*2-1
    return x,y

def rotation_matrix(axis,theta):
    w1=axis[0,0]
    w2=axis[1,0]
    w3=axis[2,0]
    ww=np.zeros([3,3])
    ww[0,1]=-w3
    ww[0,2]=w2
    ww[1,0]=w3
    ww[1,2]=-w1
    ww[2,0]=-w2
    ww[2,1]=w1
    rott=np.eye(3)+np.sin(theta)*ww+(1-np.cos(theta))*ww.dot(ww)
    return rott



def get_final_12res(origin_po,resolui,resoluj,angle,ypropx,nov,nol):
    angle2=angle
    angle=(angle/180)*np.pi
 
    final_res=np.zeros([nov,resolui*nol,resoluj,2])
    
    
    
    original_vector=get_vectors(origin_po,resolui,resoluj,angle,ypropx,nol)
    
    np.save('project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle2)+'_'+str(ypropx)+'original_vector.npy',original_vector)
    

    final_vector=np.zeros([nov,resolui*nol,resoluj,3])
    point=np.zeros([3,1])
    axis=np.asarray([0,0,1]).reshape(3,1)
    

    
    for i in range(nov):
        rotation=rotation_matrix(axis,2*np.pi/nov*i)
        stack_origin_vectors=np.concatenate([original_vector[i,:,:]for i in range(nol*resolui)])
      
        stack_new_vectors=np.transpose(rotation.dot(np.transpose(stack_origin_vectors,[1,0])),[1,0])
        new_vectors=np.stack([stack_new_vectors[i*resoluj:i*resoluj+resoluj,:] for i in range(nol*resolui)],axis=0)
        
        for j in range(nol*resolui):
            for k in range(resoluj):
                point=new_vectors[j,k,:].reshape(3,1)
                long,lat=get_latlong(point)
                x,y=rad2pos(long,lat)
                final_res[i,j,k,0]=x
                final_res[i,j,k,1]=y
                final_vector[i,j,k,0]=point[0,0]
                final_vector[i,j,k,1]=point[1,0]
                final_vector[i,j,k,2]=point[2,0]
    final_res1=np.concatenate([final_res[:,i*resolui:(i+1)*resolui,:,:] for i in range(nol)],axis=0)
    final_vector1=np.concatenate([final_vector[:,i*resolui:(i+1)*resolui,:,:] for i in range(nol)],axis=0)
                
    return final_res1, final_vector1



def tan2sph_uv(resolui,resoluj,nol,nov,ypropx,angle):
    uv_grid=np.zeros([2048,4096,3])
    angle2=angle/180.*np.pi
    focal_length=(resoluj/2)/(np.tan(angle2))
    center_lib=np.zeros([nol,nov,3])
    y_axis_lib=np.zeros([nol,nov,3])
    x_axis_lib=np.zeros([nol,nov,3])
    for layer_id in range(nol):
        for group_id in range(nov):
            center=center=np.array([math.cos(np.pi/(2*nol))*math.cos(2*group_id*np.pi/nov),
                             math.cos(np.pi/(2*nol))*math.sin(2*group_id*np.pi/nov),
                             (1-2*layer_id)*math.sin(np.pi/(2*nol))])
            #print("center:")
            #print(center)
            center_lib[layer_id,group_id,0]=center[0]
            center_lib[layer_id,group_id,1]=center[1]
            center_lib[layer_id,group_id,2]=center[2]
            a=center[0]
            b=center[1]
            c=center[2]
            y_axis=np.array([-c*a,-c*b,a**2+b**2])
            y_axis=y_axis/np.linalg.norm(y_axis)
            x_axis=np.cross(y_axis,center)
            y_axis_lib[layer_id,group_id,0]=y_axis[0]
            y_axis_lib[layer_id,group_id,1]=y_axis[1]
            y_axis_lib[layer_id,group_id,2]=y_axis[2]
            x_axis_lib[layer_id,group_id,0]=x_axis[0]
            x_axis_lib[layer_id,group_id,1]=x_axis[1]
            x_axis_lib[layer_id,group_id,2]=x_axis[2]
    for i in range(4096):
        for j in range(2048):
            ######
            rad=i*2*np.pi/4096.-np.pi/2
            if rad<0:
                rad+=2*np.pi
            #######
            lat=np.pi/2.-j*np.pi/2048.
            vector=np.array([math.cos(lat)*math.cos(rad),
                             math.cos(lat)*math.sin(rad),
                             math.sin(lat)])
            layer_id=math.floor((np.pi/2.-lat)/(np.pi/nol))
            if rad>2*np.pi-np.pi/nov:
                rad-=2*np.pi
            group_id=math.floor((rad+np.pi/nov)/(2*np.pi/nov))
#            ###############
#            group_id=(group_id-3)%nov
#            rad=rad-6*np.pi/nov
#            vector=np.array([math.cos(lat)*math.cos(rad),
#                             math.cos(lat)*math.sin(rad),
#                             math.sin(lat)])
#            ###########
            
            
            view_id=layer_id*nov+group_id
            
            center=center_lib[layer_id,group_id,:]
            y_axis=y_axis_lib[layer_id,group_id,:]
            x_axis=x_axis_lib[layer_id,group_id,:]
            
            #center=np.array([math.cos(np.pi/(2*nol))*math.cos(2*group_id*np.pi/nov),
                             #math.cos(np.pi/(2*nol))*math.sin(2*group_id*np.pi/nov),
                             #(2*layer_id-1)*math.sin(np.pi/(2*nol))])
            #print(np.linalg.norm(center))
            #print(np.linalg.norm(vector))
            
            #theta=np.arccos(np.dot(vector,center))
            #if theta>2:
                #print(center)
                #print(vector)
                #print(theta)
                #pdb.set_trace()
            tangent_vector=focal_length*vector/(np.dot(vector,center))-center*focal_length
            #a=center[0]
            #b=center[1]
            #c=center[2]
            
            #y_axis=np.array([-c*a,-c*b,a**2+b**2])
            #y_axis=y_axis/np.linalg.norm(y_axis)
            #x_axis=np.cross(y_axis,center)
            #print(np.linalg.norm(x_axis))
            #print(np.linalg.norm(y_axis))
            
            x_coor=np.dot(tangent_vector,x_axis)
            y_coor=np.dot(tangent_vector,y_axis)
            #print(x_coor)
            v=x_coor/(resoluj/2)
            u=-y_coor/(resolui/(2*ypropx))
            #if rad>2*np.pi-np.pi/nov:
                #rad-=2*np.pi
            
            #v=(math.tan(rad-group_id*(2*np.pi/nov))*focal_length)/(resoluj/2)
            #u=math.tan(np.pi/2-(layer_id*np.pi/nol+np.pi/(2*nol))-lat)*focal_length/(resolui/2*ypropx)
            #if view_id>nol*nov-1 or view_id<0:
                #print(view_id)
                #print('error: view_id surpass the limit')
            #uv_grid[j,i,2]=-1
            uv_grid[j,i,2]=(view_id/(nol*nov-1)-0.5)*2
            uv_grid[j,i,0]=v
            uv_grid[j,i,1]=u
    return uv_grid






def main():
    resolui=args.resi
    resoluj=args.resj
    angle=args.crosswise
    origin_po=np.zeros([3,1])
    ypropx=args.ypropx
    nov=args.nov
    nol=args.nolayers
    
    
    
    origin_po[0,0]=0.
    origin_po[1,0]=1.
    origin_po[2,0]=0.
    res_grid,final_vector=get_final_12res(origin_po,resolui,resoluj,angle,ypropx,nov,nol)
    uv_grid=tan2sph_uv(resolui,resoluj,nol,nov,ypropx,angle)
    
    
   # np.save('project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'final_grid.npy',res_grid)
   # np.save('project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'final_vector.npy',final_vector)
    np.save('project_'+str(nol)+'_'+str(nov)+'_'+str(resolui)+'_'+str(resoluj)+'_'+str(angle)+'_'+str(ypropx)+'uv_grid.npy',uv_grid)
    
    
    
if __name__ == "__main__":
    main()
    

