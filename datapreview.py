# !/usr/local/bin/python3
import matplotlib
matplotlib.use('agg')

import os
import time
import argparse
import scipy.io

import torch
import numpy as np
from torchvision import transforms as T
from datafolder.folder import Test_Dataset
from net import *
from PIL import Image
import cv2



label = ['backpack', 'bag', 'handbag', 'down', 'up', 'hair', 'hat', 'gender', 
         'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen', 'upothter', 
         'downblack','downwhite', 'downred', 'downgray', 'downblue', 'downbrown', 'downothter']
person_down = ["down long clothing", "down short  clothing"]
person_up = ["up long sleeve", "up short sleeve"]
person_hair = ["short hair","long hair"]        
person_gender = ['gender:male','gender:female']
def display(id,labels,img):
    img = cv2.resize(img,(144,288))
    img = cv2.copyMakeBorder(img, 0, 144, 0, 288, cv2.BORDER_CONSTANT, value = (255,255,255))
    index = 0
    for attr in labels:
        label,value = attr.split(' ')
        value = int(value)
        if label == 'down' :
            if value == -1:
                img = cv2.putText(img,attr,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1   
            else:              
                img = cv2.putText(img,person_down[value],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1            
        elif label == 'up':
            if value == -1:
                img = cv2.putText(img,attr,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1 
            else:                
                img = cv2.putText(img,person_up[value],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1        
        elif label == 'gender':
            if value == -1:
                img = cv2.putText(img,attr,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1 
            else:      
                img = cv2.putText(img,person_gender[value],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1        
        elif label == 'hair':
            if value == -1:
                img = cv2.putText(img,attr,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1 
            else:      
                img = cv2.putText(img,person_hair[value],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1

        else:
            if value == 1:
                img = cv2.putText(img,attr,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                index = index +1

    img = cv2.putText(img,id,(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    return img

def traverse(f,imageset):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            ext = os.path.splitext(tmp_path)[-1][1:]
            if ext == "jpg":
                #print('文件: %s'%tmp_path)
                imageset.append(tmp_path)
        else:
            #print('文件夹：%s'%tmp_path)
            traverse(tmp_path,imageset)
            
def get_labels():
    name_file = './dataset/all/anno/duke_train.txt'
    train_attrs = {}
    f_attr = open(name_file,'r')
    lines = f_attr.readlines() 
    for line in lines:
        line = line.strip()   
        id,attrstr = line.split(':')
        attr_list = attrstr.split(',')
        #print(attr_list)
        #print(len(lines))
        id_attr = []
        for attr in attr_list:
            attr = attr.strip()   
            id_attr.append(attr)
        train_attrs[id] = id_attr
    return train_attrs
image_datasets = []

path = './dataset/duke/bounding_box_train'
traverse(path,image_datasets)
train_labels = get_labels()
ids = []
train_image = {}
for image_file in image_datasets:
    id = image_file.split('/')[-1].split('_')[0]    
    if id not in train_image:
        train_image[id] = []
    train_image[id].append(image_file)
print(train_labels)


name_file = './dataset/all/anno/duke_train.txt'
train_attrs = {}
f_attr = open(name_file,'r')
lines = f_attr.readlines() 
for line in lines:
    line = line.strip()   
    id,attrstr = line.split(':')
    attr_list = attrstr.split(',')
    #print(attr_list)
    #print(len(lines))
    id_attr = []
    for attr in attr_list:
        attr = attr.strip()   
        id_attr.append(attr)
    train_attrs[id] = id_attr
    #id = '1508'
    if id not in train_image:
        continue
    id_image_size = len(train_image[id])
    img1_file = train_image[id][0]
    img2_file = train_image[id][int(id_image_size/2)]
    img3_file = train_image[id][id_image_size - 1]
    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)
    img3 = cv2.imread(img3_file)
    attr = train_labels[id]
    
    img1 = display(id,attr,img1)
    img2 = display(id,attr,img2)
    img3 = display(id,attr,img3)
# for image_file in image_datasets:
#     
#     id = image_file.split('/')[-1].split('_')[0]
#     if id not in ids:
#         attr = train_labels[id]
#         display(id,attr,img)
#         ids.append(id)
#         print(id)
    htitch= np.hstack((img1, img2,img3))
    cv2.imshow('0',htitch)
    cv2.waitKey(0)
