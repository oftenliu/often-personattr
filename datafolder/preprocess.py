import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from reid_dataset import import_MarketDuke_nodistractors
from reid_dataset.import_MarketDuke_nodistractors import *
from reid_dataset import import_Market1501Attribute_binary
from reid_dataset import import_DukeMTMCAttribute_binary
import cv2
data_dir = '../dataset'
dataset_name = 'market'

save_pos = '../dataset/market/finetune_pos/'
save_neg = '../dataset/market/finetune_neg/'

if not os.path.isdir(save_pos):
    os.makedirs(save_pos)


if not os.path.isdir(save_neg):
    os.makedirs(save_neg)

if dataset_name == 'market':
    dataset_name = ['market']
if dataset_name == 'all':
    dataset_name = ['market','duke']
if dataset_name == 'duke':
    dataset_name = ['duke']
        
train, val,test = import_MarketDuke_nodistractors(data_dir,dataset_name)
train_attr, test_attr, label = import_MarketDuke_attr(data_dir,dataset_name)

train_data = train['data']
for index in range(0,len(train_data)):
    img_path = train_data[index][0]
    id = train_data[index][2]
    label = np.asarray(train_attr[id])
    img = cv2.imread(img_path)
    if label[2] == 1: #finetune backpack
        filename = save_pos + img_path.split('/')[-1]
    else:
        filename = save_neg + img_path.split('/')[-1]
    cv2.imwrite(filename,img)