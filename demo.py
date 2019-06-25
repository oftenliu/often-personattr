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
from datafolder.folder import Test_Dataset,Train_Dataset
from net import *
from PIL import Image
import cv2


######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'market',
    'duke'  :  'duke',
    'all':     'all'
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':26 ,'all':10}
num_ids_dict = { 'market':751, 'duke':702 }

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/home/xxx/reid/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='duke', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=1, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='19', type=str, help='0,1,2,3...or last')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
model_dir = os.path.join('./checkpoints', 'combine_backpack_bag', args.model)
#model_dir = os.path.join('./checkpoints', args.dataset, args.model)
result_dir = os.path.join('./result', args.dataset, args.model)
 
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

######################################################################
# Argument
# --------
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
num_cls = num_cls_dict[args.dataset]

######################################################################
# Function
# --------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
output_path = './output/'
label = {
        'market':['bag', 'handbag',  'down','up', 'hair', 'hat', 'gender', 'upcolor','downcolor' ] 
        }
person_gender = ['gender:male','gender:female']
person_down = ["long lower body clothing", "short lower body clothing"]
person_up = ["long sleeve", "short sleeve"]
person_hair = ["short hair","long hair"]
person_upcolor = ['upblack', 'upwhite', 'upred', 'uppurple', 'upyellow','upgray', 'upblue', 'upgreen','upothter' ]
person_downcolor = ['downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown', 'downothter']
def display_market(preds,filename):
    print(filename)
    img = cv2.imread(filename)

    img = cv2.resize(img,(144,288))
    img = cv2.copyMakeBorder(img, 0, 144, 0, 288, cv2.BORDER_CONSTANT, value = (255,255,255))
    index = 0
    for i in range(0,2):
        if preds[i] == 1:            
            img = cv2.putText(img,label['market'][i],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            index = index +1
            break    

    img = cv2.putText(img,person_down[preds[2]],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1

    img = cv2.putText(img,person_up[preds[3]],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1

    img = cv2.putText(img,person_hair[preds[4]],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1

    if preds[5] == 1:
        img = cv2.putText(img,label['market'][5],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        index = index +1        

    img = cv2.putText(img,person_gender[preds[6]],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1
         
    img = cv2.putText(img,person_upcolor[int(preds[7])],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1

    img = cv2.putText(img,person_downcolor[preds[8]],(180,10+index*20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    index = index +1
    filename = output_path + filename.split('/')[-1]
    cv2.imwrite(filename,img)
    cv2.imshow('0',img)
    cv2.waitKey(0)

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
            

# ######################################################################
# # Load Data
# # ---------
# image_datasets = {}

# image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
#                                          train_val='train')
# image_datasets['query'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
#                                       train_val='train')
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
#                                              shuffle=True, num_workers=args.num_workers)
#               for x in ['train', 'query']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'query']}
# labels_name = image_datasets['train'].labels()
# num_label = image_datasets['train'].num_label()
image_datasets = []

path = './test_image'
#path = './dataset/duke/bounding_box_test/'
traverse(path,image_datasets)

#print(image_datasets)
# Model
# ---------
model = model_dict[args.model](9,9,7)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode

######################################################################
# Testing
# ------------------
since = time.time()

overall_acc = 0
each_acc = 0
# Iterate over data.
since = time.time()
#for count, data in enumerate(dataloaders['gallery']):
for index in range(0,len(image_datasets)):
    # get the inputs
    #image_datasets[index] = './test_image/test.jpg'
    image = Image.open(image_datasets[index])
    image_tensor =  T.Compose([
                    T.Resize(size=(288, 144)),
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(image)
    images = image_tensor.unsqueeze(0)
    # wrap them in Variable
    if use_gpu:
        images = images.cuda()

    # forward
    outputs = model(images)
    #print(outputs)
    preds = torch.zeros(10)
    for c in range(7):
        #print(outputs[c])
        pred = torch.gt(outputs[c], torch.zeros_like(outputs[c]) ).data
        pred = pred.squeeze(1)
        preds[c] = pred

    outputs_upcolor = torch.max(outputs[7],1)[1].data.byte()
    preds[7] = outputs_upcolor
    outputs_downcolor = torch.max(outputs[8],1)[1].data.byte()        
    # preds = torch.gt(outputs, torch.zeros_like(outputs) ).data#sigmod输出　阈值为0.5
    preds[8] = outputs_downcolor
    preds = preds.data.byte()
    #print(preds)
    # preds = preds.squeeze(0)    
    # print(preds)
    # #display_duke(preds,image_datasets[index])
    display_market(preds,image_datasets[index])


time_elapsed = time.time() - since
print('Testing complete in  {:.0f}ms'.format(time_elapsed*1000 ))

