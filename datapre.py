# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset,Test_Dataset
from net import *

######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'resnet101'  :  ResNet101_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='/path/to/dataset', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--num-epoch', default=80, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, args.model)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if use_gpu:
        network.cuda()


######################################################################
# Draw Curve
#-----------
x_epoch = []
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}
image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                        train_val='train')

image_datasets['val'] = Test_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                      query_gallery='query')
