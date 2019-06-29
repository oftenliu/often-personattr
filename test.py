# !/usr/local/bin/python3
import matplotlib
matplotlib.use('agg')

import os
import time
import argparse
import scipy.io

import torch

from datafolder.folder import Test_Dataset
from datafolder.Rapdata import RapTest_Dataset
from net import *

import config
######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'market',
    'duke'  :  'duke',
    'all':     'all',
    'rap':     'rap',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/home/xxx/reid/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='duke', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=16, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='79', type=str, help='0,1,2,3...or last')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
#model_dir = os.path.join('./checkpoints', args.dataset, args.model)
model_dir = os.path.join('./checkpoints', 'combine_backpack_bag', args.model)
result_dir = os.path.join('./result', args.dataset, args.model)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

######################################################################
# Argument
# --------
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

######################################################################
# Function
# --------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# DataLoader
# ---------
image_datasets = {}

if args.dataset == 'rap':
    image_datasets['test'] = RapTest_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                        train_val='test')    
else:
    image_datasets['test'] = Test_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                       query_gallery='query')


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
labels_name = image_datasets['test'].labels()
num_label = image_datasets['test'].num_label()

# Model
# ---------
model = model_dict[args.model](config.class_value,config.class_name)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode


weights = torch.zeros(num_label)
for i in range(0,num_label):
    weights[i] = 0.5
print(weights)
weights = weights.cuda()
criterion = PersonAttr_Loss(weights,config.class_value,config.class_name)

######################################################################
# Testing
# ------------------
since = time.time()

overall_acc = 0
each_acc = 0
# Iterate over data.
for count, data in enumerate(dataloaders['test']):
    # get the inputs
    images, labels = data
    # wrap them in Variable
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    labels = labels.float()
    # forward
    outputs = model(images)
    label_loss,pred_success = criterion(outputs, labels)
    
    overall_acc += pred_success


overall_acc = overall_acc / dataset_sizes['test']


print('{} Acc: {:.4f}'.format('Overall', overall_acc))
