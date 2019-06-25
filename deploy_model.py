# !/usr/local/bin/python3
import matplotlib
matplotlib.use('agg')

import os
import time
import argparse
import scipy.io

import torch
from torch.autograd import Variable
from datafolder.folder import Test_Dataset
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
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--dataset', default='duke', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--which-epoch',default='49', type=str, help='0,1,2,3...or last')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

model_dir = os.path.join('./checkpoints', args.dataset, args.model)


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

######################################################################
# Model
# ---------
model = model_dict[args.model](num_cls)
model = load_network(model)
# if use_gpu:
#     model = model.cuda()
model.train(False)  # Set model to evaluate mode



# An example input you would normally provide to your model's forward() method.
#example = torch.rand(3,3, 144,288)

input_shape = (3,288,144)
dummy_input = Variable(torch.randn(1, *input_shape))

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, example)

# save
#traced_script_module.save("resnet50.pt")


input_names = [ "data" ]
output_names = [ "pred1" ]



#out = torchModel(input_var)

torchOut = torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True)
