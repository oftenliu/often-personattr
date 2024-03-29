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
from datafolder.Rapdata import RapTrain_Dataset,RapTest_Dataset
from net import *
import torch.nn.functional as F
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
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epoch', default=80, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
#model_dir = os.path.join('./checkpoints', args.dataset, args.model)
model_dir = os.path.join('./checkpoints', 'rap_multilabel_softmax_weight', args.model)

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
y_loss['test'] = []
y_err = {}
y_err['train'] = []
y_err['test'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['test'], 'ro-', label='test')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['test'], 'ro-', label='test')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}

if args.dataset == 'rap':
    image_datasets['train'] = RapTrain_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                            train_val='train')

    image_datasets['test'] = RapTrain_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                        train_val='val')    
else:
    image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                            train_val='train')

    image_datasets['test'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                        train_val='val')




dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

images, labels = next(iter(dataloaders['train']))

num_label = image_datasets['train'].num_label()
weight_pos = image_datasets['train'].weight()
num_id = image_datasets['train'].num_id()
labels_list = image_datasets['train'].labels()

print(dataset_sizes)
######################################################################
# Model and Optimizer
# ------------------
model = model_dict[args.model](config.class_value,config.class_name)
# #save_path = os.path.join(model_dir,'net_59.pth')
# save_path = os.path.join('./checkpoints/combine_backpack_bag/resnet50','net_69.pth')
# model.load_state_dict(torch.load(save_path))

# print(model)
# for param in model.parameters():
#     #print(param)
#     param.requires_grad = False

# for param in model.class_2.parameters():
#     #print(param)
#     param.requires_grad = True

if use_gpu:
    model = model.cuda()
# loss

weights = torch.zeros(len(weight_pos))
for i in range(0,len(weight_pos)):
    weights[i] = weight_pos[i]
weights = weights.cuda()

criterion = PersonAttr_Loss(weights,config.class_value,config.class_name)
#criterion = F.binary_cross_entropy_with_logits
# optimizer
print("class value " + str(len(config.class_value)))
print("class name " + str(len(config.class_name)))
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9,
                            weight_decay = 5e-4, nesterov = True,)

# optimizer = torch.optim.SGD(
#                         filter(lambda p: p.requires_grad, model.parameters()),#重要的是这一句
#                          lr = 0.001, momentum = 0.9,weight_decay = 5e-4, nesterov = True,)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1,)


######################################################################
# Training the model
# ------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
        #for phase in ['test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, data in enumerate(dataloaders[phase]):
                # get the inputs
                #print(data)
                images, labels = data
                #print(len(labels[0]))
                # weights = torch.zeros(labels.shape)
                # for i in range(labels.shape[0]):
                #     for j in range(labels.shape[1]):
                #         if labels.data.cpu()[i, j] == 1:
                #             weights[i, j] = weight_pos[j]
                #         elif labels.data.cpu()[i, j] == 0:
                #             weights[i, j] = 1 - weight_pos[j]
                #         else:
                #             weights[i, j] = 0                
                # wrap them in Variable
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                    #weights = weights.cuda()
                images = images
                labels = labels.float()


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(images)
                #print("class {} pred {},label {}".format(0,outputs[0],labels[:,0]))
                #print(outputs.shape)
                #print(labels.size())

                label_loss,pred_success = criterion(outputs, labels) #经过loss函数的计算　outputs的值已经被修改了
                #print("class {} pred {},label {}".format(0,outputs[0],labels[:,0]))
                # backward + optimize only if in training phase
                if phase == 'train':
                    #torch.autograd.set_detect_anomaly(True)
                    label_loss.backward()
                    optimizer.step()

                # statistics
                running_loss += label_loss.item()
                # if phase == 'test':
                #     if label_loss > 10 :
                #         print(weights)
                #         print(labels)
                #         print(outputs)

                running_corrects += pred_success
                print('step : ({}/{})  |  loss : {:.4f}'.format(count*args.batch_size, dataset_sizes[phase], label_loss.item()))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'test':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Main
# -----
model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                    num_epochs = args.num_epoch)
