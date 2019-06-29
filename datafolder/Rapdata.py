import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from .reid_dataset.import_rap2 import import_rapdata

import cv2


# 'attachment-Backpack', 'attachment-ShoulderBag','attachment-WaistBag',
# 'attachment-HandBag','attachment-PlasticBag','attachment-PaperBag', 
#  'attachment-Box', 
# 'attachment-HandTrunk',
#  'attachment-Baby',
#  'attachment-Other',


class RapTrain_Dataset(data.Dataset):

    def __init__(self, data_dir, dataset_name, transforms=None, train_val='train' ):
              
        train, val,test,label = import_rapdata(data_dir,dataset_name)
        self.label = label
        #train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        #train_attr, test_attr, self.label =  import_Market1501Attribute_binary(data_dir)
        #print(self.label)
        self.num_ids = len(train['data'])

        self.num_labels = len(self.label)

        # numbers = 0
        # for v in train['attr']:
        #     for i in range(88,98):
        #         print(self.label[i])
        #         if v[i] == 1:
        #             dir = os.path.join(data_dir,self.label[i])
        #             if not os.path.exists(dir):
        #                 os.makedirs(dir)
        #             img = cv2.imread(train['data'][numbers])
        #             cv2.imwrite(os.path.join(dir,str(numbers) + '.jpg',),img)
        #     numbers +=1 
        # exit(0)

        # for v in train['attr']:
        #     print(self.label[21],self.label[30])
        #     v = v[21:30]
        #     count = np.sum(v)
        #     print(v)
        #     if count > 1:
        #         numbers += 1

        # print(numbers)
        # exit(0)
        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels,dtype = np.int32)
        for v in train['attr']:
            distribution += np.array(v,dtype = np.int32)
        self.distribution = distribution / len(train['attr'])
        print(self.distribution)
        self.select_label = np.zeros(self.num_labels,dtype = bool)

        for i,label in enumerate(self.label):
            labelname,value = label.strip().split(':')
            if value == '1':
                self.select_label[i] = True 
                print(labelname + ':' + str(distribution[i]))
        if train_val == 'train':
            self.train_data = train

        elif train_val == 'val':
            self.train_data = val

        elif train_val == 'test':
            self.train_data = test

        else:
            print('Input should only be train or val')


        self.weight_pos = self.distribution[self.select_label]
        print(self.weight_pos)
        self.label = np.asarray(self.label)[self.select_label]
        self.num_labels = len(self.label)
        print(self.label)
        print(self.num_ids)

        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.train_data['data'][index]

        label = np.asarray(self.train_data['attr'][index])
        data = Image.open(img_path)
        data = self.transforms(data)
        label = label[self.select_label]
        
        return data,  label

    def __len__(self):
        return len(self.train_data['data'])

    def num_label(self):
        return self.num_labels

    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label
    def weight(self):
        return self.weight_pos



class RapTest_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, transforms=None, train_val='train' ):
              
        train, val,test,label = import_rapdata(data_dir,dataset_name)
        self.label = label

        #train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        #train_attr, test_attr, self.label =  import_Market1501Attribute_binary(data_dir)
        #print(self.label)
        self.num_ids = len(train['data'])

        self.num_labels = len(self.label)
        
        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels,dtype = np.int32)
        for v in train['attr']:
            distribution += np.array(v,dtype = np.int32)
        self.distribution = distribution / len(train['attr'])

        self.select_label = np.zeros(self.num_labels,dtype = bool)

        for i,label in enumerate(self.label):
            labelname,value = label.strip().split(':')
            if value == '1':
               self.select_label[i] = True 

        if train_val == 'train':
            self.test_data = train

        elif train_val == 'val':
            self.test_data = val

        elif train_val == 'test':
            self.test_data = test

        else:
            print('Input should only be train or val')

        label =  np.asarray(self.test_data['attr'][0])

        label = label[self.select_label]
        self.weight_pos = 1 - self.distribution

        self.label = np.asarray(self.label)[self.select_label]
        self.num_labels = len(self.label)
        print(self.label)



        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data['data'][index]

        label = np.asarray(self.test_data['attr'][index])
        data = Image.open(img_path)
        data = self.transforms(data)
        label = label[self.select_label]
        return data,  label


    def __len__(self):
        return len(self.test_data['data'])

    def labels(self):
        return self.label
    def num_label(self):
        return self.num_labels
