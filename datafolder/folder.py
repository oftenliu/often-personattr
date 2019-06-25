import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from .reid_dataset import import_MarketDuke_nodistractors
from .reid_dataset.import_MarketDuke_nodistractors import *
from .reid_dataset import import_Market1501Attribute_binary
from .reid_dataset import import_DukeMTMCAttribute_binary


class Train_Dataset(data.Dataset):

    def __init__(self, data_dir, dataset_name, transforms=None, train_val='train' ):

        if dataset_name == 'market':
            dataset_name = ['market']
        if dataset_name == 'all':
            dataset_name = ['market','duke']
        if dataset_name == 'duke':
            dataset_name = ['duke']
                
        train, val,test = import_MarketDuke_nodistractors(data_dir,dataset_name)
        

        train_attr, test_attr, self.label = import_MarketDuke_attr(data_dir,dataset_name)
        print(self.label)
        #train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        #train_attr, test_attr, self.label =  import_Market1501Attribute_binary(data_dir)
        #print(self.label)
        self.num_ids = len(train['ids'])
        self.num_labels = len(self.label)

        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels,dtype = np.int32)
        for k, v in train_attr.items():
            distribution += np.array(v,dtype = np.int32)
        self.distribution = distribution / len(train_attr)


        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
            self.train_attr = train_attr
        elif train_val == 'val':
            self.train_data = val['data']
            self.train_ids = val['ids']
            self.train_attr = test_attr
        elif train_val == 'test':
            self.train_data = test['data']
            self.train_ids = test['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or val')


        print(len(self.train_data))
        #print(self.label)
        attr_posneg = np.zeros((self.num_labels, 2), dtype=np.int)
        for index in range(0,len(self.train_data)):
            id = self.train_data[index][2]
            labels = np.asarray(self.train_attr[id])
            for label_id in range(0,self.num_labels):
                value = labels[label_id]
                if value == 1:
                    attr_posneg[label_id][0] = attr_posneg[label_id][0] + 1
                if value == 0:
                    attr_posneg[label_id][1] = attr_posneg[label_id][1] + 1
        rate_pos = attr_posneg[:,0] /(np.sum(attr_posneg,axis = 1))
        print(attr_posneg)
        self.weight_pos = 1 - rate_pos

        self.num_ids = len(self.train_ids)
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
        img_path = self.train_data[index][0]
        i = self.train_data[index][1]
        id = self.train_data[index][2]
        cam = self.train_data[index][3]
        label = np.asarray(self.train_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.train_data[index][4]
        #print(data, i, label, id, cam, name)
        return data,  label

    def __len__(self):
        return len(self.train_data)

    def num_label(self):
        return self.num_labels

    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label
    def weight(self):
        return self.weight_pos



class Test_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, transforms=None, query_gallery='query' ):
        if dataset_name == 'market':
            dataset_name = ['market']
        if dataset_name == 'all':
            dataset_name = ['market','duke']
        if dataset_name == 'duke':
            dataset_name = ['duke']
                
        train, val,test = import_MarketDuke_nodistractors(data_dir,dataset_name)
        

        self.train_attr, self.test_attr, self.label = import_MarketDuke_attr(data_dir,dataset_name)

        self.test_data = test['data']
        self.test_ids = test['ids']

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
        img_path = self.test_data[index][0]
        id = self.test_data[index][2]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.test_data[index][4]
        return data, label

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label