import os
from .reiddataset_downloader import *


def import_MarketDuke_nodistractors(data_dir,dataset_name):

    datasets = dataset_name
    globals()['train']={}
    globals()['train']['data']=[]
    globals()['train']['ids'] = []


    globals()['query']={}
    globals()['query']['data']=[]
    globals()['query']['ids'] = []

    globals()['test']={}
    globals()['test']['data']=[]
    globals()['test']['ids'] = []

    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','test']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir, 'query')
        else:
            name_dir = os.path.join(dataset_dir, 'bounding_box_test')
        file_list=sorted(os.listdir(name_dir))

        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0]
                cam = int(name.split('_')[1][1])
                images = os.path.join(name_dir,name)
                if (id!='0000' and id !='-1'):
                    id = dataset_name + '_' + id
                    if id not in globals()[group]['ids']:
                        globals()[group]['ids'].append(id)
                    globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, test


def get_attri_labels(data_dir):
    labeltxt = os.path.join(data_dir,'label.txt')
    f_attr = open(labeltxt,'r')
    lines = f_attr.readlines() 
    labels = []
    for index in range(0,26):
        line = lines[index].strip()   
        label,value = line.split(':')
        labels.append(label)
    return labels

    
def import_MarketDuke_attr(data_dir,dataset_name):
 
    globals()['train']={}

    globals()['test']={}
    labels = get_attri_labels(data_dir)

    for  dataset_name in datasets:
        print(dataset_name)
        dataset_dir = os.path.join(data_dir,dataset_name,'anno')
        data_group = ['train','test']
        for group in data_group:
            if group == 'train':
                name_dir = os.path.join(dataset_dir , 'train')
            else:
                name_dir = os.path.join(dataset_dir, 'test')
            file_list=sorted(os.listdir(name_dir))

            for name in file_list:
                if name[-3:]=='txt':
                    #print(name)
                    id = name.split('.')[0]
                    id = dataset_name + '_' + id
                    txtfile = os.path.join(name_dir, name)
                    f_attr = open(txtfile,'r')
                    lines = f_attr.readlines() 
                    #print(len(lines))
                    id_attr = []
                    for index in range(0,26):
                        line = lines[index].strip()   
                        label,value = line.split(':')
                        value = int(value)
                        id_attr.append(value)

                    globals()[group][id] = id_attr
                    #globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, test,labels