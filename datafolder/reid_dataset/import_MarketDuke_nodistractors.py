import os
from .reiddataset_downloader import *


def import_MarketDuke_nodistractors(data_dir,datasets):

    globals()['train']={}
    globals()['train']['data']=[]
    globals()['train']['ids'] = []

    globals()['query']={}
    globals()['query']['data']=[]
    globals()['query']['ids'] = []

    globals()['test']={}
    globals()['test']['data']=[]
    globals()['test']['ids'] = []


    data_group = ['train','query','test']
    for dataset_name in datasets:
        dataset_dir = os.path.join(data_dir,dataset_name)
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
                    #print(name)
                    cam = int(name.split('_')[1][1])
                    images = os.path.join(name_dir,name)
                    if (id!='0000' and id !='-1'):
                        id = dataset_name + '_' + id
                        if id not in globals()[group]['ids']:
                            globals()[group]['ids'].append(id)
                        globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, test



def import_MarketDuke_attr(data_dir,datasets): 
    globals()['train']={}

    globals()['test']={}
    #labels = get_attri_labels(data_dir)
    labels = []
    bFirst = True
    for  dataset_name in datasets:
        print(dataset_name)
        dataset_dir = os.path.join(data_dir,'all','anno')
        data_group = ['train','test']
        for group in data_group:
            if group == 'train':
                name_file = os.path.join(dataset_dir , dataset_name+'_train.txt')
            else:
                name_file = os.path.join(dataset_dir, dataset_name+'_test.txt')
            #file_list=sorted(os.listdir(name_dir))

            f_attr = open(name_file,'r')
            lines = f_attr.readlines() 
            for line in lines:
                line = line.strip()   
                id,attrstr = line.split(':')
                id = dataset_name + '_' + id
                attr_list = attrstr.split(',')
                #print(attr_list)
                #print(len(lines))
                id_attr = []
                #print(attr_list)
                for attr in attr_list:
                    attr = attr.strip()   
                    
                    label,value = attr.split(' ')
                    value = int(value)
                    id_attr.append(value)
                    if bFirst == True:
                        labels.append(label)
                bFirst = False
                #合并upbrown到 other
                if id_attr[16] == 1:
                    #print(labels[16] + "->" + labels[17])
                    id_attr[17] =1
                #合并 downpurple downyellow downgreen 到 other
                if id_attr[21] == 1 or id_attr[22] == 1 or id_attr[25] == 1:
                    #print(labels[21] + " or "+ labels[22] +" or "+labels[25]+ "->" + labels[27])
                    id_attr[27] =1
                if id_attr[0] == 0 and id_attr[1] ==0: #合并backpack and bag
                    id_attr[1] =0
                else:
                    id_attr[1] =1
                id_attr.pop(25)
                id_attr.pop(22)
                id_attr.pop(21)
                id_attr.pop(16)             
                id_attr.pop(0)  
                bFirst = False

                globals()[group][id] = id_attr
                    #globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    labels.pop(25)
    labels.pop(22)
    labels.pop(21)
    labels.pop(16)
    labels.pop(0)
    return train, test,labels


    # ['0:backpack 1', 'bag 0', 'handbag 0', 'down -1', 'up -1', 'hair -1', 'hat 0', '7:gender 0',
    # '8:upblack 1', 'upwhite 0', 'upred 0', 'uppurple 0', 'upyellow 0', 'upgray 0', 'upblue 0', 'upgreen 0', 'upbrown 0', '17:upothter 0', 
    # '18:downblack 0', 'downwhite 0', 'downred 0', 'downpurple 0', 'downyellow 0', 'downgray 0', 'downblue 0', 'downgreen 0', 'downbrown 1', '27:downothter 0']