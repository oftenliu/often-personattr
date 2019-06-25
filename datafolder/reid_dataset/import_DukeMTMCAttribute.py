import os
from .reiddataset_downloader import *
from .import_DukeMTMC import *
import scipy.io
        
def import_DukeMTMCAttribute(dataset_dir):
    dataset_name = 'DukeMTMC-reID/attribute'
    train,query,test = import_DukeMTMC(dataset_dir)
    if not os.path.exists(os.path.join(dataset_dir,dataset_name)):
        print('Please Download the DukeMTMCATTributes Dataset')
    train_label = ['backpack',
                   'bag',
                   'handbag',
                   'boots',
                   'gender',
                   'hat',
                   'shoes',
                   'top',
                   'downblack',
                   'downwhite',
                   'downred',
                   'downgray',
                   'downblue',
                   'downgreen',
                   'downbrown',
                   'upblack',
                   'upwhite',
                   'upred',
                   'uppurple',
                   'upgray',
                   'upblue',
                   'upgreen',
                   'upbrown']
    
    test_label=['boots',
                'shoes',
                'top',
                'gender',
                'hat',
                'backpack',
                'bag',
                'handbag',
                'downblack',
                'downwhite',
                'downred',
                'downgray',
                'downblue',
                'downgreen',
                'downbrown',
                'upblack',
                'upwhite',
                'upred',
                'upgray',
                'upblue',
                'upgreen',
                'uppurple',
                'upbrown']
    
    
    train_person_id = []
    for personid in train:
        train_person_id.append(personid)
    train_person_id.sort(key=int)

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    
    f = scipy.io.loadmat(os.path.join(dataset_dir,dataset_name,'duke_attribute.mat'))

    test_attribute = {}
    train_attribute = {}
    for test_train in range(len(f['duke_attribute'][0][0])):
        if test_train == 1:
            id_list_name = 'test_person_id'
            group_name = 'test_attribute'
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'
        for attribute_id in range(len(f['duke_attribute'][0][0][test_train][0][0])):
            if isinstance(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
                continue
            for person_id in range(len(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0])):
                id = locals()[id_list_name][person_id]
                if id not in locals()[group_name]:
                    locals()[group_name][id]=[]
                locals()[group_name][id].append(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])
    
    for i in range(8):
        train_label.insert(8,train_label[-1])
        train_label.pop(-1)
    
    unified_train_atr = {}
    for k,v in train_attribute.items():
        temp_atr = list(v)
        for i in range(8):
            temp_atr.insert(8,temp_atr[-1])
            temp_atr.pop(-1)
        unified_train_atr[k] = temp_atr
    
    unified_test_atr = {}
    for k,v in test_attribute.items():
        temp_atr = [0]*len(train_label)
        for i in range(len(train_label)):
            temp_atr[i]=v[test_label.index(train_label[i])]
        unified_test_atr[k] = temp_atr
    #two zero appear in train '0370' '0679'
    #zero_check=[]
    #for id in train_attribute:
    #    if 0 in train_attribute[id]:
    #        zero_check.append(id)
    #for i in range(len(zero_check)):
    #    train_attribute[zero_check[i]] = [1 if x==0 else x for x in train_attribute[zero_check[i]]]
    unified_train_atr['0370'][7]=1
    unified_train_atr['0679'][7]=2
    print(train_label)
    return unified_train_atr,unified_test_atr,train_label

# def import_DukeMTMCAttribute_binary(dataset_dir):
#     train_duke_attr, test_duke_attr,label = import_DukeMTMCAttribute(dataset_dir)
#     for id in train_duke_attr:
#         train_duke_attr[id][:] = [x - 1 for x in train_duke_attr[id]]
#     for id in test_duke_attr:
# 	    test_duke_attr[id][:] = [x - 1 for x in test_duke_attr[id]]
#     label[7] = 'up'

#     temp = label[4]
#     label[4] = label[7]
#     label[7] = temp
#     for personid in train_duke_attr:
#         filename = './dataset/DukeMTMC-reID/anno/train/' + str(personid) + '.txt'
#         f_train = open(filename,'w')
#         tempattr = train_duke_attr[personid][4]
#         train_duke_attr[personid][4] = -1
#         train_duke_attr[personid][7] = tempattr  
#         attrs = []  
#         for index in range(0,len(train_duke_attr[personid])): 
#             if index not in {3,6}:      
#                 attrs.append(label[index] + ' '+ str(train_duke_attr[personid][index]))     
#             if index == 3:

#                 f_train.write('down'+":"+str(-1) + "\n")   
#             if index == 4:
#                 f_train.write('hair'+":"+str(-1) + "\n")                                              
#             if index == 11:
#                 f_train.write('upyellow'+":"+str(0) + "\n")
#             if index == 18:
#                 f_train.write('downpurple'+":"+str(0) + "\n")      
#                 f_train.write('downyellow'+":"+str(0) + "\n")               
#         f_train.close()
#     for personid in test_duke_attr:
#         filename = './dataset/DukeMTMC-reID/anno/test/' + str(personid) + '.txt'
#         f_test = open(filename,'w')
#         tempattr = test_duke_attr[personid][4]
#         test_duke_attr[personid][4] = -1
#         test_duke_attr[personid][7] = tempattr         
#         for index in range(0,len(test_duke_attr[personid])):            
#             if index not in {3,6}:               
#                 f_test.write(label[index]+":"+str(test_duke_attr[personid][index]) + "\n")
#             if index == 3:
#                 f_test.write('down'+":"+str(-1) + "\n")   
#             if index == 4:
#                 f_test.write('hair'+":"+str(-1) + "\n")                      
#             if index == 11:
#                 f_test.write('upyellow'+":"+str(0) + "\n")
#             if index == 18:
#                 f_test.write('downpurple'+":"+str(0) + "\n")      
#                 f_test.write('downyellow'+":"+str(0) + "\n")   
#         f_test.close()

#     return train_duke_attr, test_duke_attr, label

def import_DukeMTMCAttribute_binary(dataset_dir):
    train_duke_attr, test_duke_attr,label = import_DukeMTMCAttribute(dataset_dir)
    for id in train_duke_attr:
        train_duke_attr[id][:] = [x - 1 for x in train_duke_attr[id]]
    for id in test_duke_attr:
	    test_duke_attr[id][:] = [x - 1 for x in test_duke_attr[id]]

    label[7] = 'up'
    temp = label[4]
    label[4] = label[7]
    label[7] = temp

    filename = './dataset/duke/anno/train.txt'
    f_train = open(filename,'w') 
    for personid in train_duke_attr:    
        train_duke_attr[personid][7] = train_duke_attr[personid][4]
        train_duke_attr[personid][4] = -1

        attrs = []
        upcolor = 0
        downcolor = 0
        for index in range(0,len(train_duke_attr[personid])): 
            if index not in {3,6}:      
                attrs.append(label[index] + ' '+ str(train_duke_attr[personid][index]))
            if index > 7 and index < 16:
                upcolor = upcolor + train_duke_attr[personid][index]
            if index > 15:
                downcolor = downcolor + train_duke_attr[personid][index]
        attrs.insert(3,'down -1')
        attrs.insert(5,'hair -1')
        attrs.insert(12,'upyellow 0')

        attrs.insert(20,'downpurple 0')      
        attrs.insert(21,'downyellow 0')  
        if upcolor == 0:
            attrs.insert(17,'upothter 1')
        else:
            attrs.insert(17,'upothter 0')

        if downcolor == 0:
            attrs.append('downothter 1')
        else:
            attrs.append('downothter 0')
        strvalues = attrs[0]
        for ind in range(1,len(attrs)):
            strvalues = strvalues + ','
            strvalues = strvalues + attrs[ind]
        f_train.write(personid+":"+strvalues + "\n")        
    f_train.close()

    filename = './dataset/duke/anno/test.txt'
    f_test = open(filename,'w')  
    for personid in test_duke_attr:    
        test_duke_attr[personid][7] = test_duke_attr[personid][4]
        test_duke_attr[personid][4] = -1

        attrs = []
        upcolor = 0
        downcolor = 0
        for index in range(0,len(test_duke_attr[personid])): 
            if index not in {3,6}:      
                attrs.append(label[index] + ' '+ str(test_duke_attr[personid][index]))
            if index > 7 and index < 16:
                upcolor = upcolor + test_duke_attr[personid][index]
            if index > 15:
                downcolor = downcolor + test_duke_attr[personid][index]
        attrs.insert(3,'down -1')
        attrs.insert(5,'hair -1')
        attrs.insert(12,'upyellow 0')

        attrs.insert(20,'downpurple 0')      
        attrs.insert(21,'downyellow 0')  
        if upcolor == 0:
            attrs.insert(17,'upothter 1')
        else:
            attrs.insert(17,'upothter 0')

        if downcolor == 0:
            attrs.append('downothter 1')
        else:
            attrs.append('downothter 0')
        strvalues = attrs[0]
        for ind in range(1,len(attrs)):
            strvalues = strvalues + ','
            strvalues = strvalues + attrs[ind]
        f_test.write(personid+":"+strvalues + "\n")        
    f_test.close()

    filename = './dataset/duke/anno/label.txt'
    f_label = open(filename,'w')
    for inx in range(0,len(label)):  
        if inx not in {3,6}:
            f_label.write(label[inx]+"\n")

    print(label)
    return train_duke_attr, test_duke_attr, label