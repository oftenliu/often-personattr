import os

# name                       idx  
# attachment-Backpack:1      88
# attachment-ShoulderBag:1   89
# attachment-HandBag:1       90
# attachment-WaistBag:1      91
# attachment-Box:1           92
# attachment-PlasticBag:1    93
# attachment-PaperBag:1      94
# attachment-HandTrunk:1     95
# attachment-Baby:1          96
# attachment-Other:1         97

def proc_attr(attrlist,combine_attachment_flag = False):
    attrlist_comebine = []
    if combine_attachment_flag:   
        for idx in range(0,len(attrlist)):
            if idx == 88: #and idx < 98:
                if attrlist[88] ==1 or attrlist[89] ==1 or attrlist[91] ==1:#Backpack  ShoulderBag   WaistBag ->  bag
                    attrlist_comebine.append(1)    
                else:
                     attrlist_comebine.append(0)

                if attrlist[90] ==1 or attrlist[93] ==1 or attrlist[94] ==1: #HandBag PlasticBag PaperBag -> 拎东西
                    attrlist_comebine.append(1)
                else:
                    attrlist_comebine.append(0)
                attrlist_comebine.append(attrlist[92])         #Box
                attrlist_comebine.append(attrlist[95])         #HandTrunk
                attrlist_comebine.append(attrlist[96])         #Baby
                #舍弃attachment-Other
                #attrlist_comebine.append(attrlist[97])        #Other
            elif idx > 88 and idx < 98:
                continue
            else:
                attrlist_comebine.append(attrlist[idx])
        return attrlist_comebine
    else:
        return attrlist
     

def proc_label(labels,combine_attachment_flag = False):
    labels_comebine = []
    if combine_attachment_flag:   
        for idx in range(0,len(labels)):
            if idx == 88: #and idx < 98:
                #Backpack  ShoulderBag   WaistBag ->  bag
                labels_comebine.append("attachment-Bag:1")    

                #HandBag PlasticBag PaperBag -> 拎东西
                labels_comebine.append("attachment-HandBag:1")
                
                labels_comebine.append(labels[92])         #Box
                labels_comebine.append(labels[95])         #HandTrunk
                labels_comebine.append(labels[96])         #Baby
                #舍弃attachment-Other
                #attrlist_comebine.append(attrlist[97])        #Other
            elif idx > 88 and idx < 98:
                continue
            else:
                labels_comebine.append(labels[idx])
        return labels_comebine
    else:
        return labels    

def import_rapdata(data_dir,datasets): 
    #load label
    train = {}
    val = {}
    test = {}
    train['data'] = []
    train['attr'] = []
    labelfile = os.path.join(data_dir,'rap_train.txt')
    ftrain = open(labelfile,'r')
    lines = ftrain.readlines()
    for line in lines:
        line = line.strip()
        image_file,attrs = line.split(':')
        train['data'].append(os.path.join(data_dir,'RAP_dataset',image_file))
        attrlist = attrs.split(' ')
        attrlist = [int(i) for i in attrlist]
        attrlist = proc_attr(attrlist,True)
        train['attr'].append(attrlist)

    ftrain.close()

    val['data'] = []
    val['attr'] = []
    labelfile = os.path.join(data_dir,'rap_val.txt')
    fval = open(labelfile,'r')
    lines = fval.readlines()
    for line in lines:
        line = line.strip()
        image_file,attrs = line.split(':')
        val['data'].append(os.path.join(data_dir,'RAP_dataset',image_file))
        attrlist = attrs.split(' ')
        attrlist = [int(i) for i in attrlist]
        attrlist = proc_attr(attrlist,True)
        val['attr'].append(attrlist)

    fval.close()    

    test['data'] = []
    test['attr'] = []
    labelfile = os.path.join(data_dir,'rap_test.txt')
    ftest = open(labelfile,'r')
    lines = ftest.readlines()
    for line in lines:
        line = line.strip()
        image_file,attrs = line.split(':')
        test['data'].append(os.path.join(data_dir,'RAP_dataset',image_file))
        attrlist = attrs.split(' ')
        attrlist = [int(i) for i in attrlist]
        attrlist = proc_attr(attrlist,True)
        test['attr'].append(attrlist)

    ftest.close() 

    labelname = []
    labelfile = os.path.join(data_dir,'labelname.txt')
    flabel = open(labelfile,'r')
    lines = flabel.readlines()
    for line in lines:
        line = line.strip()
        labelname.append(line)
    labelname = proc_label(labelname,True)
    flabel.close() 


    return train,val,test,labelname
if __name__ == '__main__':
    import_rapdata('/home/ulsee/often/lqc/Pedestrain attribute/often-attribute/dataset/rap2','')