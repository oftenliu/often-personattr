import os


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
        test['attr'].append(attrlist)

    ftest.close() 

    labelname = []
    labelfile = os.path.join(data_dir,'labelname.txt')
    flabel = open(labelfile,'r')
    lines = flabel.readlines()
    for line in lines:
        line = line.strip()
        labelname.append(line)

    flabel.close() 


    return train,val,test,labelname
if __name__ == '__main__':
    import_rapdata('/home/ulsee/often/lqc/Pedestrain attribute/often-attribute/dataset/rap2','')