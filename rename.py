import os
import time
import argparse
import scipy.io

def traverse(f,imageset):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            ext = os.path.splitext(tmp_path)[-1][1:]
            if ext == "jpg":
                #print('文件: %s'%tmp_path)
                imageset.append(tmp_path)
        else:
            #print('文件夹：%s'%tmp_path)
            traverse(tmp_path,imageset)
path = '/home/ulsee/temp'
image_datasets = []
traverse(path,image_datasets)

ids = []
train_image = {}
oldname = '0516'
newname = '1520'
for image_file in image_datasets:
    id = image_file.split('/')[-1].split('_')[0]
    if id == oldname:
        newfilename = image_file.replace(oldname,newname)
        os.rename(image_file, newfilename)