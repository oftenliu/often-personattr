import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from PIL import Image
from torchvision import transforms as T
from net import *
import os 

hooks = []
def visual_feature(features,layer_name):
    print(layer_name)
    output_dir = os.path.join('./output/feature',layer_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    features = features.squeeze(0)
    channel = features.size(0)

    for i in range(0,channel):
        feature=features[i,:,:]
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        
        filename = str(i) + '.jpg'
        filename =os.path.join(output_dir,filename)
        cv2.imwrite(filename,feature)


def register_hook(module):
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        visual_feature(output,class_name)

    if (
        isinstance(module, nn.Sequential)
        or isinstance(module, nn.ModuleList)
        #and not (module == model)
    ):
        hooks.append(module.register_forward_hook(hook))




class FeatureVisualization():
    def __init__(self,img_path,selected_layer):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.model = ResNet50_nFC_softmax(10,9,7)
        
        model_dir = os.path.join('./checkpoints', 'duke_finetune_handbag', 'resnet50_softmax')
        save_path = os.path.join(model_dir,'net_9.pth')
        print(save_path)
        self.model.load_state_dict(torch.load(save_path))
        #self.model.apply(register_hook)
        self.model.train(False) 
        print(self.model)
    def process_image(self):
        image = Image.open(self.img_path)
        image_tensor =  T.Compose([
                        T.Resize(size=(288, 144)),
                        #T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])(image)
        images = image_tensor.unsqueeze(0)
        return images
 
    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        x=input
        x = self.model.features.conv1(x)
        self.visual_feature(x,'1conv1')

        x = self.model.features.bn1(x)
        self.visual_feature(x,'2bn1')

        x = self.model.features.relu(x)
        self.visual_feature(x,'3relu')

        x = self.model.features.maxpool(x)
        self.visual_feature(x,'4maxpool')
        
        x = self.model.features.layer1(x)
        self.visual_feature(x,'5layer1')

        x = self.model.features.layer2(x)
        self.visual_feature(x,'6layer2')

        x = self.model.features.layer3(x)
        self.visual_feature(x,'7layer3')
        
        x = self.model.features.layer4(x)
        self.visual_feature(x,'8layer4')
        
        x = self.model.features.avgpool(x)
        self.visual_feature(x,'9avgpool')
        
        x = x.view(x.size(0), -1)
        x = self.model.features.fc(x)

 
    def visual_feature(self,features,layer_name):
        print(layer_name)
        output_dir = os.path.join('./output/feature',layer_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        features = features.squeeze(0)
        channel = features.size(0)

        for i in range(0,channel):
            feature=features[i,:,:]
            feature=feature.data.numpy()

            #use sigmod to [0,1]
            feature= 1.0/(1+np.exp(-1*feature))

            # to [0,255]
            feature=np.round(feature*255)
            
            filename = str(i) + '.jpg'
            filename =os.path.join(output_dir,filename)
            cv2.imwrite(filename,feature)


         

 
if __name__=='__main__':
    # get class
    myClass=FeatureVisualization('./test_image/201905281808320201.jpg',5)
    myClass.get_feature()
