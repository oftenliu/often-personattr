import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)




# class_value = [1,5,1,1,1, 1, 1,1, 1,1, 1, 1,1,1, 1, 1, 1,1 ,1, 1, 1,1, 1, 1,1, 1,               
#                1, 1,1, 1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1,1, 1, 1,
#                1, 1, 1,1, 1, 1,1,
# ]

class ResNet50_nFC_softmax(nn.Module):
    def __init__(self,class_value,class_name,**kwargs):
        super(ResNet50_nFC_softmax, self).__init__()
        self.model_name = 'resnet50_nfc_softmax'
        # self.class_num = class_num
        # self.upcolor_num = upcolor_num
        # self.downcolor_num = downcolor_num
        self.class_value = class_value
        self.class_name = class_name
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512
        
        for c in  range(0,len(class_value)):
            #print(self.class_name[c])
            if class_value[c] != 1:
                if c == 1:
                    print(self.class_name[c])
                    self.__setattr__('class_%s' % self.class_name[c],
                    nn.Sequential(nn.Linear(self.num_ftrs,num_bottleneck), 
                                nn.BatchNorm1d(num_bottleneck),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(p=0.5),
                                nn.Linear(num_bottleneck, class_value[c]),
                                nn.Softmax()))    
                else:
                    print(self.class_name[c])
                    self.__setattr__('class_%s' % self.class_name[c],
                    nn.Sequential(nn.Linear(self.num_ftrs,num_bottleneck), 
                                nn.BatchNorm1d(num_bottleneck),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(p=0.5),
                                nn.Linear(num_bottleneck, class_value[c])))                      
            else:
                self.__setattr__('class_%s' % self.class_name[c],
                nn.Sequential(nn.Linear(self.num_ftrs,num_bottleneck), 
                              nn.BatchNorm1d(num_bottleneck),
                              nn.LeakyReLU(0.1),
                              nn.Dropout(p=0.5),
                              nn.Linear(num_bottleneck, 1)))      

    def forward(self, x):
        x = self.features(x)
        output = []
        for c in range(0,len(self.class_value)):
            output.append(self.__getattr__('class_%s' % self.class_name[c])(x))
        output.append(x)
        #print(len(class_value),len(output))
        return output
