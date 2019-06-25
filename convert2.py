import io
import numpy as np

from torch import nn
import torch.onnx

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from torch.autograd import Variable

import SRmodel

class Arguments():
    def __init__(self):
        self.nChannel = 3
        self.nDenselayer = 6
        self.nFeat = 64
        self.scale = 3
        self.growthRate = 32        
        
#------------------------------------------------------------
# convert Pytorch model to ONNX (Caffe2)
#------------------------------------------------------------
# load Pytorch model
stateDictPath = "./test/model_lastest.pt"

args = Arguments()

torchModel = getattr(SRmodel, 'RDN_Tiny')(args)
torchModel.load_state_dict(torch.load(stateDictPath))
torchModel.train(False)
#torchModel.eval()

#print(torchModel)

# input to the model
#input_size = (1, 3, 112, 112)
input_size = (1, 12, 112, 112)
image = np.random.randint(0, 255, input_size)
input_data = image.astype(np.float32)
input_var = Variable(torch.from_numpy(input_data))

# export the model
input_names = [ "data" ]
output_names = [ "ConvNdBackward219" ]

print(input_var.shape)

#out = torchModel(input_var)

torchOut = torch.onnx.export(torchModel, input_var, "./test/model.onnx", verbose=True, input_names=input_names, output_names=output_names)
#torch.onnx.export(torchModel, input_var, "./test/model.onnx", verbose=True)

#torchOut = torch.onnx._export(torchModel, input_var, "./test/model.onnx", export_params=True)

#torchOut = torch.onnx._export(torchModel, input_var, "./test/model.onnx")
