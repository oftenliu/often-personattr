from .ResNet18_nFC import ResNet18_nFC
from .ResNet34_nFC import ResNet34_nFC
from .ResNet50_nFC import ResNet50_nFC
from .ResNet101_nFC import ResNet101_nFC
from .DenseNet121_nFC import DenseNet121_nFC
from .ResNet50_nFC_softmax import ResNet50_nFC_softmax
from .loss import PersonAttr_Loss
__all__ = [
    'ResNet101_nFC',    
    'ResNet50_nFC',
    'DenseNet121_nFC',
    'ResNet34_nFC',
    'ResNet18_nFC',
    'ResNet50_nFC_softmax',
    'PersonAttr_Loss',
]

