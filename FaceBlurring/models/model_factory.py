from operator import mod
import torch
from mobilenet import *
from resnet import *

def model_build(model_name:str, num_classes:int):
    """
    <args>
        model_name : from config/@.yaml
        resume:
            if '' : no pretrained model, no resume
            else : use .ckpt file and resume the model
    """

    model_name = model_name.lower()

    ######      build model     ######
    # define class or def for building model at each [@ Net].py
    # just add if ~ : ~ code for another model like below.


    if model_name == 'resnet18':
        model = ResNet(block=ResidualBlock, num_block=[2, 2, 2, 2], num_classes=num_classes)
    if model_name == 'resnet34':
        model = ResNet(block=ResidualBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)
    if model_name == 'resnet50':
        model = ResNet(block=BottleNeckResidualBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)


    return model







    return model
