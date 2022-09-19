import torch
import torch.nn as nn

def squeezenet1_1(num_classes=1):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Sigmoid())
    return model
    
