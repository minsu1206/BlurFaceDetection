import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, size=18, num_classes=1):
        super().__init__()

        if size == 18:
            self.model = models.resnet18(pretrained=True)
        elif size == 34:
            self.model = models.resnet34(pretrained=True)
        elif size == 50:
            self.model = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError()

        if num_classes == 1:    # regression
            self.model.fc = nn.Sequential(
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
        else:   # classification
            self.model.fc = nn.Sequential(
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        return self.model(x)
