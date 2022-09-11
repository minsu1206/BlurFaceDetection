import torch
import torch.nn as nn

class MobileNet (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True)
        self.classifier = nn.Sequential(nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
