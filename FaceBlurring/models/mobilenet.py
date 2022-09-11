import torch
import torch.nn as nn

class MobileNet (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=True)
        self.classifier = nn.Sequential(nn.Linear(1280, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 32),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
