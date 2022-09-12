import torch
import torch.nn as nn
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile

class EfficientNetLite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()        
        weights_path = EfficientnetLite0ModelFile.get_model_file_path()
        self.model = EfficientNet.from_pretrained('efficientnet-lite0', weights_path=weights_path)
        self.model._fc = nn.Sequential(
            nn.Linear(1280, num_classes),
            nn.Sigmoid()
        )
        self.model._swish = nn.Sequential()

    def forward(self, x):
        return self.model(x)
