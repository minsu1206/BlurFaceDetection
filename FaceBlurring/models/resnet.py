import torch
import torch.nn as nn
import torch.nn.functional as F

class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(ShortCut, self).__init__()
        if stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.shortcut(x)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.residual_network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * ResidualBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
        )

        self.shortcut = ShortCut(in_channels, out_channels, stride, ResidualBlock.expansion)

    def forward(self, x):
        x = self.residual_network(x) + self.shortcut(x)
        x = F.relu(x)
        return x


class BottleNeckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeckResidualBlock, self).__init__()
        self.residual_network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeckResidualBlock.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeckResidualBlock.expansion),
        )

        self.shortcut = ShortCut(in_channels, out_channels, stride, BottleNeckResidualBlock.expansion)

    def forward(self, x):
        x = self.shortcut(x) + self.residual_network(x)
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self.build_layer(block, 64, num_block[0], 1)
        self.conv3 = self.build_layer(block, 128, num_block[1], 2)
        self.conv4 = self.build_layer(block, 256, num_block[2], 2)
        self.conv5 = self.build_layer(block, 512, num_block[3], 2)
        # building regressor
        self.regressor = nn.Sequential(nn.Linear(512 * block.expansion, 100),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Linear(100, 1))
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.mean(3).mean(2)  # Global average pooling
        x = torch.sigmoid(self.regressor(x))
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def build_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)