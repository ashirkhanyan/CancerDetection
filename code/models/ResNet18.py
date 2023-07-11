import torch
import torch.nn as nn

class BBock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsampling: nn.Module = None):
        super(BBock, self).__init__()
        
        # the paper has downsample if stride != 1
        self.downsampling = downsampling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # for skip connections 
        resid = x

        result = self.conv1(x)
        result = self.bn1(result)
        result = self.relu(result)
        result = self.conv2(result)
        result = self.bn2(result)
        
        if self.downsampling is not None:
            resid = self.downsampling(x)

        result += resid
        result = self.relu(result)
        return  result

class ResNet(nn.Module):

    def __init__(self, image_channels=3, block=BBock, n_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=self.in_channels, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # basic blocks part
        self.layer1 = self._create_layers(block, 64)
        self.layer2 = self._create_layers(block, 128, stride=2)
        self.layer3 = self._create_layers(block, 256, stride=2)
        self.layer4 = self._create_layers(block, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def _create_layers(self, block, out_channels, stride = 1):
        downsampling = None
        if stride != 1:
            downsampling = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        
        # append the two repeating layers - if stride = 2, downsample the first one
        repeatlayers = []
        repeatlayers.append(block(self.in_channels, out_channels, stride, downsampling))
        self.in_channels = out_channels
        
        # append the second time 
        repeatlayers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*repeatlayers)

    def forward(self, x):
        # first part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # repeating layer part
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # last part
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
