import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2


class RetinaNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        num_classes = 3   # two labels + the background
        if backbone == "resnet":
            self.model = retinanet_resnet50_fpn_v2(num_classes=3)
        else:
            self.model = None

    def forward(self, x, targets = None):
        if targets is not None:
            outputs = self.model(x, targets)
        else:
            outputs = self.model(x)
        return outputs