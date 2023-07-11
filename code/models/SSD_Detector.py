import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, ssd300_vgg16
from config import *
from Focal import FocalLoss

from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)


class SSD_Detector(nn.Module):
    def __init__(self):
        super().__init__()
        
        nclasses = 3 # Two labels + the background

        # SSD model with backbone resnet
        
        if MODEL_BACKBONE == "resnet":
            w, h = 256, 256                        # this is the target size
            #
            #print(self.model)
            resnet_bbone = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT
                )
            c1 = resnet_bbone.conv1
            bn1 = resnet_bbone.bn1
            relu = resnet_bbone.relu
            max_pool = resnet_bbone.maxpool
            lr1 = resnet_bbone.layer1
            lr2 = resnet_bbone.layer2
            lr3 = resnet_bbone.layer3
            lr4 = resnet_bbone.layer4
            backbone = nn.Sequential(
                c1, bn1, relu, max_pool, 
                lr1, lr2, lr3, lr4
                )
            out_channels = [512, 512, 512, 512, 512, 512]
            anchor_generator = DefaultBoxGenerator(
                [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                )
            nanchors = anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, nanchors, nclasses)
            self.model = SSD(backbone = backbone, num_classes=nclasses,        
                             anchor_generator=anchor_generator,
                             size=(w, h),
                             head=head)

        else:
            self.model = None
        

        if LOSS == "fl":
            self.classification_loss = FocalLoss()


    def forward(self, x, targets=None):
        if targets is not None:
            outputs = self.model(x, targets)
        else:
            outputs = self.model(x)
        return outputs
