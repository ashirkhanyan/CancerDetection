import torch
import torch.nn as nn
import torchvision
from functools import partial
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class RetinaNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        if backbone == "resnet":
            self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head.classification_head = RetinaNetClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=3, norm_layer=partial(torch.nn.GroupNorm, 32))
        else:
            self.model = None

    def forward(self, x, targets = None):
        if targets is not None:
            outputs = self.model(x, targets)
        else:
            outputs = self.model(x)
        return outputs