import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from config import *
from Focal import FocalLoss


class FasterRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        num_classes = 3   # two labels + the background
        if MODEL_BACKBONE == "resnet":
            self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif MODEL_BACKBONE == "mobilenet":
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        elif MODEL_BACKBONE == "vit":
            self.model = None
        
        if LOSS == "fl":
            self.classification_loss = FocalLoss()

        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, targets = None):
        if targets is not None:
            outputs = self.model(x, targets)
        else:
            outputs = self.model(x)
        return outputs