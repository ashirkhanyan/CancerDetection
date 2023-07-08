import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(x,target)
        # get the softmax
        smax = x.softmax(dim=1)
        # select the corresponding probabilities with one_hot
        oh = F.one_hot(target, num_classes=2).bool()
        pt = smax[oh]
        # calculate alpha
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        # calculate loss
        focalloss = alpha * (1-pt)**self.gamma * CE_loss
        return focalloss.mean()

