
import torch
import torch.nn as nn


class DenseNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121')

    
    def forward(self, x):
        out = self.model(x)
        return out
    