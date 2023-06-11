
import torch
import torch.nn as nn


class DenseNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121')
        self.model.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=2, bias=True))

    
    def forward(self, x):
        out = self.model(x)
        return out
    