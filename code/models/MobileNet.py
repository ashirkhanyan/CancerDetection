
import torch
import torch.nn as nn

class MobileNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False), nn.Linear(in_features=1280, out_features=2, bias=True))
    

    def forward(self, x):
        out = self.model(x)
        return out