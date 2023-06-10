
import torch
import torch.nn as nn

class MobileNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
    

    def forward(self, x):
        out = self.model(x)
        return out