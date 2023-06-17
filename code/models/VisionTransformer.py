import torch
import torch.nn as nn
from torchvision.transforms import Resize
from transformers import DeiTModel

class VisionTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.resize = Resize((self.model.config.image_size, self.model.config.image_size))
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)



    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, height * width)  # Flatten height and width dimensions
        x = x.transpose(1, 2)  # Swap dimensions to match DeiT model's input expectations
        outputs = self.model(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits



# =============================================================================
#     def forward(self, x):
#         x = self.resize(x)
#         outputs = self.model(x)
#         logits = self.classifier(outputs.last_hidden_state[:, 0])
#         return logits
# =============================================================================
