import os
import torch
from torch.utils.data import DataLoader
from config import *
from tqdm import tqdm
from captum_utils import visualize_attr_maps



class Visualizer():

    def __init__(self, activation_map, model: torch.nn.Module, model_path, data_loader: DataLoader, device=torch.device("cpu"), save_path="./") -> None:
        self.activation_map = activation_map
        self.model = model
        model_weights = torch.load(model_path)
        self.model.load_state_dict(model_weights)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.save_path = save_path
        for param in self.model.parameters():
            param.requires_grad = True

    
    def visualize(self, layer):
        activations = self.activation_map(self.model, layer)
        for idx, (image, label, json_shape) in enumerate(tqdm(self.data_loader)):
            image = image.to(self.device)
            label = label.to(self.device)
            attribution = activations.attribute(image, target = label)
            attr_mean = attribution.mean(axis=1, keepdim=True)
            image = image.permute(0, 2, 3, 1).cpu()
            visualize_attr_maps(os.path.join(self.save_path, f"{idx}.png"), image.cpu(), label.cpu(), INV_CLASS_MAP, [attr_mean.cpu()], [self.activation_map.get_name()])