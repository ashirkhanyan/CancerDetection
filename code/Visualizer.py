import os
import torch
from torch.utils.data import DataLoader
import logging
import sys
from models import *
from config import *
from tqdm import tqdm
from captum_utils import visualize_attr_maps



class Visualizer():

    def __init__(self, activation_map, model: torch.nn.Module, model_path, data_loader: DataLoader, device=torch.device("cpu"), logger: logging.Logger=None, save_path="./") -> None:
        self.activation_map = activation_map
        self.model = model
        model_weights = torch.load(model_path)
        self.model.load_state_dict(model_weights)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.save_path = save_path
        if logger:
            self.logger = logger
        else:
            self.logger = logging.basicConfig(format=f"%(message)s", level=logging.INFO, handlers=[
                logging.FileHandler(os.path.join(self.save_path, "training.log"), mode="w"),
                logging.StreamHandler(sys.stdout)
            ])
        for param in self.model.parameters():
            param.requires_grad = True

    
    def visualize(self, layer):
        for idx, module in enumerate(self.model.modules()):
            if idx == layer:
                model_layer = module
        activations = self.activation_map(self.model, model_layer)
        for idx, (image, label, json_shape) in enumerate(tqdm(self.data_loader)):
            image = image.to(self.device)
            label = label.to(self.device)
            # out = self.model(image)
            # max_out = torch.argmax(out, dim=1)
            # pred_label = max_out.cpu().detach().numpy()[0] - 1
            # cp_labels = torch.argmax(label, dim=0)[::self.data_loader.batch_size]
            attribution = activations.attribute(image, target = label)
            attr_mean = attribution.mean(axis=1, keepdim=True)
            image = image.permute(0, 2, 3, 1).cpu()
            # labels = torch.argmax(labels, dim=1) - 1
            # actual_label = labels.cpu().detach().numpy()[0]
            # attrs = attrs.permute(0, 2, 3, 1)
            # correctness = "correct" if actual_label == pred_label else "incorrect"
            # if CLASS_MAP[actual_label] == "Background":
            #     continue
            # else:
            #     halt=1
            # classif = "-".join(CLASS_MAP[actual_label].split(" ")) + "_" + correctness + "_" + "-".join(CLASS_MAP[pred_label].split(" "))
            visualize_attr_maps(os.path.join(self.save_path, f"{idx}.png"), image.cpu(), label.cpu(), INV_CLASS_MAP, [attr_mean.cpu()], [self.activation_map.get_name()])