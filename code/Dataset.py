

import torch
from torchvision import transforms
from PIL import Image
import os
import json
from torch.utils.data import Dataset
import random
from config import *

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256), only_malignant=False, box_shape = "xyxy"):
        self.root_dir = root_dir
        self.only_malignant = only_malignant
        self.image_paths, self.labels, self.json_data_shape, self.counter = self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        self.target_size = target_size
        self.box_shape = box_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        json_shape = self.json_data_shape[index]

        image = Image.open(image_path)
        resized_image = self.transform(image)

        # Resize the position points of the tumors
        original_width, original_height = image.size
        resized_width, resized_height = self.target_size
        resize_ratio_x = resized_width / original_width
        resize_ratio_y = resized_height / original_height

        ratios = torch.as_tensor([resize_ratio_x, resize_ratio_y], dtype=torch.float32)
        resized_points = torch.as_tensor(json_shape, dtype=torch.float32)*ratios
        

        if self.box_shape == "xywh":
            center_points = torch.sum(resized_points, dim=0)/2
            width_height = resized_points[1] - resized_points[0]
            final_coords = torch.cat((center_points, width_height))
        elif self.box_shape == "xyxy":
            final_coords = resized_points.view(4)

        return resized_image, label, final_coords

    def _load_data(self):
        image_paths = []
        labels = []
        json_data_shape = []
        counter = 0
        malignant_paths = []
        benign_paths = []
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            if self.only_malignant and "malignant" not in dirpath:
                continue
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(dirpath, filename)
                    image_paths.append(file_path)
                    
                    if 'benign' in file_path:
                        benign_paths.append(file_path)
                        label = CLASS_MAP['benign']
                    else:
                        malignant_paths.append(file_path)
                        label = CLASS_MAP['malignant']
                    labels.append(label)
                    
                    json_path = os.path.splitext(file_path)[0] + '.json'
                    with open(json_path) as json_file:
                        data = json.load(json_file)
                        json_data_shape.append(data['shapes'][0]['points'])

        if not self.only_malignant:
            # Perform random oversampling on the minority class (malignant)
            oversampled_malignant_paths = random.choices(malignant_paths, k=len(benign_paths))
            image_paths.extend(oversampled_malignant_paths)
            labels.extend([1] * len(oversampled_malignant_paths))
            json_data_shape.extend([json_data_shape[malignant_paths.index(path)] for path in oversampled_malignant_paths])
            
            # Randomly undersample the majority class (benign)
            undersampled_benign_indices = random.sample(range(len(benign_paths)), len(malignant_paths))
            undersampled_benign_paths = [benign_paths[i] for i in undersampled_benign_indices]
            image_paths.extend(undersampled_benign_paths)
            labels.extend([0] * len(undersampled_benign_paths))
            json_data_shape.extend([json_data_shape[benign_paths.index(path)] for path in undersampled_benign_paths])
        
        # Update the counter
        counter = len(image_paths) - len(benign_paths) - len(malignant_paths)
        
        return image_paths, labels, json_data_shape, counter

#root_dir = 'CSE6748/Ultrasound-labeled'
#dataset = UltrasoundDataset(root_dir)

#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Compute the new distribution
#total_count = len(dataset)
#benign_count = sum(label == 0 for label in dataset.labels)
#malignant_count = sum(label == 1 for label in dataset.labels)
#benign_percentage = benign_count / total_count * 100
#malignant_percentage = malignant_count / total_count * 100

#print(f"Total Images: {total_count}")
#print(f"Benign Images: {benign_count} ({benign_percentage:.2f}%)")
#print(f"Malignant Images: {malignant_count} ({malignant_percentage:.2f}%)")
