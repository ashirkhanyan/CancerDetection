#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import transforms
from PIL import Image
import os
import json
from torch.utils.data import Dataset, DataLoader

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, target_size):
        self.root_dir = root_dir
        self.image_paths, self.labels, self.json_data_shape, self.counter = self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        self.target_size = target_size

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

        resized_points = []
        for point in json_shape:
            x = int(point[0] * resize_ratio_x)
            y = int(point[1] * resize_ratio_y)
            resized_points.append([x, y])

        return resized_image, label, resized_points

    def _load_data(self):
        image_paths = []
        labels = []
        json_data_shape = []
        counter = 0
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(dirpath, filename)
                    image_paths.append(file_path)

                    if 'benign' in file_path:
                        label = 'benign'
                    else:
                        label = 'malignant'
                    labels.append(label)

                    json_path = os.path.splitext(file_path)[0] + '.json'
                    with open(json_path) as json_file:
                        data = json.load(json_file)
                        json_data_shape.append(data['shapes'][0]['points'])

        return image_paths, labels, json_data_shape, counter

root_dir = 'CSE6748/Ultrasound-labeled'
target_size = (256, 256)  # Specify the desired target size for resizing
dataset = UltrasoundDataset(root_dir, target_size)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Example usage:
for batch in dataloader:
    images, labels, json_shapes = batch
    # Do something with the batch of images, labels, and JSON shapes
    pass


# In[ ]:




