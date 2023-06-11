import torch
from torchvision import transforms
from PIL import Image
import os
import json
from torch.utils.data import Dataset
from config import IMAGE_WIDTH, IMAGE_HEIGHT

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.type_map = {
            "malignant": 0,
            "benign": 1,
        }
        self.image_paths, self.labels, self.json_data_shape, self.counter = self._load_data()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        json_shape = self.json_data_shape[index]

        image = self.transform(Image.open(image_path))

        return image, label, json_shape

    def _load_data(self):
        image_paths = []
        labels = []
        json_data_shape = []
        counter = 0
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(dirpath, filename)
                    image_open = Image.open(file_path)
                    width, height = image_open.size
                    # if width == IMAGE_WIDTH and height == IMAGE_HEIGHT:
                    image_paths.append(file_path)
                    # else:
                    #     counter += 1
                    cancer_type = file_path.split(os.path.sep)[-3]
                    # if 'benign' in file_path:
                    #     label = 'benign'
                    # else:
                    #     label = 'malignant'
                    labels.append(self.type_map[cancer_type])

                    json_path = os.path.splitext(file_path)[0] + '.json'
                    with open(json_path) as json_file:
                        data = json.load(json_file)
                        json_data_shape.append(data['shapes'][0]['points'])

        return image_paths, labels, json_data_shape, counter
