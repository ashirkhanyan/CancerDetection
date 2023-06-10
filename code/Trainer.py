
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class Trainer():

    def __init__(self, criterion, model, optimizer, dataloader: DataLoader) -> None:
        self.criterion = criterion
        self.model = model 
        self.optimizer = optimizer
        self.dataloader = dataloader


    def start_train(self, epochs):
        self.epochs = epochs
        start_time = datetime.now()
        for epoch in range(epochs):
            self.train(epoch)
            # self.validate()

    def train(self, epoch):
        for idx, (image, label, json_shape) in enumerate(self.dataloader):
            out = self.model(image)
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"{epoch+1}/{self.epochs}: {idx}/{len(self.dataloader)}: loss = {loss}")


    def validate(self):
        for idx, (image, label, json_shape) in enumerate(self.dataloader):
            with torch.no_grad():
                out = self.model(image)
                loss = self.criterion(out, label)
