
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import UltrasoundDataset
from models.ResNet18 import ResNet
from Trainer import Trainer
from config import *


if __name__ == "__main__":
    if LOSS == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown Loss")

    if MODEL == "resnet":
        model = ResNet()
    else:
        raise ValueError("Unknown Model")


    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError("Unknown Optimizer")


    dataset = UltrasoundDataset(ROOT_DIR)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    trainer = Trainer(criterion=criterion, model=model, optimizer=optimizer, dataloader=dataloader)

    trainer.start_train(epochs=EPOCHS)
