
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from Dataset import UltrasoundDataset
from models import *
from Trainer import Trainer
from config import *
import numpy as np


if __name__ == "__main__":

    torch.manual_seed(SEED)


    all_losses = {
        "ce": nn.CrossEntropyLoss(),
        "hb": nn.HuberLoss(),
        "l1": nn.L1Loss(),
    }
    try:
        criterion = all_losses[LOSS]
    except:
        print("Unknown Loss")



    all_models = {
        "resnet": ResNet(),
        "densenet": DenseNet(),
        "mobilenet": MobileNet(),
    }
    try:
        model = all_models[MODEL]
    except:
        print("Unknown Model")



    all_optimizers = {
        "sgd": SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
        "adam": Adam(params=model.parameters(), lr=LEARNING_RATE),
    }
    try:
        optimizer = all_optimizers[OPTIMIZER]
    except:
        print("Unknown Optimizer")


    dataset = UltrasoundDataset(DATA_FOLDER)

    split_at = int(len(dataset) * TRAIN_PART)
    idxs = np.array(range(len(dataset)))
    np.random.seed(SEED)
    np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:split_at], idxs[split_at:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    trainer = Trainer(criterion=criterion, model=model, optimizer=optimizer, train_dataloader=train_loader, val_dataloader=val_loader)

    trainer.start_train(epochs=EPOCHS, plot=True)
