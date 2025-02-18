
import torch
from torch.utils.data import DataLoader
from torch.nn.modules import Module
import os
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from matplotlib import pyplot as plt
from config import *
import logging
import sys

class Trainer():

    def __init__(self, criterion, model: Module, optimizer: Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader = None, device = torch.device("cpu"), logger: logging.Logger = None, save_path="./") -> None:
        self.criterion = criterion
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=REDUCE_FACTOR)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device
        self.save_path = save_path
        if logger:
            self.logger = logger
        else:
            self.logger = logging.basicConfig(format=f"%(message)s", level=logging.INFO, handlers=[
                logging.FileHandler(os.path.join(self.save_path, "training.log"), mode="w"),
                logging.StreamHandler(sys.stdout)
            ])


    def start_train(self, epochs, plot=False):
        self.epochs = epochs
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []
        start_time = datetime.now()
        max_acc = 0
        self.logger.info(f"Training started at {start_time}")
        for epoch in range(epochs):
            train_acc, train_loss = self.train(epoch)
            val_acc, val_loss = self.validate(epoch)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if val_acc > max_acc:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
                max_acc = val_acc
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if self.lr != new_lr:
                self.logger.info(f"\nEpoch: {epoch+1:03d}/{self.epochs:03d}: Reduced LR from {self.lr:.07f} to {new_lr:.07f}")
                self.lr = new_lr
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "last_model.pt"))
        self.logger.info(f"\nTraining finished in at {datetime.now()}. Took {((datetime.now()-start_time).total_seconds())/60:.02f} minutes.")
        if plot:
            train_acc_plot_name = os.path.join(self.save_path, "train_accuracy.png")
            train_loss_plot_name = os.path.join(self.save_path, "train_loss.png")
            val_acc_plot_name = os.path.join(self.save_path, "val_accuracy.png")
            val_loss_plot_name = os.path.join(self.save_path, "val_loss.png")
            plot_epochs = range(1, self.epochs+1)
            self.plot_graph(plot_epochs, train_accs, "Epochs", "Train Accuracy", "Accuracy Curve (Training)", f"{train_acc_plot_name}")
            self.plot_graph(plot_epochs, train_losses, "Epochs", "Train Loss", "Loss Curve (Training)", f"{train_loss_plot_name}")
            self.plot_graph(plot_epochs, val_accs, "Epochs", "Validation Accuracy", "Accuracy Curve (Validation)", f"{val_acc_plot_name}", color='orange')
            self.plot_graph(plot_epochs, val_losses, "Epochs", "Validation Loss", "Loss Curve (Validation)", f"{val_loss_plot_name}", color='orange')
            with open(os.path.join(self.save_path, "plot_values.py"), "w") as text_file:
                text_file.write(f"epochs={list(plot_epochs)}\n")
                text_file.write(f"train_accs={train_accs}\n")
                text_file.write(f"train_losses={train_losses}\n")
                text_file.write(f"val_accs={val_accs}\n")
                text_file.write(f"val_losses={val_losses}\n")
            self.logger.info(f"Plots are available at {self.save_path}\n")
        


    def train(self, epoch):
        correct = 0
        samples = 0
        tot_loss = 0
        self.logger.info("\n------------Training------------")
        for idx, (image, label, json_shape) in enumerate(self.train_loader):
            self.model = self.model.to(self.device)
            image = image.to(self.device)
            label = label.to(self.device)
            out = self.model(image)
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            acc = self.accuracy(out, label)
            batch_acc = acc/image.shape[0]
            correct += acc
            samples += image.shape[0]
            avg_acc = correct/samples
            tot_loss += loss
            avg_loss = tot_loss/(idx+1)
            self.logger.info(f"Epoch: {epoch+1:03d}/{self.epochs:03d}: Batch: {idx+1:03d}/{len(self.train_loader):03d}: batch_train_loss = {loss:.05f}, avg_train_loss = {avg_loss:.05f}, batch_train_acc = {batch_acc:.05f}, avg_train_acc = {avg_acc:.05f}")
        return avg_acc.item(), avg_loss.item()
            

    def validate(self, epoch):
        correct = 0
        samples = 0
        tot_loss = 0
        self.logger.info("\n------------Validation------------")
        for idx, (image, label, json_shape) in enumerate(self.val_loader):
            with torch.no_grad():
                self.model = self.model.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device)
                out = self.model(image)
                loss = self.criterion(out, label)
            acc = self.accuracy(out, label)
            batch_acc = acc/image.shape[0]
            correct += acc
            samples += image.shape[0]
            avg_acc = correct/samples
            tot_loss += loss
            avg_loss = tot_loss/(idx+1)
            self.logger.info(f"Epoch: {epoch+1:03d}/{self.epochs:03d}: Batch: {idx+1:03d}/{len(self.val_loader):03d}: batch_val_loss = {loss:.05f}, avg_val_loss = {avg_loss:.05f}, batch_val_acc = {batch_acc:.05f}, avg_val_acc = {avg_acc:.05f}")
        return avg_acc.item(), avg_loss.item()


    def accuracy(self, out, target):
        _, pred = torch.max(out, dim=-1)
        correct = pred.eq(target).sum() * 1.0
        return correct
    

    def plot_graph(self, x_axis, y_axis, x_label, y_label, title, path, color="b", marker=None):
        fig, axis = plt.subplots()
        axis.plot(x_axis, y_axis, color=color, marker=marker)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        fig.savefig(path)
        plt.close()