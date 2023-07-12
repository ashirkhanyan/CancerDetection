
import torch
from torch.utils.data import DataLoader
from torch.nn.modules import Module
import matplotlib.pyplot as plt

import os
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from matplotlib import pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from config import *
import logging
import sys
from copy import deepcopy

class Trainer():

    def __init__(self, criterion, model: Module, optimizer: Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader = None, test_dataloader: DataLoader = None, device = torch.device("cpu"), logger: logging.Logger = None, save_path="./") -> None:
        self.criterion = criterion
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=REDUCE_FACTOR)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
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
        test_accs = []
        test_losses = []
        start_time = datetime.now()
        max_acc = 0
        self.logger.info(f"Training started at {start_time}")
        for epoch in range(epochs):
            train_acc, train_loss = self.train(epoch)
            val_acc, val_loss = self.validate(epoch)
            test_acc, test_loss = self.test(epoch)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
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
        accuracy_name = "Accuracy" if MODEL_TYPE == "classification" else "mAP"
        if plot:
            train_acc_plot_name = os.path.join(self.save_path, f"train_{accuracy_name.lower()}.png")
            train_loss_plot_name = os.path.join(self.save_path, "train_loss.png")
            val_acc_plot_name = os.path.join(self.save_path, f"val_{accuracy_name.lower()}.png")
            val_loss_plot_name = os.path.join(self.save_path, "val_loss.png")
            test_acc_plot_name = os.path.join(self.save_path, f"test_{accuracy_name.lower()}.png")
            test_loss_plot_name = os.path.join(self.save_path, "test_loss.png")
            plot_epochs = range(1, self.epochs+1)
            self.plot_graph(plot_epochs, train_accs, "Epochs", f"Train {accuracy_name}", f"{accuracy_name} Curve (Training)", f"{train_acc_plot_name}")
            self.plot_graph(plot_epochs, train_losses, "Epochs", "Train Loss", "Loss Curve (Training)", f"{train_loss_plot_name}")
            self.plot_graph(plot_epochs, val_accs, "Epochs", f"Validation {accuracy_name}", f"{accuracy_name} Curve (Validation)", f"{val_acc_plot_name}", color='orange')
            self.plot_graph(plot_epochs, val_losses, "Epochs", "Validation Loss", "Loss Curve (Validation)", f"{val_loss_plot_name}", color='orange')
            self.plot_graph(plot_epochs, test_accs, "Epochs", f"Test {accuracy_name}", f"{accuracy_name} Curve (Test)", f"{test_acc_plot_name}", color='black')
            self.plot_graph(plot_epochs, test_losses, "Epochs", "Test Loss", "Loss Curve (Test)", f"{test_loss_plot_name}", color='black')
            with open(os.path.join(self.save_path, "plot_values.py"), "w") as text_file:
                text_file.write(f"epochs={list(plot_epochs)}\n")
                text_file.write(f"train_{accuracy_name.lower()}={train_accs}\n")
                text_file.write(f"train_losses={train_losses}\n")
                text_file.write(f"val_{accuracy_name.lower()}={val_accs}\n")
                text_file.write(f"val_losses={val_losses}\n")
                text_file.write(f"test_{accuracy_name.lower()}={test_accs}\n")
                text_file.write(f"test_losses={test_losses}\n")
            self.logger.info(f"Plots are available at {self.save_path}\n")


    def train(self, epoch):
        self.logger.info("\n------------Training------------")
        self.map = MeanAveragePrecision(box_format=BOX_SHAPE)
        acc, loss = self.calculate_metrics(self.train_loader, epoch)
        return acc, loss


    @torch.no_grad()
    def validate(self, epoch):
        self.logger.info("\n------------Validation------------")
        self.map = MeanAveragePrecision(box_format=BOX_SHAPE)
        acc, loss = self.calculate_metrics(self.val_loader, epoch, mode="val")
        return acc, loss
    
    @torch.no_grad()
    def test(self, epoch):
        self.logger.info("\n------------Inference------------")
        self.map = MeanAveragePrecision(box_format=BOX_SHAPE)
        acc, loss = self.calculate_metrics(self.test_loader, epoch, mode="test")
        return acc, loss


    def calculate_metrics(self, data_loader, epoch, mode="train"):
        correct = 0
        samples = 0
        tot_loss = 0
        self.model = self.model.to(self.device)
        start_time = datetime.now()
        for idx, (image, label, json_shape) in enumerate(data_loader):
            if idx == SKIP_RUN_AFTER:
                break
            image = image.to(self.device)
            label = label.to(self.device)
            json_shape = json_shape.to(self.device)
            if MODEL_TYPE == "obj_detection":
                label_FRCNN = label + 1
                targets = []
                images = list(im.to(self.device) for im in image)
                
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = torch.unsqueeze(json_shape[i, :], 0).to(self.device)
                    d['labels'] = torch.unsqueeze(label_FRCNN[i], 0).to(self.device)
                    targets.append(d)
                losses = self.model(images, targets)
                loss = sum(l for l in losses.values())
                if MODEL == "fasterrcnn":
                    batch_loss_info = f"loss_classifier = {losses['loss_classifier']:.02f}, loss_box_reg = {losses['loss_box_reg']:.02f}, loss_objectness = {losses['loss_objectness']:.02f}, loss_rpn_box_reg = {losses['loss_rpn_box_reg']:.02f}, batch_{mode}_loss = {loss:.02f}"
                elif MODEL in ("ssd", "retinanet"):
                    batch_loss_info = f"loss_classifier = {losses['classification']:.02f}, loss_box_reg = {losses['bbox_regression']:.02f}, batch_{mode}_loss = {loss:.02f}"
            elif MODEL_TYPE == "classification":
                out = self.model(image)
                loss = self.criterion(out, label)
                batch_loss_info = f"batch_{mode}_loss = {loss:.02f}"

            if mode=="train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                if MODEL_TYPE == "obj_detection":
                    eval_model = deepcopy(self.model)
                    eval_model = eval_model.to(device=self.device)
                    eval_model.eval()
                    images = list(im.to(self.device) for im in image)
                    preds = eval_model(images)
                    self.map.update(preds, targets)
                    self.batch_map = MeanAveragePrecision(box_format=BOX_SHAPE)
                    self.batch_map.update(preds, targets)
                    map_vals = self.map.compute()
                    batch_map_vals = self.batch_map.compute()
                    tot_loss += loss
                    avg_loss = tot_loss/(idx+1)
                    batch_map = batch_map_vals['map_50']
                    avg_map = map_vals['map_50']
                    avg_acc = avg_map
                    batch_acc_info = f"avg_{mode}_loss = {avg_loss:.02f}, batch_{mode}_mAP = {batch_map:.02f}, avg_{mode}_mAP = {avg_map:.02f}"
                elif MODEL_TYPE == "classification":
                    acc = self.accuracy(out, label)
                    batch_acc = acc/image.shape[0]
                    correct += acc
                    samples += image.shape[0]
                    avg_acc = correct/samples
                    tot_loss += loss
                    avg_loss = tot_loss/(idx+1)
                    batch_acc_info = f"avg_{mode}_loss = {avg_loss:.02f}, batch_{mode}_acc = {batch_acc:.02f}, avg_{mode}_acc = {avg_acc:.02f}"
            if epoch == 0 and idx == 0 and mode=="train":
                print(f"Estimated ETA: {((datetime.now()-start_time).total_seconds()/3600) * len(data_loader.dataset)/BATCH_SIZE * EPOCHS} hours")
            self.logger.info(f"Epoch: {epoch+1:03d}/{self.epochs:03d}: Batch: {idx+1:03d}/{len(data_loader):03d}: {batch_loss_info}, {batch_acc_info}")
        return avg_acc.item(), avg_loss.item()


    @torch.no_grad()
    def accuracy(self, out, target):
        if MODEL == 'fasterrcnn':
            correct = out.eq(target).sum() * 1.0
        else:
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