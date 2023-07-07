
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
from copy import deepcopy

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
        train_ious = []
        val_accs = []
        val_losses = []
        val_ious = []
        start_time = datetime.now()
        max_acc = 0
        self.logger.info(f"Training started at {start_time}")
        for epoch in range(epochs):
            train_acc, train_loss, train_iou = self.train(epoch)
            val_acc, val_loss, val_iou = self.validate(epoch)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            train_ious.append(train_iou)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            val_ious.append(val_iou)
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
            train_iou_plot_name = os.path.join(self.save_path, "train_iou.png")
            val_acc_plot_name = os.path.join(self.save_path, "val_accuracy.png")
            val_loss_plot_name = os.path.join(self.save_path, "val_loss.png")
            val_iou_plot_name = os.path.join(self.save_path, "val_iou.png")
            plot_epochs = range(1, self.epochs+1)
            self.plot_graph(plot_epochs, train_accs, "Epochs", "Train Accuracy", "Accuracy Curve (Training)", f"{train_acc_plot_name}")
            self.plot_graph(plot_epochs, train_losses, "Epochs", "Train Loss", "Loss Curve (Training)", f"{train_loss_plot_name}")
            self.plot_graph(plot_epochs, train_ious, "Epochs", "Train IOU", "IOU Curve (Training)", f"{train_iou_plot_name}")
            self.plot_graph(plot_epochs, val_accs, "Epochs", "Validation Accuracy", "Accuracy Curve (Validation)", f"{val_acc_plot_name}", color='orange')
            self.plot_graph(plot_epochs, val_losses, "Epochs", "Validation Loss", "Loss Curve (Validation)", f"{val_loss_plot_name}", color='orange')
            self.plot_graph(plot_epochs, val_ious, "Epochs", "Validation IOU", "IOU Curve (Validation)", f"{val_iou_plot_name}", color='orange')
            with open(os.path.join(self.save_path, "plot_values.py"), "w") as text_file:
                text_file.write(f"epochs={list(plot_epochs)}\n")
                text_file.write(f"train_accs={train_accs}\n")
                text_file.write(f"train_losses={train_losses}\n")
                text_file.write(f"train_ious={train_ious}\n")
                text_file.write(f"val_accs={val_accs}\n")
                text_file.write(f"val_losses={val_losses}\n")
                text_file.write(f"val_ious={val_ious}\n")
            self.logger.info(f"Plots are available at {self.save_path}\n")


    def train(self, epoch):
        self.logger.info("\n------------Training------------")
        acc, loss, iou = self.calculate_metrics(self.train_loader, epoch)
        return acc, loss, iou


    @torch.no_grad()
    def validate(self, epoch):
        self.logger.info("\n------------Validation------------")
        acc, loss, iou = self.calculate_metrics(self.val_loader, epoch, train=False)
        return acc, loss, iou


    def calculate_metrics(self, data_loader, epoch, train=True):
        correct = 0
        samples = 0
        tot_loss = 0
        tot_iou = 0
        mode = "train" if train else "val"
        self.model = self.model.to(self.device)
        
        for idx, (image, label, json_shape) in enumerate(data_loader):
            if idx == SKIP_RUN_AFTER:
                break
            image = image.to(self.device)
            label = label.to(self.device)
            if MODEL == 'fasterrcnn':                 # FROM here on - ALik's edits
                label_FRCNN = label + 1                # 0 is considered as background
                targets = []                       # Alik's edits
                images = list(im.to(self.device) for im in image)
                
                # create the targets accordingly to feed into the Faster R-CNN
                for i in range(len(images)):
                    d = {}
                    d['boxes'] = torch.unsqueeze(json_shape[i, :], 0).to(self.device)
                    d['labels'] = torch.unsqueeze(label_FRCNN[i], 0).to(self.device)
                    targets.append(d)
                losses = self.model(images, targets)
                loss = sum(l for l in losses.values())
            else:
                out = self.model(image)
                loss = self.criterion(out, label)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if MODEL == 'fasterrcnn':
                eval_model = deepcopy(self.model)
                eval_model = eval_model.to(device=self.device)
                eval_model.eval()
                images = list(im.to(self.device) for im in image)
                pred = eval_model(images)
                try:
                    out = [pred[i]['labels'][0] for i in range(len(pred))]
                    out_box = torch.stack([pred[i]['boxes'][0] for i in range(len(pred))])
                    out = torch.stack(out)
                    out = out - 1 # return back to the 0 and 1 labels
                    batch_iou = self.calc_iou(json_shape, out_box)
                    tot_iou += batch_iou
                    avg_iou = tot_iou/(idx+1)
                    acc = self.accuracy(out, label)
                    batch_acc = acc/image.shape[0]
                    correct += acc
                    samples += image.shape[0]
                    avg_acc = correct/samples
                    tot_loss += loss
                    avg_loss = tot_loss/(idx+1)
                    batch_log_info = f"avg_{mode}_loss = {avg_loss:.02f}, batch_{mode}_acc = {batch_acc:.02f}, avg_{mode}_acc = {avg_acc:.02f}, batch_{mode}_iou = {batch_iou:.02f}, avg_{mode}_iou = {avg_iou:.02f}"
                except:
                    batch_log_info = "No Boxes Detected by model to compare!"
            self.logger.info(f"Epoch: {epoch+1:03d}/{self.epochs:03d}: Batch: {idx+1:03d}/{len(data_loader):03d}: loss_classifier = {losses['loss_classifier']:.02f}, loss_box_reg = {losses['loss_box_reg']:.02f}, loss_objectness = {losses['loss_objectness']:.02f}, loss_rpn_box_reg = {losses['loss_rpn_box_reg']:.02f}, batch_{mode}_loss = {loss:.02f}, {batch_log_info}")
        return avg_acc.item(), avg_loss.item(), avg_iou.item()


    @torch.no_grad()
    def accuracy(self, out, target):
        if MODEL == 'fasterrcnn':
            correct = out.eq(target).sum() * 1.0
        else:
            _, pred = torch.max(out, dim=-1)
            correct = pred.eq(target).sum() * 1.0
        return correct


    @torch.no_grad()
    def calc_iou(self, json_shape, out_box):
        upper_left_points_act = json_shape[:, 0:2]
        lower_right_points_act = json_shape[:, 2:4]
        upper_left_points_pred = out_box[:, 0:2]
        lower_right_points_pred = out_box[:, 2:4]
        upper_left_intersection = torch.maximum(upper_left_points_act, upper_left_points_pred)
        lower_right_intersection = torch.minimum(lower_right_points_act, lower_right_points_pred)
        intersection_tensor_points = torch.cat((upper_left_intersection, lower_right_intersection), axis=1)
        union_area = self.area_from_corner_points(json_shape) + self.area_from_corner_points(out_box)
        intersection_area = self.area_from_corner_points(intersection_tensor_points)
        iou = intersection_area/union_area
        batch_avg_iou = torch.mean(iou)
        return batch_avg_iou
    
    @torch.no_grad()
    def area_from_corner_points(self, tensor_points):
        points = tensor_points.view(BATCH_SIZE, 2, -1)
        sub = torch.abs(points[:, 0] - points[:, 1])
        areas = sub[:, 0] * sub[:, 1]
        return areas
    

    def plot_graph(self, x_axis, y_axis, x_label, y_label, title, path, color="b", marker=None):
        fig, axis = plt.subplots()
        axis.plot(x_axis, y_axis, color=color, marker=marker)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        fig.savefig(path)
        plt.close()