
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from Dataset import UltrasoundDataset
from models import *
from TrainerDetectionModels import Trainer

from config import *
import numpy as np
from captum.attr import LayerGradCam
from Visualizer import Visualizer
import os, sys
import logging

from Focal import FocalLoss
from captum_utils import plot_boxes


if __name__ == "__main__":

    torch.manual_seed(SEED)


    all_losses = {
        "ce": nn.CrossEntropyLoss(),
        "hb": nn.HuberLoss(),
        "l1": nn.L1Loss(),
        "fl": FocalLoss(),
    }
    try:
        criterion = all_losses[LOSS]
    except:
        print("Unknown Loss")



    all_models = {
        "resnet": ResNet(),
        "densenet": DenseNet(),
        "mobilenet": MobileNet(),
        "transformer": VisionTransformer(),   # Alik's edits
        "fasterrcnn": FasterRCNN(),           # Alik's edits
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


    all_activation_map = {
        "gdcam": LayerGradCam
    }
    try:
        activation_map = all_activation_map[VISUALIZER]
    except:
        print("Unknown Class Activation Map")

    if torch.has_mps:
        device = torch.device("cpu")
    elif torch.has_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    folders = os.listdir(PLOT_FOLDER)
    folders = [int(dir.split("_")[0]) for dir in folders if os.path.isdir(os.path.join(PLOT_FOLDER, dir))]
    run_name = 0 if not len(folders) else max(folders) + 1
    save_path = os.path.join(PLOT_FOLDER, str(run_name)+f"_{MODEL}_{MODEL_BACKBONE}_{LOSS}_{OPTIMIZER}_{LEARNING_RATE}_{REDUCE_FACTOR}_{EPOCHS}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving at {save_path}")

    logging.basicConfig(format=f"%(message)s", level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(save_path, "training.log"), mode="w"),
        logging.StreamHandler(sys.stdout)
    ])
    logger = logging.getLogger()
    with open("config.py", "r") as config_file:
        config = config_file.read()
    logger.info(f"\nUsing device: {device}")
    logger.info("\nTraining Config:")
    logger.info(config)
    

    train_dataset = UltrasoundDataset(DATA_FOLDER, box_shape=BOX_SHAPE)
    test_dataset = UltrasoundDataset(TEST_FOLDER, box_shape=BOX_SHAPE)

    split_at = int(len(train_dataset) * TRAIN_PART)
    idxs = np.array(range(len(train_dataset)))
    np.random.seed(SEED)
    np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:split_at], idxs[split_at:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    trainer = Trainer(criterion=criterion, model=model, optimizer=optimizer, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, device=device, logger=logger, save_path=save_path)

    trainer.start_train(epochs=EPOCHS, plot=True)


    if VIS_BATCH_SIZE:
        best_model_path = os.path.join(save_path, "best_model.pt")
        visual_save_path = os.path.join(save_path, "cam_visual")
        malignant_dataset = UltrasoundDataset(DATA_FOLDER, only_malignant=True)
        vis_loader = DataLoader(malignant_dataset, batch_size=VIS_BATCH_SIZE)
        os.makedirs(visual_save_path)
        visualizer = Visualizer(activation_map=activation_map, model=model, model_path=best_model_path, data_loader = vis_loader, device=device, logger=logger, save_path=visual_save_path)
        layer = 64
        visualizer.visualize(layer)
        
    if VIS_BOUND_BOX and MODEL == 'fasterrcnn':
        best_model_path = os.path.join(save_path, "best_model.pt")
        test_dataset = UltrasoundDataset(TEST_FOLDER)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model_weights = torch.load(best_model_path)
        model.load_state_dict(model_weights)

        model.eval()
        for idx, (image, label, json_shape) in enumerate(test_loader):
            images = list(im.to(device) for im in image)
            with torch.no_grad():
                pred = model(images)
                out_box = torch.stack([pred[i]['boxes'][0] for i in range(len(pred))])
                if idx < 20:
                    plot_boxes(json_shape, out_box, images, save_path, idx, BATCH_SIZE, ngraphs = 1)


