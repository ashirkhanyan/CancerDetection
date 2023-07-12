
import torch
from torch.utils.data import DataLoader
from UltrasoundDataset import UltrasoundDataset
from models import *
from utils.captum_utils import plot_boxes
from config import *
from captum.attr import LayerGradCam
from Visualizer import Visualizer
import os

if __name__ == "__main__":

    torch.manual_seed(SEED)


    all_models = {
        "resnet": ResNet(),
        "densenet": DenseNet(),
        "mobilenet": MobileNet(),
        "transformer": VisionTransformer(),
        "fasterrcnn": FasterRCNN(backbone=VIS_MODEL_BACKBONE),
        "ssd": SSD_Detector(backbone=VIS_MODEL_BACKBONE),
    }
    try:
        model = all_models[VIS_MODEL]
    except:
        print("Unknown Model")


    all_activation_map = {
        "gdcam": LayerGradCam
    }
    try:
        activation_map = all_activation_map[VISUALIZER]
    except:
        print("Unknown Class Activation Map")

    if torch.has_mps:
        device = torch.device("mps")
    elif torch.has_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    
    folders = os.listdir(PLOT_FOLDER)
    folders = [int(dir.split("_")[0]) for dir in folders if os.path.isdir(os.path.join(PLOT_FOLDER, dir))]
    run_name = 0 if not len(folders) else max(folders) + 1
    save_path = os.path.join(PLOT_FOLDER, VIS_MODEL_WEIGHTS)
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving at {save_path}")
    
    if MODEL_TYPE == "classification":
        best_model_path = os.path.join(save_path, "best_model.pt")
        visual_save_path = os.path.join(save_path, "cam_visual")
        malignant_dataset = UltrasoundDataset(DATA_FOLDER, only_malignant=True)
        vis_loader = DataLoader(malignant_dataset, batch_size=VIS_BATCH_SIZE)
        os.makedirs(visual_save_path, exist_ok=True)
        visualizer = Visualizer(activation_map=activation_map, model=model, model_path=best_model_path, data_loader = vis_loader, device=device, save_path=visual_save_path)
        layer = model.layer4[-1]
        visualizer.visualize(layer)

    elif MODEL_TYPE == "obj_detection":
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
                try:
                    out_box = torch.stack([pred[i]['boxes'][0] for i in range(len(pred))])
                except:
                    print("No box found, skipping")
                if idx > 30 and idx < 60:
                    plot_boxes(json_shape, out_box, images, save_path, idx, BATCH_SIZE, ngraphs = 1)
