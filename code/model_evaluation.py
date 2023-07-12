
import torch
from torch.utils.data import DataLoader
from UltrasoundDataset import UltrasoundDataset
from models import *
import json
from tqdm import tqdm
from config import *
from collections import defaultdict
import os

if __name__ == "__main__":

    all_models = {
        "resnet": ResNet(),
        "densenet": DenseNet(),
        "mobilenet": MobileNet(),
        "transformer": VisionTransformer(),
        "fasterrcnn_mobilenet": FasterRCNN(backbone="mobilenet"),
        "fasterrcnn_resnet": FasterRCNN(backbone="resnet"),
        "ssd_resnet": SSD_Detector(backbone="resnet"),
        "retinanet": RetinaNet(backbone="resnet"),
    }

    device = torch.device("cpu")
    
    save_path = os.path.sep.join(PLOT_FOLDER.split(os.path.sep)[:-1])
    os.makedirs(save_path, exist_ok=True)
    json_file = os.path.join(save_path, "confusion_data.json")
    print(f"Saving as {json_file}")

    all_data = {}

    with torch.no_grad():
        test_dataset = UltrasoundDataset(TEST_FOLDER)
        test_loader = DataLoader(test_dataset, batch_size=1)
        for model_weight in MODEL_WEIGHTS:
            if 'yolo' in model_weight: continue
            metrics = defaultdict(int)
            model = all_models[model_weight]
            weight_folder = MODEL_WEIGHTS[model_weight]['folder']
            model_type = MODEL_WEIGHTS[model_weight]['type']
            weight_file = os.path.join(PLOT_FOLDER, weight_folder, "best_model.pt")
            weight_tensor = torch.load(weight_file, map_location=device)
            model.load_state_dict(weight_tensor)
            model.eval()
            for image, label, json_shape in tqdm(test_loader, desc=model_weight):
                if label.item() == 0:
                    key = "true_benign_"
                elif label.item() == 1:
                    key = "true_malignant_"
                out = model(image)
                if model_type == "obj_detection":
                    out_label = out[0]['labels']
                    if not len(out_label) or out_label[0].item() == 0:
                        key += "pred_background"
                    elif out_label[0].item() == 1:
                        key += "pred_benign"
                    elif out_label[0].item() == 2:
                        key += "pred_malignant"
                elif model_type == "classification":
                    if out.argmax().item() == 0:
                        key += "pred_benign"
                    elif out.argmax().item() == 1:
                        key += "pred_malignant"
                metrics[key] += 1
            all_data[model_weight] = dict(metrics)
        json.dump(all_data, open(json_file, mode="w"))