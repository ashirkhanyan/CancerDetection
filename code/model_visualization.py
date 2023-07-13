
import torch
from torch.utils.data import DataLoader
from UltrasoundDataset import UltrasoundDataset
from models import *
from utils.captum_utils import plot_boxes, get_iou, xywh_to_xyxy, compute_iou
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
        test_loader = DataLoader(test_dataset, batch_size=1)
        model_weights = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(model_weights)          

        model.eval()
        for idx, (image, label, json_shape) in enumerate(test_loader):
            if 'yolo' in model_weights: continue
            images = list(im.to(device) for im in image)
            with torch.no_grad():
                if label.item() == 0:
                    tumor_actual = "Actual_Benign"
                elif label.item() == 1:
                    tumor_actual = "Actual_Malignant"
                
                pred = model(images)
                out_label = pred[0]['labels']
                if not len(out_label) or out_label[0].item() == 0:
                    tumor_pred = "Predicted_Background"
                elif out_label[0].item() == 1:
                    tumor_pred = "Predicted_Benign"
                elif out_label[0].item() == 2:
                    tumor_pred = "Predicted_Malignant"
                #out_box = torch.stack([pred[i]['boxes'][0] for i in range(len(pred))])
                outbox = pred[0]['boxes']
                if not len(outbox):
                    continue
                if BOX_SHAPE == "xywh": 
                    outbox = xywh_to_xyxy(outbox)
                # first ones
                json, outbox, image = json_shape[0], outbox[0], images[0]
                iou = get_iou(json, outbox)
                print(iou)
                # count the different types of images
                if iou.item() < 0.5 and label.item() == 0:
                    detection_tumor = 'bad_detection_true_benign_'
                elif iou.item() < 0.5 and label.item() == 1:
                    detection_tumor = 'bad_detection_true_malignant_'
                elif iou.item() > 0.7 and label.item() == 0:
                    detection_tumor = 'good_detection_true_benign_'
                elif iou.item() > 0.7 and label.item() == 1:
                    detection_tumor = 'good_detection_true_malignant_'
   
                plot_boxes(json, outbox, image, tumor_actual, tumor_pred, save_path, detection_tumor)
                
                # this part is needed to stop the code from running when we have
                # enough images - we need 4 bad ones, 4 good ones
                # we did not use this part actually and we left the model to go over
                # all the images, so commenting them out for now
                #files = os.listdir(save_path)
                #files = [image for image in files if image.startswith('bad_detect') or image.startswith('good_detect')]
                #if len(files) == 8:
                #    print("We have enough images, aborting")
                #    break
