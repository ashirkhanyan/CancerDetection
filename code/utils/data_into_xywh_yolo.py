
import os, json, shutil
import numpy as np
from PIL import Image

TRAIN_PART = 0.9

CLASS_MAP = {
    "benign": 0, "malignant": 1
}

def _load_data(root_dir):
        image_paths = []
        labels = []
        json_data_shape = []
        counter = 0
        malignant_paths = []
        benign_paths = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(dirpath, filename)
                    image_paths.append(file_path)
                    
                    if 'benign' in file_path:
                        benign_paths.append(file_path)
                        label = CLASS_MAP['benign']
                    else:
                        malignant_paths.append(file_path)
                        label = CLASS_MAP['malignant']
                    labels.append(label)
                    
                    json_path = os.path.splitext(file_path)[0] + '.json'
                    with open(json_path) as json_file:
                        data = json.load(json_file)
                        for dat in data['shapes']:
                            if dat['label'] == "tumor":
                                json_data_shape.append(dat['points'])
        
        
        return image_paths, labels, json_data_shape


def copy_file_in_structure(data_idxs, data_type):
    file_list = []
    for idx in data_idxs:
        file_name = image_paths[idx].split("/")[-1]
        shutil.copyfile(image_paths[idx], os.path.join(data_folder, "images", data_type, file_name))
        label_file_name = file_name[:-3]+"txt"
        image = Image.open(image_paths[idx])
        iw, ih = image.size
        [[x1, y1], [x2, y2]] = json_data_shape[idx]
        w = x2-x1
        h = y2-y1
        normx = (x1 + w/2)/iw
        normy = (y1 + h/2)/ih
        normw = w/iw
        normh = h/ih
        label_text = f"{labels[idx]} {normx} {normy} {normw} {normh}"
        with open(os.path.join(data_folder, "labels", data_type, label_file_name), mode="w") as label_file:
            label_file.write(label_text)
        file_list.append(f"./images/{data_type}/{file_name}\n")
    with open(os.path.join(data_folder, f"{data_type}.txt"), mode="w") as data_file:
        data_file.writelines(file_list)

if __name__ == "__main__":

    data_folder = "/home/saurav.suman/Documents/Git/data"
    shutil.rmtree(os.path.join(data_folder, "images"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_folder, "labels"), ignore_errors=True)
    os.makedirs(os.path.join(data_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "labels", "test"), exist_ok=True)

    image_paths, labels, json_data_shape = _load_data(os.path.join(data_folder, "train_data"))

    split_at = int(len(image_paths) * TRAIN_PART)
    idxs = np.array(range(len(image_paths)))
    np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:split_at], idxs[split_at:]

    copy_file_in_structure(train_idx, "train")
    copy_file_in_structure(val_idx, "val")

    image_paths, labels, json_data_shape = _load_data(os.path.join(data_folder, "test_data"))
    test_idxs = np.array(range(len(image_paths)))

    copy_file_in_structure(test_idxs, "test")