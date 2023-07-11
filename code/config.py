from __hidden_config import *

# Config expects 4 values from hidden config: DATA_FOLDER (Training Data), TEST_FOLDER (Testing Data), PLOT_FOLDER (Folder for Training Output), BASE_FOLDER (Baseline info folder)

# Training Config
SEED = 42
LOSS = "fl"  # ce [Cross Entropy], hb [Huberloss], l1 [MAE]
OPTIMIZER = "sgd" # sgd, adam
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MODEL = "retinanet" # resnet, densenet, mobilenet, transformer, fasterrcnn, ssd, retinanet
MODEL_TYPE = "obj_detection"  # obj_detection, classification
MODEL_BACKBONE = "resnet"   # resnet, mobilenet
BOX_SHAPE = "xyxy" # xyxy, xywh
BATCH_SIZE = 8
EPOCHS = 20
TRAIN_PART = 0.9
PATIENCE = 1
REDUCE_FACTOR = 0.1
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 576
VISUALIZER = "gdcam" #GradCAM
CLASS_MAP = {
    "benign": 0,
    "malignant": 1,
}
INV_CLASS_MAP = {
    0: "benign",
    1: "malignant",
}
SKIP_RUN_AFTER = -1

# Visualization Config
VIS_MODEL_WEIGHTS = "1_resnet_ce_sgd_0.01_0.1_30"
VIS_MODEL = "fasterrcnn"
VIS_MODEL_BACKBONE = "resnet"
VIS_BATCH_SIZE = 0


# Evaluation Config
MODEL_WEIGHTS = {
    "fasterrcnn_mobilenet": {"folder": "0_cuda_10_fasterrcnn_mobilenet_fl_sgd_0.01_0.1_20", "type": "obj_detection", "backbone": "mobilenet"},
    "fasterrcnn_resnet": {"folder": "218_fasterrcnn_resnet_fl_sgd_0.01_0.1_20", "type": "obj_detection", "backbone": "resnet"},   # Change
    "ssd_resnet": {"folder": "0_cuda_7_ssd_resnet_fl_sgd_0.01_0.1_30", "type": "obj_detection", "backbone": "resnet"},
    "mobilenet": {"folder": "156_mobilenet__fl_sgd_0.01_0.1_20", "type": "classification"},
    "resnet": {"folder": "157_resnet__fl_sgd_0.01_0.1_20", "type": "classification"},
    "densenet": {"folder": "158_densenet__fl_sgd_0.01_0.1_20", "type": "classification"},
    "transformer": {"folder": "159_transformer__fl_sgd_0.01_0.1_20", "type": "classification"},
}