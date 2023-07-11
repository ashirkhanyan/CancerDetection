from __hidden_config import *

# Config expects 4 values from hidden config: DATA_FOLDER (Training Data), TEST_FOLDER (Testing Data), PLOT_FOLDER (Folder for Training Output), BASE_FOLDER (Baseline info folder)

# Training Config
SEED = 42
LOSS = "fl"  # ce [Cross Entropy], hb [Huberloss], l1 [MAE]
OPTIMIZER = "sgd" # sgd, adam
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MODEL = "fasterrcnn" # resnet, densenet, mobilenet, transformer, fasterrcnn, ssd
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