from __hidden_config import *

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

VIS_BATCH_SIZE = 0
VIS_BOUND_BOX = True


EPOCHS = 20

VIS_MODEL_WEIGHTS = "1_resnet_ce_sgd_0.01_0.1_30"
VIS_MODEL = "resnet"

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