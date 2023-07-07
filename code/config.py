import getpass

SEED = 42
LOSS = "ce"  # ce [Cross Entropy], hb [Huberloss], l1 [MAE]
OPTIMIZER = "sgd" # sgd, adam
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MODEL = "fasterrcnn" # resnet, densenet, mobilenet, transformer, fasterrcnn
BATCH_SIZE = 4
VIS_BATCH_SIZE = 0
EPOCHS = 1

if getpass.getuser() == 'srv':
    fld = "/Users/srv/Documents/Cloud"
else:
    fld = "/Users/aleks"

DATA_FOLDER = fld + "/Georgia Institute of Technology/MVP - General/Ultrasound-labeled"
PLOT_FOLDER = fld + "/Georgia Institute of Technology/MVP - General/plots"


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
SKIP_RUN_AFTER = 6