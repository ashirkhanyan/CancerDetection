import os

SEED = 42
LOSS = "ce"  # ce [Cross Entropy], hb [Huberloss], l1 [MAE]
OPTIMIZER = "sgd" # sgd, adam
LEARNING_RATE = 0.1
MOMENTUM = 0.9
MODEL = "resnet" # resnet, densenet, mobilenet
BATCH_SIZE = 32
EPOCHS = 30

if os.environ.get('USERNAME') == 'srv':
    fld = "/Users/srv/Documents/Cloud"
else:
    fld = "/Users/aleks"

DATA_FOLDER = fld + "/Georgia Institute of Technology/MVP - General/Ultrasound-labeled"
PLOT_FOLDER = fld + "/Georgia Institute of Technology/MVP - General/plots"


TRAIN_PART = 0.9
PATIENCE = 2
REDUCE_FACTOR = 0.5
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 576
