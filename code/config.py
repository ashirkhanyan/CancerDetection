
SEED = 42
LOSS = "ce"  # ce [Cross Entropy], hb [Huberloss], l1 [MAE]
OPTIMIZER = "sgd" # sgd, adam
LEARNING_RATE = 0.1
MOMENTUM = 0.9
MODEL = "mobilenet" # resnet, densenet, mobilenet
BATCH_SIZE = 32
EPOCHS = 70
DATA_FOLDER = "/Users/srv/Documents/Cloud/Georgia Institute of Technology/MVP - General/Ultrasound-labeled"
PLOT_FOLDER = "/Users/srv/Documents/Cloud/Georgia Institute of Technology/MVP - General/plots"
TRAIN_PART = 0.9
PATIENCE = 2
REDUCE_FACTOR = 0.5
