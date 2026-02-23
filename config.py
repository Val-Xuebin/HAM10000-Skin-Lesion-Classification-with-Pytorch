import os
import torch

# Data paths
DATA_DIR = 'input'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')

# Image preprocessing
INPUT_SIZE = 224
NORM_MEAN = [0.7630392, 0.5456477, 0.57004845]
NORM_STD = [0.1409286, 0.15261266, 0.16997074]

# Model config
MODEL_NAME = 'swin'  # resnet, vgg, densenet, inception, custom_cnn, swin
NUM_CLASSES = 7
FEATURE_EXTRACT = True # True: only train the last layer, False: train the whole model
USE_PRETRAINED = True

# Training config
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 2  # Reduced to avoid Bus error during long training

# Learning rate scheduler config
# Options: 'fixed', 'step', 'cosine', 'exponential', 'plateau'
LR_SCHEDULER = 'cosine'  # Default: cosine annealing

# Scheduler parameters (using dictionaries for better organization)
LR_SCHEDULER_PARAMS = {
    'step': {
        'step_size': 30,  # Decay learning rate every N epochs
        'gamma': 0.1  # Multiply learning rate by gamma
    },
    'cosine': {
        'eta_min': 1e-6  # Minimum learning rate for cosine annealing
    },
    'exponential': {
        'gamma': 0.95  # Gamma for exponential scheduler
    },
    'plateau': {
        'mode': 'max',  # 'max' for accuracy, 'min' for loss
        'factor': 0.5,  # Reduce lr by this factor
        'patience': 10  # Number of epochs with no improvement
    }
}

# Data split (70% train, 20% val, 10% test)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1
RANDOM_STATE = 101

# Class balancing
DATA_AUG_RATE = [15, 10, 5, 50, 0, 40, 5]

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Lesion type mapping
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

