import os

# Data paths
DATA_DIR = 'input'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')

# Image preprocessing
INPUT_SIZE = 224
NORM_MEAN = [0.7630392, 0.5456477, 0.57004845]
NORM_STD = [0.1409286, 0.15261266, 0.16997074]

# Model config
MODEL_NAME = 'resnet'  # resnet, vgg, densenet, inception
NUM_CLASSES = 7
FEATURE_EXTRACT = True # True: only train the last layer, False: train the whole model
USE_PRETRAINED = True

# Training config
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

# Data split
VAL_SIZE = 0.2
RANDOM_STATE = 101

# Class balancing
DATA_AUG_RATE = [15, 10, 5, 50, 0, 40, 5]

# Device
DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

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

