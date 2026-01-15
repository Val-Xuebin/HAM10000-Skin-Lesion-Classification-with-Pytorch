
# Skin Lesion Classification

Deep learning project for skin lesion classification using PyTorch and HAM10000 dataset.

### Prerequisites

Python 3.13.3 is included in venv/. CUDA-capable GPU is optional but recommended for faster training.

### Installation

Activate the virtual environment:

```
source venv/bin/activate
```

Dependencies should already be installed in venv. If needed, install with:

```
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm pillow kaggle
```

### Data Download

The project automatically downloads data from Kaggle on first run. For API download, install kaggle and configure credentials at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables).

If automatic download fails, manually download from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 and extract to the input/ directory with the following structure:

```
input/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### Training

Run training with:

```
python main.py --mode train
```

The process downloads data if needed, trains the model, saves the best model to best_model.pth, and generates training curves plot.

### Evaluation

Run evaluation with:

```
python main.py --mode eval
```

This loads best_model.pth, evaluates on the validation set, and generates a confusion matrix and classification report.

### Configuration

Edit config.py to adjust model architecture (resnet-50, vgg, densenet, inception), training hyperparameters (NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE), and class balancing (DATA_AUG_RATE). Set FEATURE_EXTRACT to True for fine-tuning only the last layer, or False for full model training.

## Output Files

- best_model.pth: Best model weights based on validation accuracy
- training_curves.png: Training and validation loss/accuracy curves
- confusion_matrix.png: Confusion matrix visualization
