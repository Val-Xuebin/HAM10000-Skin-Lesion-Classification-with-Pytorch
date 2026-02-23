# Skin Lesion Classification

Deep learning project for skin lesion classification using PyTorch and the HAM10000 dataset.

### Prerequisites

Python 3.8 or higher. A CUDA-capable GPU is optional but recommended for faster training.

### Installation

Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

### Data Download

The project downloads data from Kaggle on first run. Configure the Kaggle API: install the `kaggle` package and set credentials at `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`. You can use `./setup_kaggle.sh` to configure credentials interactively.

If automatic download fails, download from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 and extract into `input/` with:

```
input/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### Project Layout

- `config.py` — Data paths, model choice, training and scheduler settings.
- `main.py` — Entry point for training and evaluation.
- `src/data/` — Dataset and preprocessing (train/val/test split, balancing).
- `src/models/` — Model definitions (ResNet, VGG, DenseNet, Inception, custom CNN, Swin).
- `src/utils/` — Training utilities.
- `scripts/` — Experiment and visualization scripts (see below).

### Training

From the project root:

```
python main.py --mode train
```

This downloads data if needed, trains the model with the settings in `config.py`, logs to Weights & Biases (wandb), and saves the best checkpoint and training curves into a timestamped experiment directory under `results/`. Final evaluation on the held-out test set produces confusion matrices, ROC curves, and `test_metrics.json` in that directory.

### Evaluation

To evaluate the current best checkpoint on the validation set (and print metrics):

```
python main.py --mode eval
```

Evaluation uses the same split as training; test-set metrics are produced automatically at the end of `--mode train` and stored in the run’s experiment folder.

### Scripts

Run from the project root.

| Script | Purpose |
|--------|---------|
| `scripts/experiment.py` | Trains multiple models in sequence (e.g. swin, custom_cnn). Each run gets its own wandb run and results under `results/`. |
| `scripts/ablation_study.py` | Runs ablation experiments (e.g. baseline vs step/exponential/plateau LR). Outputs go to `ablation_results/` and a summary CSV/JSON. |
| `scripts/generate_visualizations.py` | Loads a saved checkpoint and regenerates confusion matrix and ROC curve plots. Edit the script to set model name and checkpoint path. |

Example:

```
python scripts/experiment.py
python scripts/ablation_study.py
python scripts/generate_visualizations.py
```

### Configuration

Edit `config.py` to change:

- **Model**: `MODEL_NAME` — `resnet`, `vgg`, `densenet`, `inception`, `custom_cnn`, or `swin`.
- **Training**: `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_WORKERS`.
- **Scheduler**: `LR_SCHEDULER` — `fixed`, `step`, `cosine`, `exponential`, or `plateau`; tune via `LR_SCHEDULER_PARAMS`.
- **Data split**: `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE` (e.g. 0.7 / 0.2 / 0.1).
- **Fine-tuning**: `FEATURE_EXTRACT = True` trains only the classifier head; `False` trains the full model.

### Output Files

- **Per run**: Under `results/<run_name>/` — `best_model.pth`, `training_curves.png`, test confusion matrices, ROC curves, and `test_metrics.json`. (These directories are gitignored.)
- **Single run (legacy)**: If not using experiment/script wrappers, `best_model.pth` and `training_curves.png` may be written in the project root; evaluation plots use a `test_` prefix when produced by the training pipeline.
