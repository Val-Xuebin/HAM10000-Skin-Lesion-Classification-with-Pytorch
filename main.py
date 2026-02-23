import argparse
import os
import json
import shutil
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ExponentialLR, 
    ReduceLROnPlateau
)
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import wandb
import config
from src.data.dataset import HAM10000, ensure_data_downloaded
from src.data.preprocessing import (
    load_image_paths, prepare_dataframe, identify_duplicates,
    split_train_val_test, balance_classes
)
from src.models.model import initialize_model
from src.utils.utils import AverageMeter

# Set random seeds for reproducibility
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)


def get_transforms(input_size, norm_mean, norm_std):
    """Get train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((int(input_size*1.2), int(input_size*1.2))),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15, fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    return train_transform, val_transform


def train_epoch(train_loader, model, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for images, labels in pbar:
        N = images.size(0)
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        prediction = outputs.max(1, keepdim=True)[1]
        batch_acc = prediction.eq(labels.view_as(prediction)).sum().item() / N
        train_acc.update(batch_acc)
        train_loss.update(loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{train_loss.avg:.4f}',
            'acc': f'{batch_acc:.4f}',
            'avg_acc': f'{train_acc.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion, epoch, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        for images, labels in pbar:
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            prediction = outputs.max(1, keepdim=True)[1]
            batch_acc = prediction.eq(labels.view_as(prediction)).sum().item() / N
            val_acc.update(batch_acc)
            val_loss.update(loss.item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{val_loss.avg:.4f}',
                'acc': f'{batch_acc:.4f}',
                'avg_acc': f'{val_acc.avg:.4f}'
            })

    return val_loss.avg, val_acc.avg


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(epochs) + 1])
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(epochs) + 1])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_path}')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_path='confusion_matrix.png', cmap=plt.cm.Blues):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to: {save_path}')


def plot_roc_curves(y_true, y_scores, class_names, save_path='roc_curves.png'):
    """Plot ROC curves for each class"""
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle='--', linewidth=2)
    
    # Plot ROC curve for each class
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC curves saved to: {save_path}')
    
    return roc_auc


def evaluate_model(val_loader, model, device, class_names, save_prefix=''):
    """Evaluate model and generate confusion matrix, ROC curves, and classification report"""
    model.eval()
    y_label = []
    y_predict = []
    y_scores = []  # Store probability scores for ROC curves
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = Variable(images).to(device)
            outputs = model(images)
            
            # Get predictions
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

            # Get probability scores (softmax)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_scores.extend(probs.cpu().numpy())

    y_label = np.array(y_label)
    y_predict = np.array(y_predict)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    confusion_mtx = confusion_matrix(y_label, y_predict)
    f1_scores = f1_score(y_label, y_predict, average=None)  # Per-class F1
    f1_macro = f1_score(y_label, y_predict, average='macro')  # Macro F1
    f1_weighted = f1_score(y_label, y_predict, average='weighted')  # Weighted F1
    
    # Plot confusion matrix
    cm_path = f'{save_prefix}confusion_matrix.png' if save_prefix else 'confusion_matrix.png'
    plot_confusion_matrix(confusion_mtx, class_names, save_path=cm_path)
    
    # Plot normalized confusion matrix
    cm_norm_path = f'{save_prefix}confusion_matrix_normalized.png' if save_prefix else 'confusion_matrix_normalized.png'
    plot_confusion_matrix(confusion_mtx, class_names, normalize=True, 
                         title='Normalized Confusion Matrix', save_path=cm_norm_path)
    
    # Plot ROC curves
    roc_path = f'{save_prefix}roc_curves.png' if save_prefix else 'roc_curves.png'
    roc_auc = plot_roc_curves(y_label, y_scores, class_names, save_path=roc_path)
    
    # Classification report
    report = classification_report(y_label, y_predict, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Print F1 scores
    print("\nF1 Scores:")
    print(f"  Macro F1: {f1_macro:.5f}")
    print(f"  Weighted F1: {f1_weighted:.5f}")
    print("  Per-class F1:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {f1_scores[i]:.5f}")
    
    # Print ROC AUC scores
    print("\nROC AUC Scores:")
    print(f"  Micro-average AUC: {roc_auc['micro']:.5f}")
    print("  Per-class AUC:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {roc_auc[i]:.5f}")
    
    return {
        'y_label': y_label,
        'y_predict': y_predict,
        'y_scores': y_scores,
        'confusion_matrix': confusion_mtx.tolist(),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': {class_names[i]: float(f1_scores[i]) for i in range(len(class_names))},
        'roc_auc_micro': float(roc_auc['micro']),
        'roc_auc_per_class': {class_names[i]: float(roc_auc[i]) for i in range(len(class_names))}
    }


def train():
    """Training function"""
    device = torch.device(config.DEVICE)
    print(f'Using device: {device}')
    
    data_dir = ensure_data_downloaded(config.DATA_DIR, force_download=False)
    
    print("Loading data...")
    imageid_path_dict = load_image_paths(data_dir)
    df_original = prepare_dataframe(
        config.METADATA_PATH,
        imageid_path_dict, 
        config.LESION_TYPE_DICT
    )
    
    df_original = identify_duplicates(df_original)
    df_train, df_val, df_test = split_train_val_test(df_original)
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
    
    # Balance classes only for training set
    df_train = balance_classes(df_train, config.DATA_AUG_RATE)
    print(f"Train samples after balancing: {len(df_train)}")
    
    model, input_size = initialize_model(
        config.MODEL_NAME,
        config.NUM_CLASSES,
        config.FEATURE_EXTRACT,
        config.USE_PRETRAINED
    )
    model = model.to(device)
    
    train_transform, val_transform = get_transforms(
        input_size, 
        config.NORM_MEAN,
        config.NORM_STD
    )
    
    train_dataset = HAM10000(df_train, transform=train_transform)
    val_dataset = HAM10000(df_val, transform=val_transform)
    test_dataset = HAM10000(df_test, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=False,  # Disable pin_memory to avoid Bus error
        persistent_workers=False if config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False if config.NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False if config.NUM_WORKERS > 0 else False
    )
    
    # Only optimize parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if len(params_to_optimize) == 0:
        raise ValueError("No parameters to optimize! Check feature_extract setting and model initialization.")
    optimizer = optim.Adam(params_to_optimize, lr=config.LEARNING_RATE)
    
    # Initialize learning rate scheduler based on config
    scheduler = None
    scheduler_type = config.LR_SCHEDULER.lower()
    
    if scheduler_type == 'fixed' or scheduler_type == 'none':
        # No scheduler - fixed learning rate
        scheduler = None
        print(f'Using fixed learning rate: {config.LEARNING_RATE}')
    elif scheduler_type == 'step':
        # Step decay: reduce LR by gamma every step_size epochs
        params = config.LR_SCHEDULER_PARAMS.get('step', {})
        scheduler = StepLR(optimizer, step_size=params.get('step_size', 30), gamma=params.get('gamma', 0.1))
        print(f'Using StepLR scheduler: step_size={params.get("step_size", 30)}, gamma={params.get("gamma", 0.1)}')
    elif scheduler_type == 'cosine':
        # Cosine annealing: cosine decay from initial LR to eta_min
        params = config.LR_SCHEDULER_PARAMS.get('cosine', {})
        scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=params.get('eta_min', 1e-6))
        print(f'Using CosineAnnealingLR scheduler: T_max={config.NUM_EPOCHS}, eta_min={params.get("eta_min", 1e-6)}')
    elif scheduler_type == 'exponential':
        # Exponential decay: multiply LR by gamma every epoch
        params = config.LR_SCHEDULER_PARAMS.get('exponential', {})
        scheduler = ExponentialLR(optimizer, gamma=params.get('gamma', 0.95))
        print(f'Using ExponentialLR scheduler: gamma={params.get("gamma", 0.95)}')
    elif scheduler_type == 'plateau':
        # Reduce on plateau: reduce LR when metric stops improving
        params = config.LR_SCHEDULER_PARAMS.get('plateau', {})
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode=params.get('mode', 'max'),
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 10),
            verbose=True
        )
        print(f'Using ReduceLROnPlateau scheduler: mode={params.get("mode", "max")}, factor={params.get("factor", 0.5)}, patience={params.get("patience", 10)}')
    else:
        print(f'Warning: Unknown scheduler type "{scheduler_type}", using fixed learning rate')
        scheduler = None
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    num_epochs = config.NUM_EPOCHS
    
    # Initialize wandb with unique run name for each model
    # Check for custom run name (used by ablation study)
    custom_run_name = os.environ.get('WANDB_RUN_NAME')
    wandb_tags = os.environ.get('WANDB_TAGS', '').split(',') if os.environ.get('WANDB_TAGS') else None
    wandb_tags = [tag.strip() for tag in wandb_tags if tag.strip()] if wandb_tags else None
    
    if custom_run_name:
        run_name = custom_run_name
    else:
        run_name = f"{config.MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    init_kwargs = {
        # Set the wandb entity where your project will be logged (generally your team name).
        "entity": "xuebin04",
        # Set the wandb project where this run will be logged.
        "project": "HAM10000-classification",
        # Set unique run name for this experiment
        "name": run_name,
        # Track hyperparameters and run metadata.
        "config": {
            "learning_rate": config.LEARNING_RATE,
            "architecture": config.MODEL_NAME,
            "dataset": "HAM10000",
            "epochs": num_epochs,
            "batch_size": config.BATCH_SIZE,
            "feature_extract": config.FEATURE_EXTRACT,
            "scheduler": config.LR_SCHEDULER,
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
            "data_augmentation": any(config.DATA_AUG_RATE) if hasattr(config, 'DATA_AUG_RATE') else None,
        },
    }
    
    if wandb_tags:
        init_kwargs["tags"] = wandb_tags
    
    run = wandb.init(**init_kwargs)
    
    # Create experiment directory in results/
    results_dir = 'results'
    exp_dir = os.path.join(results_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f'\nExperiment directory: {exp_dir}')
    
    print(f'\nStarting training for {num_epochs} epochs...\n')
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, epoch, device
        )
        val_loss, val_acc = validate_epoch(
            val_loader, model, criterion, epoch, device
        )
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate based on scheduler type
        if scheduler is not None:
            if config.LR_SCHEDULER.lower() == 'plateau':
                # Plateau scheduler needs the metric value
                scheduler.step(val_acc)
            else:
                # Other schedulers step every epoch
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'epoch': epoch
        })
        
        # Check if best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print epoch summary
        best_marker = ' (best)' if is_best else ''
        print(f'Epoch {epoch}/{num_epochs} [Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}{best_marker}]')
    
    # Plot training curves
    curves_path = os.path.join(exp_dir, 'training_curves.png')
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=curves_path)
    
    # Move best model to experiment directory
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists('best_model.pth'):
        shutil.move('best_model.pth', best_model_path)
        print(f'Best model saved to: {best_model_path}')
    
    # Load best model for test evaluation
    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded best model (val_acc: {best_val_acc:.5f}) for test evaluation')
    except FileNotFoundError:
        print('Warning: best_model.pth not found, using current model for test evaluation')
    
    # Evaluate on test set
    test_loss, test_acc = validate_epoch(
        test_loader, model, criterion, num_epochs + 1, device
    )
    print(f'Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.5f}')
    
    # Generate detailed evaluation metrics (confusion matrix, ROC curves, F1 scores)
    print('\n' + '='*60)
    print('Generating detailed evaluation metrics on test set...')
    print('='*60)
    
    # Save evaluation visualizations to experiment directory
    eval_prefix = os.path.join(exp_dir, 'test_')
    eval_results = evaluate_model(test_loader, model, device, config.CLASS_NAMES, save_prefix=eval_prefix)
    
    # Save test metrics to JSON file (not to wandb)
    metrics_file = os.path.join(exp_dir, 'test_metrics.json')
    test_metrics = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'best_val_acc': float(best_val_acc),
        'f1_macro': eval_results['f1_macro'],
        'f1_weighted': eval_results['f1_weighted'],
        'f1_per_class': eval_results['f1_per_class'],
        'roc_auc_micro': eval_results['roc_auc_micro'],
        'roc_auc_per_class': eval_results['roc_auc_per_class'],
        'wandb_run_name': run_name
    }
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f'\nTest metrics saved to: {metrics_file}')
    
    # Only log test_loss and test_acc to wandb (not F1 and ROC)
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })
    
    wandb.finish()
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.5f}')
    print(f'Test accuracy: {test_acc:.5f}')
    print(f'Test F1 (macro): {eval_results["f1_macro"]:.5f}')
    print(f'Test F1 (weighted): {eval_results["f1_weighted"]:.5f}')
    print(f'Test ROC AUC (micro): {eval_results["roc_auc_micro"]:.5f}')
    
    # Return key results for experiment script
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'final_train_acc': train_accs[-1] if train_accs else None,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_acc': val_accs[-1] if val_accs else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'test_f1_macro': eval_results['f1_macro'],
        'test_f1_weighted': eval_results['f1_weighted'],
        'test_f1_per_class': eval_results['f1_per_class'],
        'test_roc_auc_micro': eval_results['roc_auc_micro'],
        'test_roc_auc_per_class': eval_results['roc_auc_per_class'],
        'confusion_matrix_path': 'test_confusion_matrix.png',
        'confusion_matrix_normalized_path': 'test_confusion_matrix_normalized.png',
        'roc_curves_path': 'test_roc_curves.png',
    }


def eval():
    """Evaluation function"""
    device = torch.device(config.DEVICE)
    print(f'Using device: {device}')
    
    data_dir = ensure_data_downloaded(config.DATA_DIR, force_download=False)
    
    print("Loading data...")
    imageid_path_dict = load_image_paths(data_dir)
    df_original = prepare_dataframe(
        config.METADATA_PATH,
        imageid_path_dict, 
        config.LESION_TYPE_DICT
    )
    
    df_original = identify_duplicates(df_original)
    _, df_val, df_test = split_train_val_test(df_original)
    print(f"Validation samples: {len(df_val)}, Test samples: {len(df_test)}")
    
    model, input_size = initialize_model(
        config.MODEL_NAME,
        config.NUM_CLASSES,
        config.FEATURE_EXTRACT,
        config.USE_PRETRAINED
    )
    
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model weights")
    except FileNotFoundError:
        print("Warning: best_model.pth not found. Using untrained model.")
    
    model = model.to(device)
    
    _, val_transform = get_transforms(
        input_size, 
        config.NORM_MEAN,
        config.NORM_STD
    )
    
    val_dataset = HAM10000(df_val, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    evaluate_model(val_loader, model, device, config.CLASS_NAMES)


def main():
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], 
                       default='train', help='Mode: train or eval')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        eval()


if __name__ == '__main__':
    main()
