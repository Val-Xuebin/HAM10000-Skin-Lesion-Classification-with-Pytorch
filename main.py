import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import config
from src.data.dataset import HAM10000, ensure_data_downloaded
from src.data.preprocessing import (
    load_image_paths, prepare_dataframe, identify_duplicates,
    split_train_val, balance_classes
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
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        
        pbar.set_postfix({
            'loss': f'{train_loss.avg:.5f}',
            'acc': f'{train_acc.avg:.5f}'
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
            prediction = outputs.max(1, keepdim=True)[1]
            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
            val_loss.update(criterion(outputs, labels).item())
            
            pbar.set_postfix({
                'loss': f'{val_loss.avg:.5f}',
                'acc': f'{val_acc.avg:.5f}'
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()


def evaluate_model(val_loader, model, device, class_names):
    """Evaluate model and generate confusion matrix and classification report"""
    model.eval()
    y_label = []
    y_predict = []
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    confusion_mtx = confusion_matrix(y_label, y_predict)
    plot_confusion_matrix(confusion_mtx, class_names)
    
    report = classification_report(y_label, y_predict, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    return y_label, y_predict


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
    df_train, df_val = split_train_val(df_original)
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")
    
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss().to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    num_epochs = config.NUM_EPOCHS
    
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
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{num_epochs} Summary:')
        print(f'  Train - Loss: {train_loss:.5f}, Acc: {train_acc:.5f}')
        print(f'  Val   - Loss: {val_loss:.5f}, Acc: {val_acc:.5f}\n')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('*****************************************************')
            print(f'★ Best model! [epoch {epoch}], [val loss {val_loss:.5f}], [val acc {val_acc:.5f}]')
            print('*****************************************************\n')
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.5f}')


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
    _, df_val = split_train_val(df_original)
    print(f"Validation samples: {len(df_val)}")
    
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
