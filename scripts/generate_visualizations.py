"""
Script to generate confusion matrix and ROC curves from existing model checkpoints
"""
import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import config
from src.data.dataset import HAM10000, ensure_data_downloaded
from src.data.preprocessing import (
    load_image_paths, prepare_dataframe, identify_duplicates,
    split_train_val_test, balance_classes
)
from src.models.model import initialize_model
import sys

# Ensure project root is on path when run as scripts/generate_visualizations.py
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from main import plot_confusion_matrix, plot_roc_curves
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize


def load_model_from_checkpoint(model_name, checkpoint_path, num_classes=7, device='cuda'):
    """Load model from checkpoint"""
    model = initialize_model(
        model_name=model_name,
        num_classes=num_classes,
        feature_extract=config.FEATURE_EXTRACT,
        use_pretrained=config.USE_PRETRAINED
    )
    model = model.to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from: {checkpoint_path}')
    else:
        print(f'Warning: Checkpoint not found at {checkpoint_path}')
        return None
    
    return model


def evaluate_model_for_visualization(model, test_loader, device, class_names, save_prefix=''):
    """Evaluate model and generate visualizations"""
    model.eval()
    y_label = []
    y_predict = []
    y_scores = []
    
    print('Evaluating model on test set...')
    with torch.no_grad():
        for i, data in enumerate(test_loader):
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
    f1_scores = f1_score(y_label, y_predict, average=None)
    f1_macro = f1_score(y_label, y_predict, average='macro')
    f1_weighted = f1_score(y_label, y_predict, average='weighted')
    
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
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"  Macro F1: {f1_macro:.5f}")
    print(f"  Weighted F1: {f1_weighted:.5f}")
    print("  Per-class F1:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {f1_scores[i]:.5f}")
    
    print("\nROC AUC Scores:")
    print(f"  Micro-average AUC: {roc_auc['micro']:.5f}")
    print("  Per-class AUC:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {roc_auc[i]:.5f}")
    
    return {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': {class_names[i]: float(f1_scores[i]) for i in range(len(class_names))},
        'roc_auc_micro': float(roc_auc['micro']),
        'roc_auc_per_class': {class_names[i]: float(roc_auc[i]) for i in range(len(class_names))}
    }


def main():
    """Generate visualizations for existing models"""
    device = torch.device(config.DEVICE)
    print(f'Using device: {device}\n')
    
    # Load data
    ensure_data_downloaded()
    df_original = load_image_paths()
    df_original = prepare_dataframe(df_original)
    df_original = identify_duplicates(df_original)
    df_train, df_val, df_test = split_train_val_test(df_original)
    
    print(f"Test samples: {len(df_test)}")
    
    # Get transforms
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]
    val_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    test_dataset = HAM10000(df_test, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Models to process
    models_to_process = ['resnet', 'densenet']
    results_dir = 'results'
    
    for model_name in models_to_process:
        print(f'\n{"="*60}')
        print(f'Processing {model_name.upper()}')
        print(f'{"="*60}')
        
        checkpoint_path = os.path.join(results_dir, f'best_model_{model_name}.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f'  Skipping {model_name}: checkpoint not found at {checkpoint_path}')
            continue
        
        # Load model
        model = load_model_from_checkpoint(
            model_name, 
            checkpoint_path, 
            num_classes=len(config.CLASS_NAMES),
            device=device
        )
        
        if model is None:
            continue
        
        # Evaluate and generate visualizations
        save_prefix = f'{model_name}_'
        metrics = evaluate_model_for_visualization(
            model, test_loader, device, config.CLASS_NAMES, save_prefix=save_prefix
        )
        
        # Move visualizations to results directory
        import shutil
        for viz_file in ['confusion_matrix', 'confusion_matrix_normalized', 'roc_curves']:
            source = f'{save_prefix}{viz_file}.png'
            if os.path.exists(source):
                dest = os.path.join(results_dir, f'{viz_file}_{model_name}.png')
                shutil.move(source, dest)
                print(f'  Saved {viz_file} to: {dest}')
        
        # Update result JSON if exists
        result_files = [f for f in os.listdir(results_dir) 
                       if f.startswith(f'{model_name}_') and f.endswith('.json')]
        if result_files:
            result_file = os.path.join(results_dir, result_files[0])
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Update metrics
            if 'result' in result_data and 'metrics' in result_data['result']:
                result_data['result']['metrics'].update({
                    'test_f1_macro': metrics['f1_macro'],
                    'test_f1_weighted': metrics['f1_weighted'],
                    'test_f1_per_class': metrics['f1_per_class'],
                    'test_roc_auc_micro': metrics['roc_auc_micro'],
                    'test_roc_auc_per_class': metrics['roc_auc_per_class']
                })
                
                # Update visualization paths
                result_data['result']['evaluation_visualizations'] = {
                    'confusion_matrix': f'results/confusion_matrix_{model_name}.png',
                    'confusion_matrix_normalized': f'results/confusion_matrix_normalized_{model_name}.png',
                    'roc_curves': f'results/roc_curves_{model_name}.png'
                }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            print(f'  Updated result file: {result_file}')
    
    print(f'\n{"="*60}')
    print('Visualization generation completed!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

