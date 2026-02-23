#!/usr/bin/env python3
"""
Experiment script to train multiple models (resnet, densenet, vgg, inception)
Each experiment runs independently with separate wandb runs and saves results.
"""

import os
import json
import shutil
import csv
from datetime import datetime
import config
import main

# Models to experiment with
MODELS = ['swin', 'custom_cnn']

RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiment_results.json')


def collect_config():
    """Collect all config parameters"""
    return {
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'LR_SCHEDULER': config.LR_SCHEDULER,
        'LR_SCHEDULER_PARAMS': config.LR_SCHEDULER_PARAMS,
        'TRAIN_SIZE': config.TRAIN_SIZE,
        'VAL_SIZE': config.VAL_SIZE,
        'TEST_SIZE': config.TEST_SIZE,
        'RANDOM_STATE': config.RANDOM_STATE,
        'FEATURE_EXTRACT': config.FEATURE_EXTRACT,
        'USE_PRETRAINED': config.USE_PRETRAINED,
        'DATA_AUG_RATE': config.DATA_AUG_RATE,
        'INPUT_SIZE': config.INPUT_SIZE,
    }


def ensure_results_dir():
    """Ensure results directory exists"""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_model_checkpoint(model_name, source_path='best_model.pth'):
    """Save model checkpoint with model name"""
    if os.path.exists(source_path):
        dest_path = os.path.join(RESULTS_DIR, f'best_model_{model_name}.pth')
        shutil.copy(source_path, dest_path)
        return dest_path
    return None


def save_training_curves(model_name, source_path='training_curves.png'):
    """Save training curves with model name"""
    if os.path.exists(source_path):
        dest_path = os.path.join(RESULTS_DIR, f'training_curves_{model_name}.png')
        shutil.copy(source_path, dest_path)
        return dest_path
    return None


def save_evaluation_visualizations(model_name, source_prefix='test_'):
    """Save confusion matrix and ROC curves with model name"""
    saved_files = {}
    
    # Confusion matrix
    cm_path = f'{source_prefix}confusion_matrix.png'
    if os.path.exists(cm_path):
        dest_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png')
        shutil.copy(cm_path, dest_path)
        saved_files['confusion_matrix'] = dest_path
    
    # Normalized confusion matrix
    cm_norm_path = f'{source_prefix}confusion_matrix_normalized.png'
    if os.path.exists(cm_norm_path):
        dest_path = os.path.join(RESULTS_DIR, f'confusion_matrix_normalized_{model_name}.png')
        shutil.copy(cm_norm_path, dest_path)
        saved_files['confusion_matrix_normalized'] = dest_path
    
    # ROC curves
    roc_path = f'{source_prefix}roc_curves.png'
    if os.path.exists(roc_path):
        dest_path = os.path.join(RESULTS_DIR, f'roc_curves_{model_name}.png')
        shutil.copy(roc_path, dest_path)
        saved_files['roc_curves'] = dest_path
    
    return saved_files if saved_files else None


def save_model_result(model_name, result, experiment_config):
    """Save individual model result to separate JSON file with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{model_name}_{timestamp}.json'
    filepath = os.path.join(RESULTS_DIR, filename)
    
    model_result = {
        'model_name': model_name,
        'timestamp': timestamp,
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': experiment_config,
        'result': result
    }
    
    with open(filepath, 'w') as f:
        json.dump(model_result, f, indent=2, default=str)
    
    return filepath


def load_existing_results():
    """Load existing results from summary file if available"""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    
    # Also check for individual model result files
    if os.path.exists(RESULTS_DIR):
        results = {
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_config': {},
            'models': {}
        }
        
        # Load individual model results
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json') and filename != 'experiment_results.json':
                try:
                    filepath = os.path.join(RESULTS_DIR, filename)
                    with open(filepath, 'r') as f:
                        model_data = json.load(f)
                        model_name = model_data.get('model_name')
                        if model_name:
                            results['models'][model_name] = model_data.get('result', {})
                            if not results['experiment_config']:
                                results['experiment_config'] = model_data.get('config', {})
                except:
                    continue
        
        if results['models']:
            return results
    
    return None


def save_results(results_dict):
    """Save summary results to JSON file"""
    ensure_results_dir()
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    return RESULTS_FILE


def run_experiment(model_name):
    """Run training experiment for a specific model"""
    print(f'\n[{model_name.upper()}] Training...')
    
    # Save original model name
    original_model_name = config.MODEL_NAME
    
    try:
        # Set model name for this experiment
        config.MODEL_NAME = model_name
        
        # Run training and get results (progress bars will be shown by main.train())
        training_results = main.train()
        
        # Collect results
        # Save visualizations
        eval_viz = save_evaluation_visualizations(model_name)
        
        result = {
            'model': model_name,
            'status': 'completed',
            'best_model_path': save_model_checkpoint(model_name),
            'training_curves_path': save_training_curves(model_name),
            'evaluation_visualizations': eval_viz,
            'metrics': training_results if training_results else {},
        }
        
        # Save individual model result
        experiment_config = collect_config()
        model_result_file = save_model_result(model_name, result, experiment_config)
        print(f'  Model result saved to: {model_result_file}')
        
        if training_results:
            print(f'[{model_name.upper()}] Completed - Val Acc: {training_results.get("best_val_acc", 0):.5f}, Test Acc: {training_results.get("test_acc", 0):.5f}')
        else:
            print(f'[{model_name.upper()}] Completed')
        
        return result
        
    except Exception as e:
        print(f'[{model_name.upper()}] Failed: {str(e)[:100]}')
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        # Restore original model name
        config.MODEL_NAME = original_model_name


def generate_comparison_table(results_dict, output_format='both'):
    """Generate comparison table in CSV and/or Markdown format"""
    models = results_dict.get('models', {})
    
    # Prepare table data
    table_data = []
    for model_name, result in models.items():
        row = {'Model': model_name}
        
        if result['status'] == 'completed' and 'metrics' in result:
            metrics = result['metrics']
            row['Status'] = 'Completed'
            row['Best Val Acc'] = f"{metrics.get('best_val_acc', 0):.5f}" if metrics.get('best_val_acc') else 'N/A'
            row['Test Acc'] = f"{metrics.get('test_acc', 0):.5f}" if metrics.get('test_acc') else 'N/A'
            row['Test Loss'] = f"{metrics.get('test_loss', 0):.5f}" if metrics.get('test_loss') else 'N/A'
            row['Final Train Acc'] = f"{metrics.get('final_train_acc', 0):.5f}" if metrics.get('final_train_acc') else 'N/A'
            row['Final Val Acc'] = f"{metrics.get('final_val_acc', 0):.5f}" if metrics.get('final_val_acc') else 'N/A'
        else:
            row['Status'] = 'Failed'
            row['Best Val Acc'] = 'N/A'
            row['Test Acc'] = 'N/A'
            row['Test Loss'] = 'N/A'
            row['Final Train Acc'] = 'N/A'
            row['Final Val Acc'] = 'N/A'
            row['Error'] = result.get('error', 'Unknown')[:100]
        
        table_data.append(row)
    
    # Generate CSV
    if output_format in ['csv', 'both']:
        csv_file = os.path.join(RESULTS_DIR, 'experiment_comparison.csv')
        ensure_results_dir()
        with open(csv_file, 'w', newline='') as f:
            if table_data:
                writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
                writer.writeheader()
                writer.writerows(table_data)
        print(f'  CSV table saved to: {csv_file}')
    
    # Generate Markdown
    if output_format in ['markdown', 'both']:
        md_file = os.path.join(RESULTS_DIR, 'experiment_comparison.md')
        ensure_results_dir()
        with open(md_file, 'w') as f:
            f.write('# Experiment Results Comparison\n\n')
            f.write(f'**Experiment Date:** {results_dict.get("experiment_date", "N/A")}\n\n')
            
            if table_data:
                # Write header
                headers = list(table_data[0].keys())
                f.write('| ' + ' | '.join(headers) + ' |\n')
                f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
                
                # Write rows
                for row in table_data:
                    f.write('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |\n')
        print(f'  Markdown table saved to: {md_file}')
    
    return table_data


def main_experiment():
    """Run all experiments"""
    ensure_results_dir()
    
    print('HAM10000 Model Comparison Experiment')
    print(f'Models: {", ".join(MODELS)} | Total: {len(MODELS)} experiments')
    print(f'Results directory: {RESULTS_DIR}\n')
    
    # Try to load existing results
    results = load_existing_results()
    if results:
        print(f'Found existing results in: {RESULTS_DIR}')
        completed_models = [m for m, r in results.get('models', {}).items() 
                           if r.get('status') == 'completed']
        if completed_models:
            print(f'  Already completed: {", ".join(completed_models)}')
            print(f'  Will skip these models and continue with remaining ones\n')
        else:
            print('  No completed models found, starting fresh\n')
    else:
        # Initialize new results
        results = {
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_config': {},
            'models': {}
        }
        print('Starting new experiment\n')
    
    # Update experiment config (in case it changed)
    results['experiment_config'] = collect_config()
    
    # Run experiments (skip already completed ones)
    for i, model_name in enumerate(MODELS, 1):
        # Check if model is already completed
        if model_name in results.get('models', {}) and results['models'][model_name].get('status') == 'completed':
            print(f'Experiment {i}/{len(MODELS)}: [{model_name.upper()}] Already completed, skipping...')
            continue
        
        print(f'Experiment {i}/{len(MODELS)}:')
        result = run_experiment(model_name)
        results['models'][model_name] = result
        
        # Save summary immediately after each model completes
        save_results(results)
        print(f'  Summary saved to: {RESULTS_FILE}')
    
    # Generate comparison tables
    print('\nGenerating comparison tables...')
    generate_comparison_table(results, output_format='both')
    
    # Print summary
    print('\nSummary:')
    completed = sum(1 for r in results['models'].values() if r.get('status') == 'completed')
    failed = sum(1 for r in results['models'].values() if r.get('status') == 'failed')
    pending = len(MODELS) - completed - failed
    print(f'  Completed: {completed}/{len(MODELS)}')
    print(f'  Failed: {failed}/{len(MODELS)}')
    if pending > 0:
        print(f'  Pending: {pending}/{len(MODELS)}')
    print(f'\nAll results saved in: {RESULTS_DIR}/')


if __name__ == '__main__':
    main_experiment()
