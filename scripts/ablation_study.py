"""
Ablation Study Script for HAM10000 Classification
Experiments:
1. Baseline: ResNet + Cosine + Augmentation
2. ResNet + Step LR + Augmentation
3. ResNet + Exponential LR + Augmentation
4. ResNet + Plateau LR + Augmentation
"""
import os
import json
import shutil
import csv
from datetime import datetime
import config
import main


ABLATION_DIR = 'ablation_results'
ABLATION_FILE = os.path.join(ABLATION_DIR, 'ablation_study.json')


def ensure_ablation_dir():
    """Ensure ablation results directory exists"""
    os.makedirs(ABLATION_DIR, exist_ok=True)


def collect_config_snapshot():
    """Collect current config as snapshot"""
    return {
        'MODEL_NAME': config.MODEL_NAME,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'LR_SCHEDULER': config.LR_SCHEDULER,
        'LR_SCHEDULER_PARAMS': config.LR_SCHEDULER_PARAMS.copy(),
        'DATA_AUG_RATE': config.DATA_AUG_RATE.copy(),
        'FEATURE_EXTRACT': config.FEATURE_EXTRACT,
        'USE_PRETRAINED': config.USE_PRETRAINED,
    }


def save_experiment_result(exp_name, exp_config, metrics, model_path=None, curves_path=None):
    """Save individual experiment result"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'config': exp_config,
        'metrics': metrics,
        'model_path': model_path,
        'training_curves_path': curves_path,
    }
    
    filename = f'{exp_name}_{timestamp}.json'
    filepath = os.path.join(ABLATION_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return filepath


def save_model_and_curves(exp_name, source_model='best_model.pth', source_curves='training_curves.png'):
    """Save model checkpoint and training curves with experiment name"""
    saved_files = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if os.path.exists(source_model):
        dest_model = os.path.join(ABLATION_DIR, f'{exp_name}_model_{timestamp}.pth')
        shutil.copy(source_model, dest_model)
        saved_files['model'] = dest_model
    
    if os.path.exists(source_curves):
        dest_curves = os.path.join(ABLATION_DIR, f'{exp_name}_curves_{timestamp}.png')
        shutil.copy(source_curves, dest_curves)
        saved_files['curves'] = dest_curves
    
    # Save evaluation visualizations if exist
    for viz_file in ['test_confusion_matrix.png', 'test_confusion_matrix_normalized.png', 'test_roc_curves.png']:
        if os.path.exists(viz_file):
            dest_viz = os.path.join(ABLATION_DIR, f'{exp_name}_{viz_file.replace("test_", "")}')
            shutil.copy(viz_file, dest_viz)
            saved_files[viz_file.replace('test_', '')] = dest_viz
    
    return saved_files


def load_ablation_results():
    """Load existing ablation study results"""
    if os.path.exists(ABLATION_FILE):
        with open(ABLATION_FILE, 'r') as f:
            return json.load(f)
    return {
        'study_name': 'ResNet Ablation Study',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }


def save_ablation_results(results):
    """Save ablation study results"""
    ensure_ablation_dir()
    with open(ABLATION_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def run_experiment(exp_name, exp_config_modifier):
    """Run a single ablation experiment"""
    print(f'\n{"="*80}')
    print(f'Running Experiment: {exp_name}')
    print(f'{"="*80}')
    
    # Save original config
    original_config = collect_config_snapshot()
    
    try:
        # Apply experiment-specific config modifications
        exp_config_modifier()
        
        # Print experiment config
        print(f'\nExperiment Configuration:')
        print(f'  Model: {config.MODEL_NAME}')
        print(f'  LR Scheduler: {config.LR_SCHEDULER}')
        print(f'  Data Augmentation: {"Enabled" if any(config.DATA_AUG_RATE) else "Disabled"}')
        if config.LR_SCHEDULER != 'fixed':
            print(f'  LR Scheduler Params: {config.LR_SCHEDULER_PARAMS.get(config.LR_SCHEDULER, {})}')
        
        # Set wandb run name and tags for this experiment
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb_run_name = f"ablation_{exp_name}_{timestamp}"
        os.environ['WANDB_RUN_NAME'] = wandb_run_name
        os.environ['WANDB_TAGS'] = 'ablation_study,resnet'
        
        # Run training
        training_results = main.train()
        
        # Save model and curves
        saved_files = save_model_and_curves(exp_name)
        
        # Collect experiment config
        exp_config = collect_config_snapshot()
        
        # Save experiment result
        result_file = save_experiment_result(
            exp_name,
            exp_config,
            training_results if training_results else {},
            saved_files.get('model'),
            saved_files.get('curves')
        )
        
        print(f'\n[{exp_name}] Completed')
        print(f'  Result saved to: {result_file}')
        if training_results:
            print(f'  Best Val Acc: {training_results.get("best_val_acc", 0):.5f}')
            print(f'  Test Acc: {training_results.get("test_acc", 0):.5f}')
        
        return {
            'status': 'completed',
            'result_file': result_file,
            'metrics': training_results if training_results else {},
            'config': exp_config
        }
        
    except Exception as e:
        print(f'\n[{exp_name}] Failed: {str(e)[:200]}')
        return {
            'status': 'failed',
            'error': str(e),
            'config': collect_config_snapshot()
        }
    
    finally:
        # Clean up environment variables
        if 'WANDB_RUN_NAME' in os.environ:
            del os.environ['WANDB_RUN_NAME']
        if 'WANDB_TAGS' in os.environ:
            del os.environ['WANDB_TAGS']
        # Restore original config
        restore_config(original_config)


def restore_config(original_config):
    """Restore config to original values"""
    config.MODEL_NAME = original_config['MODEL_NAME']
    config.LR_SCHEDULER = original_config['LR_SCHEDULER']
    config.LR_SCHEDULER_PARAMS = original_config['LR_SCHEDULER_PARAMS']
    config.DATA_AUG_RATE = original_config['DATA_AUG_RATE']


def generate_comparison_table(results):
    """Generate comparison table for all experiments"""
    print(f'\n{"="*80}')
    print('Generating Comparison Table')
    print(f'{"="*80}')
    
    experiments = results.get('experiments', {})
    
    # Prepare table data
    table_data = []
    for exp_name, exp_data in experiments.items():
        if exp_data.get('status') == 'completed' and 'metrics' in exp_data:
            metrics = exp_data['metrics']
            config_snap = exp_data.get('config', {})
            
            row = {
                'Experiment': exp_name,
                'LR_Scheduler': config_snap.get('LR_SCHEDULER', 'N/A'),
                'Data_Augmentation': 'Yes' if any(config_snap.get('DATA_AUG_RATE', [])) else 'No',
                'Best_Val_Acc': f"{metrics.get('best_val_acc', 0):.5f}" if metrics.get('best_val_acc') else 'N/A',
                'Test_Acc': f"{metrics.get('test_acc', 0):.5f}" if metrics.get('test_acc') else 'N/A',
                'Test_Loss': f"{metrics.get('test_loss', 0):.5f}" if metrics.get('test_loss') else 'N/A',
                'Test_F1_Macro': f"{metrics.get('test_f1_macro', 0):.5f}" if metrics.get('test_f1_macro') else 'N/A',
                'Test_ROC_AUC': f"{metrics.get('test_roc_auc_micro', 0):.5f}" if metrics.get('test_roc_auc_micro') else 'N/A',
            }
            table_data.append(row)
        else:
            row = {
                'Experiment': exp_name,
                'LR_Scheduler': 'N/A',
                'Data_Augmentation': 'N/A',
                'Best_Val_Acc': 'N/A',
                'Test_Acc': 'N/A',
                'Test_Loss': 'N/A',
                'Test_F1_Macro': 'N/A',
                'Test_ROC_AUC': 'N/A',
            }
            if exp_data.get('status') == 'failed':
                row['Error'] = exp_data.get('error', 'Unknown')[:100]
            table_data.append(row)
    
    # Generate CSV
    csv_file = os.path.join(ABLATION_DIR, 'ablation_comparison.csv')
    if table_data:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)
        print(f'  CSV table saved to: {csv_file}')
    
    # Generate Markdown
    md_file = os.path.join(ABLATION_DIR, 'ablation_comparison.md')
    with open(md_file, 'w') as f:
        f.write('# Ablation Study Results\n\n')
        f.write(f"**Study:** {results.get('study_name', 'N/A')}\n")
        f.write(f"**Created:** {results.get('created_at', 'N/A')}\n\n")
        
        if table_data:
            headers = list(table_data[0].keys())
            f.write('| ' + ' | '.join(headers) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
            
            for row in table_data:
                f.write('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |\n')
    
    print(f'  Markdown table saved to: {md_file}')
    
    return table_data


def main_ablation_study():
    """Run all ablation experiments"""
    ensure_ablation_dir()
    
    print('='*80)
    print('ResNet Ablation Study')
    print('='*80)
    print('\nExperiments:')
    print('  1. Baseline: ResNet + Cosine + Augmentation')
    print('  2. ResNet + Step LR + Augmentation')
    print('  3. ResNet + Exponential LR + Augmentation')
    print('  4. ResNet + Plateau LR + Augmentation')
    print(f'\nResults directory: {ABLATION_DIR}\n')
    
    # Load existing results
    results = load_ablation_results()
    
    # Define experiments (baseline + learning rate scheduler comparisons)
    experiments = [
        {
            'name': 'baseline',
            'description': 'ResNet + Cosine + Augmentation',
            'modifier': lambda: None  # Use default config (cosine scheduler)
        },
        {
            'name': 'step_lr',
            'description': 'ResNet + Step LR + Augmentation',
            'modifier': lambda: (
                setattr(config, 'LR_SCHEDULER', 'step'),
                setattr(config, 'DATA_AUG_RATE', [15, 10, 5, 50, 0, 40, 5])  # Restore augmentation
            )
        },
        {
            'name': 'exp_lr',
            'description': 'ResNet + Exponential LR + Augmentation',
            'modifier': lambda: (
                setattr(config, 'LR_SCHEDULER', 'exponential'),
                setattr(config, 'DATA_AUG_RATE', [15, 10, 5, 50, 0, 40, 5])  # Restore augmentation
            )
        },
        {
            'name': 'plateau_lr',
            'description': 'ResNet + Plateau LR + Augmentation',
            'modifier': lambda: (
                setattr(config, 'LR_SCHEDULER', 'plateau'),
                setattr(config, 'DATA_AUG_RATE', [15, 10, 5, 50, 0, 40, 5])  # Restore augmentation
            )
        },
    ]
    
    # Set model to ResNet for all experiments
    original_model = config.MODEL_NAME
    config.MODEL_NAME = 'resnet'
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        exp_name = exp['name']
        
        # Check if already completed
        if exp_name in results.get('experiments', {}) and results['experiments'][exp_name].get('status') == 'completed':
            print(f'\nExperiment {i}/{len(experiments)}: [{exp_name}] Already completed, skipping...')
            continue
        
        print(f'\nExperiment {i}/{len(experiments)}:')
        result = run_experiment(exp_name, exp['modifier'])
        results['experiments'][exp_name] = result
        
        # Save after each experiment
        save_ablation_results(results)
        print(f'  Progress saved to: {ABLATION_FILE}')
    
    # Restore original model
    config.MODEL_NAME = original_model
    
    # Generate comparison table
    generate_comparison_table(results)
    
    # Print summary
    print(f'\n{"="*80}')
    print('Ablation Study Summary')
    print(f'{"="*80}')
    completed = sum(1 for exp in results['experiments'].values() if exp.get('status') == 'completed')
    failed = sum(1 for exp in results['experiments'].values() if exp.get('status') == 'failed')
    print(f'  Completed: {completed}/{len(experiments)}')
    print(f'  Failed: {failed}/{len(experiments)}')
    print(f'\nAll results saved in: {ABLATION_DIR}/')
    print(f'  - ablation_study.json: Complete results')
    print(f'  - ablation_comparison.csv: Comparison table (CSV)')
    print(f'  - ablation_comparison.md: Comparison table (Markdown)')


if __name__ == '__main__':
    main_ablation_study()

