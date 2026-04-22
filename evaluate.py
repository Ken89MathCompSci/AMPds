import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

# Import from our modules
from data_loader import load_and_preprocess_ukdale, explore_available_appliances
from models import LSTMModel, GRUModel, TCNModel, LiquidNetworkModel, AdvancedLiquidNetworkModel, ResNetModel, SimpleTransformerModel
from utils import calculate_nilm_metrics, load_model, plot_prediction_examples

def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a dataset
    
    Args:
        model: The PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run the model on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store targets and outputs for metrics calculation
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    # Calculate metrics
    metrics = calculate_nilm_metrics(all_targets, all_outputs)
    
    return metrics, all_targets, all_outputs

def load_trained_model(model_type, model_path, device):
    """Load a trained model from disk"""
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_params = checkpoint['model_params']
        
        # Create the appropriate model
        if model_type == 'lstm':
            model = LSTMModel(**model_params)
        elif model_type == 'gru':
            model = GRUModel(**model_params)
        elif model_type == 'tcn':
            model = TCNModel(**model_params)
        elif model_type == 'liquid':
            model = LiquidNetworkModel(**model_params)
        elif model_type == 'advanced_liquid':
            model = AdvancedLiquidNetworkModel(**model_params)
        elif model_type == 'resnet':
            model = ResNetModel(**model_params) 
        elif model_type == 'transformer':
            model = SimpleTransformerModel(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def evaluate_and_compare_all_models(models_info, house_number=1, results_dir='results'):
    """
    Evaluate and compare all trained models for all appliances
    
    Args:
        models_info: Dictionary mapping model types to their base directories
        house_number: House number in the UK-DALE dataset
        results_dir: Directory to save the results
    """
    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(results_dir, f"evaluation_house{house_number}_{timestamp}")
    
    # Create results directory
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Load data file
    file_path = f"preprocessed_datasets/ukdale/ukdale{house_number}.mat"
    
    # Get available appliances
    appliances = explore_available_appliances(file_path)
    print(f"Evaluating models for {len(appliances)} appliances in house {house_number}:")
    for idx, name in appliances.items():
        print(f"  Index {idx}: {name}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store evaluation results
    all_results = {}
    
    # Create configuration
    config = {
        'house_number': house_number,
        'timestamp': timestamp,
        'models_evaluated': list(models_info.keys()),
        'appliances': appliances
    }
    
    with open(os.path.join(base_results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Evaluate models for each appliance
    for appliance_idx, appliance_name in appliances.items():
        print(f"\n{'-'*50}")
        print(f"Evaluating models for {appliance_name} (index {appliance_idx})")
        print(f"{'-'*50}\n")
        
        # Create appliance-specific results directory
        appliance_dir = os.path.join(base_results_dir, f"{appliance_name}")
        os.makedirs(appliance_dir, exist_ok=True)
        
        try:
            # Load data for this appliance
            data_dict = load_and_preprocess_ukdale(
                file_path,
                appliance_idx,
                window_size=100,  # Use consistent window size
                target_size=1
            )
            
            # Use the test data loader for evaluation
            test_loader = data_dict['test_loader']
            
            # Dictionary to store results for this appliance
            appliance_results = {}
            
            # Evaluate each model type
            for model_type, model_base_dir in models_info.items():
                print(f"\nEvaluating {model_type} model for {appliance_name}...")
                
                # Find the model for this appliance
                model_dir = None
                for root, dirs, files in os.walk(model_base_dir):
                    for dir_name in dirs:
                        if appliance_name in dir_name:
                            model_dir = os.path.join(root, dir_name)
                            break
                    if model_dir:
                        break
                
                if not model_dir:
                    print(f"No {model_type} model directory found for {appliance_name}, skipping...")
                    continue
                
                # Find the best model file
                model_path = None
                best_model_name = f"{model_type}_model_best.pth" if model_type != 'advanced_liquid' else "advanced_liquid_model_best.pth"
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file == best_model_name:
                            model_path = os.path.join(root, file)
                            break
                    if model_path:
                        break
                
                if not model_path:
                    print(f"No {model_type} model file found for {appliance_name}, skipping...")
                    continue
                
                # Load the model
                model = load_trained_model(model_type, model_path, device)
                
                if model is None:
                    print(f"Failed to load {model_type} model for {appliance_name}, skipping...")
                    continue
                
                # Evaluate the model
                try:
                    metrics, targets, outputs = evaluate_model(model, test_loader, device)
                    
                    # Store results
                    appliance_results[model_type] = {
                        'metrics': metrics,
                        'model_path': model_path
                    }
                    
                    # Plot example predictions
                    plot_path = os.path.join(appliance_dir, f"{model_type}_predictions.png")
                    plot_prediction_examples(
                        targets.flatten(),
                        outputs.flatten(),
                        f"{appliance_name} - {model_type.upper()}",
                        plot_path
                    )
                    
                    print(f"{model_type} evaluation metrics for {appliance_name}:")
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
                
                except Exception as e:
                    print(f"Error evaluating {model_type} model for {appliance_name}: {str(e)}")
            
            # Store appliance results
            all_results[appliance_name] = appliance_results
            
            # Generate comparison plots for this appliance
            if len(appliance_results) > 0:
                generate_appliance_comparison_plots(appliance_name, appliance_results, appliance_dir)
            
        except Exception as e:
            print(f"Error processing appliance {appliance_name}: {str(e)}")
    
    # Generate summary plots across all appliances
    generate_summary_plots(all_results, base_results_dir)
    
    # Save all results to JSON
    with open(os.path.join(base_results_dir, 'all_results.json'), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_results = {}
        for appliance, models in all_results.items():
            json_results[appliance] = {}
            for model_type, result in models.items():
                json_results[appliance][model_type] = {
                    'metrics': {k: float(v) for k, v in result['metrics'].items()},
                    'model_path': result['model_path']
                }
        
        json.dump(json_results, f, indent=4)
    
    print(f"\nEvaluation completed. Results saved to {base_results_dir}")
    
    return all_results, base_results_dir

def generate_appliance_comparison_plots(appliance_name, results, save_dir):
    """
    Generate comparison plots for a single appliance
    
    Args:
        appliance_name: Name of the appliance
        results: Dictionary mapping model types to their results
        save_dir: Directory to save the plots
    """
    # Prepare data for bar plots
    model_types = list(results.keys())
    metrics = ['mae', 'rmse', 'nete', 'f1', 'precision', 'recall']
    
    # Create separate plot for each metric
    for metric in metrics:
        values = [results[model_type]['metrics'][metric] for model_type in model_types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_types, values)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 * max(values),
                f'{value:.4f}',
                ha='center', va='bottom',
                rotation=0
            )
        
        plt.title(f'{appliance_name} - {metric.upper()} Comparison')
        plt.ylabel(metric.upper())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_comparison.png"))
        plt.close()
    
    # Create combined plot for MAE and F1 score
    plt.figure(figsize=(12, 6))
    
    # Left y-axis for MAE
    ax1 = plt.gca()
    mae_values = [results[model_type]['metrics']['mae'] for model_type in model_types]
    bars1 = ax1.bar([x - 0.2 for x in range(len(model_types))], mae_values, width=0.4, label='MAE', color='blue')
    
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('MAE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add value labels
    for bar, value in zip(bars1, mae_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001 * max(mae_values),
            f'{value:.4f}',
            ha='center', va='bottom',
            color='blue', rotation=0
        )
    
    # Right y-axis for F1 score
    ax2 = ax1.twinx()
    f1_values = [results[model_type]['metrics']['f1'] for model_type in model_types]
    bars2 = ax2.bar([x + 0.2 for x in range(len(model_types))], f1_values, width=0.4, label='F1 Score', color='red')
    
    ax2.set_ylabel('F1 Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add value labels
    for bar, value in zip(bars2, f1_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001 * max(f1_values),
            f'{value:.4f}',
            ha='center', va='bottom',
            color='red', rotation=0
        )
    
    # Set x-tick labels
    ax1.set_xticks(range(len(model_types)))
    ax1.set_xticklabels(model_types)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title(f'{appliance_name} - Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_performance.png"))
    plt.close()

def generate_summary_plots(all_results, save_dir):
    """
    Generate summary plots comparing model performance across all appliances
    
    Args:
        all_results: Dictionary mapping appliance names to their results for all models
        save_dir: Directory to save the plots
    """
    # Check if we have results
    if not all_results:
        print("No results to generate summary plots.")
        return
    
    # Get list of all appliances and model types
    appliances = list(all_results.keys())
    
    # Get all unique model types across all appliances
    model_types = set()
    for app_results in all_results.values():
        model_types.update(app_results.keys())
    model_types = list(model_types)
    
    # Metrics to compare
    metrics = ['mae', 'rmse', 'f1']
    
    # Create separate plot for each metric
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(appliances))  # x-axis positions for appliances
        width = 0.8 / len(model_types)  # width of bars, adjusted for number of models
        
        # Plot bars for each model type
        for i, model_type in enumerate(model_types):
            values = []
            for appliance in appliances:
                if model_type in all_results[appliance]:
                    values.append(all_results[appliance][model_type]['metrics'][metric])
                else:
                    values.append(np.nan)  # Use NaN for missing data
            
            # Calculate x positions for this model type's bars
            positions = x + width * (i - len(model_types) / 2 + 0.5)
            
            # Plot bars
            plt.bar(positions, values, width, label=model_type)
        
        # Set labels and title
        plt.xlabel('Appliance')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Comparison Across All Appliances')
        
        # Set x-tick labels
        plt.xticks(x, appliances, rotation=45)
        
        # Add legend
        plt.legend()
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"summary_{metric}_comparison.png"))
        plt.close()
    
    # Generate heatmap of best model for each appliance
    generate_best_model_heatmap(all_results, save_dir)

def generate_best_model_heatmap(all_results, save_dir):
    """
    Generate a heatmap showing which model performs best for each appliance on each metric
    
    Args:
        all_results: Dictionary mapping appliance names to their results for all models
        save_dir: Directory to save the heatmap
    """
    # Get list of all appliances and model types
    appliances = list(all_results.keys())
    
    # Get all unique model types across all appliances
    model_types = set()
    for app_results in all_results.values():
        model_types.update(app_results.keys())
    model_types = list(model_types)
    
    # Metrics to compare
    metrics = ['mae', 'rmse', 'f1']
    
    # Create a matrix to store the best model for each appliance and metric
    best_models = {}
    
    # Determine which model is best for each appliance and metric
    for metric in metrics:
        best_models[metric] = []
        
        for appliance in appliances:
            if appliance not in all_results or not all_results[appliance]:
                best_models[metric].append("N/A")
                continue
                
            best_model = None
            best_value = float('inf') if metric in ['mae', 'rmse'] else float('-inf')
            
            for model_type, results in all_results[appliance].items():
                value = results['metrics'][metric]
                
                if metric in ['mae', 'rmse'] and value < best_value:
                    best_value = value
                    best_model = model_type
                elif metric not in ['mae', 'rmse'] and value > best_value:
                    best_value = value
                    best_model = model_type
            
            best_models[metric].append(best_model)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Create a numeric matrix for the heatmap
    heatmap_data = np.zeros((len(metrics), len(appliances)))
    
    # Assign a unique number to each model type
    model_to_num = {model: i+1 for i, model in enumerate(model_types)}
    
    # Fill in the matrix
    for i, metric in enumerate(metrics):
        for j, appliance in enumerate(appliances):
            best_model = best_models[metric][j]
            if best_model in model_to_num:
                heatmap_data[i, j] = model_to_num[best_model]
    
    # Create color map
    cmap = plt.cm.get_cmap('tab10', len(model_types) + 1)
    
    # Plot heatmap
    plt.imshow(heatmap_data, cmap=cmap, aspect='auto')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(appliances)):
            best_model = best_models[metrics[i]][j]
            plt.text(j, i, best_model, ha='center', va='center', 
                     color='white' if best_model in model_to_num and model_to_num[best_model] > len(model_types) / 2 else 'black')
    
    # Add colorbar
    cbar = plt.colorbar(ticks=list(range(1, len(model_types) + 1)))
    cbar.set_ticklabels(model_types)
    
    # Set labels and title
    plt.xlabel('Appliance')
    plt.ylabel('Metric')
    plt.title('Best Performing Model for Each Appliance and Metric')
    
    # Set tick labels
    plt.xticks(range(len(appliances)), appliances, rotation=45)
    plt.yticks(range(len(metrics)), [m.upper() for m in metrics])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_model_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    # Define the base directories for each model type
    models_info = {
        'lstm': 'models/lstm',
        'gru': 'models/gru',
        'tcn': 'models/tcn',
        'advanced_liquid': 'models/liquid',
        'resnet': 'models/resnet',
        'transformer': 'models/transformer'
    }
    
    # Evaluate and compare all models
    results, results_dir = evaluate_and_compare_all_models(models_info, house_number=1)
    
    print(f"Evaluation results saved to {results_dir}")