import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_nilm_metrics(y_true, y_pred, threshold=10, scaler=None):
    """
    Calculate NILM-specific metrics with custom SAE calculation matching test_redd_specific_splits.py
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Power threshold for on/off state (Watts)
        scaler: Optional scaler to denormalize data for classification metrics

    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Denormalize if scaler provided
    if scaler is not None:
        y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # Normalized Error in Total Energy (NETE)
    energy_true = np.sum(y_true)
    energy_pred = np.sum(y_pred)
    if energy_true > 0:
        nete = np.abs(energy_true - energy_pred) / energy_true
    else:
        nete = np.inf
    
    # Custom SAE calculation matching test_redd_specific_splits.py
    N = 100  # Window size used in sequence creation
    num_period = int(len(y_true) / N)
    diff = 0
    for i in range(num_period):
        diff += abs(np.sum(y_true[i * N: (i + 1) * N]) - np.sum(y_pred[i * N: (i + 1) * N]))
    sae = diff / (N * num_period)
    
    # State Accuracy Error (SAE) - traditional calculation for comparison
    y_true_binary = y_true_orig > threshold
    y_pred_binary = y_pred_orig > threshold
    
    # Calculate confusion matrix components
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    total_samples = len(y_true_binary)
    if total_samples > 0:
        state_accuracy = (tp + tn) / total_samples
        traditional_sae = 1.0 - state_accuracy
    else:
        traditional_sae = 0.0
    
    # Calculate precision, recall, and F1 score
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    
    if np.sum(y_true_binary) > 0 or np.sum(y_pred_binary) > 0:
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'nete': nete,
        'sae': sae,  # Custom SAE calculation
        'traditional_sae': traditional_sae,  # Traditional SAE for comparison
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(model, model_params, train_params, metrics, model_path):
    """
    Save model and metadata
    
    Args:
        model: The PyTorch model
        model_params: Dictionary of model parameters
        train_params: Dictionary of training parameters
        metrics: Dictionary of evaluation metrics
        model_path: Path to save the model
    """
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Make sure the path is valid (no trailing spaces in folder or file names)
    model_path = model_path.replace(' \\', '\\').replace('\\ ', '\\')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'train_params': train_params,
        'metrics': metrics
    }
    
    try:
        torch.save(checkpoint, model_path)
        print(f"Model successfully saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        # Try an alternative path if original fails
        alt_path = os.path.join(os.path.dirname(model_path), "model_backup.pth")
        try:
            torch.save(checkpoint, alt_path)
            print(f"Model saved to alternative path: {alt_path}")
            return alt_path
        except Exception as e2:
            print(f"Failed to save model to alternative path: {str(e2)}")

def load_model(model_class, model_path):
    """
    Load model and metadata
    
    Args:
        model_class: The model class to instantiate
        model_path: Path to the saved model
        
    Returns:
        Loaded model and metadata
    """
    checkpoint = torch.load(model_path)
    
    model_params = checkpoint['model_params']
    
    # Create model instance
    model = model_class(**model_params)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def plot_prediction_examples(y_true, y_pred, appliance_name, save_path=None, num_examples=3, sample_length=200):
    """
    Plot examples of true vs predicted values
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        appliance_name: Name of the appliance
        save_path: Path to save the plot
        num_examples: Number of example segments to plot
        sample_length: Length of each example segment
    """
    plt.figure(figsize=(15, 5 * num_examples))
    
    total_samples = len(y_true)
    
    for i in range(num_examples):
        # Generate random start index
        start_idx = np.random.randint(0, total_samples - sample_length)
        
        plt.subplot(num_examples, 1, i+1)
        plt.plot(y_true[start_idx:start_idx+sample_length], label='True', color='blue')
        plt.plot(y_pred[start_idx:start_idx+sample_length], label='Predicted', color='red')
        plt.title(f'Example {i+1}: {appliance_name} Power Consumption')
        plt.xlabel('Timestep')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_model_results(results_dict, metric_name='mae', save_path=None):
    """
    Compare results from different models
    
    Args:
        results_dict: Dictionary mapping model names to their results
        metric_name: Name of the metric to compare
        save_path: Path to save the plot
    """
    models = list(results_dict.keys())
    metric_values = [results_dict[model][metric_name] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, metric_values)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(metric_values),
            f'{value:.4f}',
            ha='center', va='bottom',
            rotation=0
        )
    
    plt.title(f'Comparison of {metric_name.upper()} across Models')
    plt.ylabel(metric_name.upper())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_all_metrics_comparison(results_dict, save_dir=None):
    """
    Generate comparison plots for all metrics
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_dir: Directory to save the plots
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get all metrics
    model_name = list(results_dict.keys())[0]
    metrics = list(results_dict[model_name].keys())
    
    for metric in metrics:
        if save_dir:
            save_path = os.path.join(save_dir, f'{metric}_comparison.png')
        else:
            save_path = None
        
        compare_model_results(results_dict, metric, save_path)
