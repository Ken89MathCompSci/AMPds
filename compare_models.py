import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
from datetime import datetime

def load_evaluation_results(results_dir):
    """
    Load evaluation results from a directory
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Dictionary containing the loaded results
    """
    results_file = os.path.join(results_dir, 'all_results.json')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_comparison_tables(results):
    """
    Create comparison tables from evaluation results
    
    Args:
        results: Dictionary containing evaluation results
        
    Returns:
        Dictionary mapping metrics to pandas DataFrames with comparison tables
    """
    # Get list of all appliances and model types
    appliances = list(results.keys())
    
    # Get all unique model types across all appliances
    model_types = set()
    for app_results in results.values():
        model_types.update(app_results.keys())
    model_types = sorted(list(model_types))
    
    # Metrics to compare
    metrics = ['mae', 'rmse', 'nete', 'f1', 'precision', 'recall']
    
    # Create tables
    tables = {}
    
    for metric in metrics:
        # Create a DataFrame for this metric
        df = pd.DataFrame(index=appliances, columns=model_types)
        
        # Fill in the values
        for appliance in appliances:
            for model_type in model_types:
                if model_type in results[appliance]:
                    df.at[appliance, model_type] = results[appliance][model_type]['metrics'][metric]
        
        # Add a column for the best model
        if metric in ['mae', 'rmse', 'nete']:  # Lower is better
            df['best_model'] = df.idxmin(axis=1)
        else:  # Higher is better
            df['best_model'] = df.idxmax(axis=1)
        
        tables[metric] = df
    
    return tables

def create_summary_dataframe(tables):
    """
    Create a summary DataFrame counting how many times each model is the best
    
    Args:
        tables: Dictionary mapping metrics to comparison tables
        
    Returns:
        Summary DataFrame
    """
    # Get all model types (excluding 'best_model' column)
    model_types = [col for col in tables['mae'].columns if col != 'best_model']
    
    # Create a DataFrame to store the counts
    summary = pd.DataFrame(index=model_types, columns=tables.keys())
    summary = summary.fillna(0)
    
    # Count how many times each model is the best for each metric
    for metric, table in tables.items():
        counts = table['best_model'].value_counts()
        
        for model_type in model_types:
            if model_type in counts:
                summary.at[model_type, metric] = counts[model_type]
    
    # Add a total column
    summary['total'] = summary.sum(axis=1)
    
    return summary

def generate_effectiveness_report(results_dir, output_dir=None):
    """
    Generate a comprehensive effectiveness report comparing all models
    
    Args:
        results_dir: Directory containing evaluation results
        output_dir: Directory to save the report (defaults to results_dir)
        
    Returns:
        Path to the generated report
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_evaluation_results(results_dir)
    
    # Create comparison tables
    tables = create_comparison_tables(results)
    
    # Create summary DataFrame
    summary = create_summary_dataframe(tables)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"effectiveness_report_{timestamp}.md")
    
    with open(report_file, 'w') as f:
        f.write("# NILM Model Effectiveness Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This table shows how many times each model type was the best performer across all appliances for each metric:\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        # Create a pie chart of the total column
        plt.figure(figsize=(10, 8))
        plt.pie(summary['total'], labels=summary.index, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('Proportion of Best Performance Across All Metrics and Appliances')
        plt.axis('equal')
        pie_chart_path = os.path.join(output_dir, "best_model_distribution.png")
        plt.savefig(pie_chart_path)
        plt.close()
        
        f.write(f"![Best Model Distribution]({os.path.basename(pie_chart_path)})\n\n")
        
        # Add detailed tables for each metric
        f.write("## Detailed Metric Comparisons\n\n")
        
        for metric in tables:
            f.write(f"### {metric.upper()}\n\n")
            f.write(tables[metric].to_markdown())
            f.write("\n\n")
            
            # Create a heatmap for this metric
            plt.figure(figsize=(12, 8))
            sns.heatmap(tables[metric].drop(columns=['best_model']), annot=True, cmap='YlGnBu', fmt=".4f")
            plt.title(f'{metric.upper()} Comparison Across Models and Appliances')
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, f"{metric}_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            f.write(f"![{metric.upper()} Heatmap]({os.path.basename(heatmap_path)})\n\n")
        
        # Highlight the Liquid Neural Network performance
        f.write("## Liquid Neural Network Performance\n\n")
        
        liquid_models = [model for model in summary.index if 'liquid' in model.lower()]
        
        if liquid_models:
            f.write("### Comparison with Traditional Models\n\n")
            
            for liquid_model in liquid_models:
                f.write(f"#### {liquid_model}\n\n")
                
                # Compare where Liquid Network is better than others
                for metric in tables:
                    f.write(f"**{metric.upper()}**:\n\n")
                    
                    better_count = 0
                    total_comparisons = 0
                    
                    metric_table = tables[metric].drop(columns=['best_model'])
                    appliances = metric_table.index
                    
                    for appliance in appliances:
                        if liquid_model not in metric_table.columns or np.isnan(metric_table.at[appliance, liquid_model]):
                            continue
                            
                        liquid_value = metric_table.at[appliance, liquid_model]
                        other_models = [m for m in metric_table.columns if m != liquid_model and not np.isnan(metric_table.at[appliance, m])]
                        
                        for other_model in other_models:
                            other_value = metric_table.at[appliance, other_model]
                            total_comparisons += 1
                            
                            if metric in ['mae', 'rmse', 'nete']:  # Lower is better
                                if liquid_value < other_value:
                                    better_count += 1
                            else:  # Higher is better
                                if liquid_value > other_value:
                                    better_count += 1
                    
                    if total_comparisons > 0:
                        percentage = (better_count / total_comparisons) * 100
                        f.write(f"{liquid_model} outperforms other models in {better_count} out of {total_comparisons} comparisons ({percentage:.2f}%).\n\n")
                    else:
                        f.write("No valid comparisons available.\n\n")
        
        # Add conclusions
        f.write("## Conclusions\n\n")
        
        # Determine the overall best model
        overall_best = summary['total'].idxmax()
        f.write(f"Based on the analysis, the **{overall_best}** model shows the best overall performance across all metrics and appliances.\n\n")
        
        # Check if Liquid Neural Network is competitive
        liquid_models = [model for model in summary.index if 'liquid' in model.lower()]
        if liquid_models:
            for liquid_model in liquid_models:
                relative_performance = (summary.at[liquid_model, 'total'] / summary.at[overall_best, 'total']) * 100
                f.write(f"The {liquid_model} model shows {relative_performance:.2f}% of the best performance of the {overall_best} model.\n\n")
        
        # Highlight best models for different metrics
        f.write("### Best Models by Metric\n\n")
        
        for metric in tables:
            best_for_metric = summary[metric].idxmax()
            count = summary.at[best_for_metric, metric]
            total = len(tables[metric])
            f.write(f"- **{metric.upper()}**: {best_for_metric} (best in {count} out of {total} appliances, {(count/total)*100:.2f}%)\n")
        
        f.write("\n")
    
    print(f"Effectiveness report generated: {report_file}")
    return report_file

if __name__ == "__main__":
    results_dir = "results/evaluation_house1_20250320_120000"
    
    report_file = generate_effectiveness_report(results_dir)