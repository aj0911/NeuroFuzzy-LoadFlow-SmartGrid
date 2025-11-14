"""
Evaluation Script for Neuro-Fuzzy Load Flow Estimation
Computes detailed metrics, comparisons, and visualizations
"""

import numpy as np
import pandas as pd
import torch
import pickle
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neurofuzzy_model import NeuroFuzzyLoadFlowModel, BaselineANN
from fuzzy_preprocessor import FuzzyPreprocessor


def load_model(checkpoint_path: str, model_class, **model_kwargs):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore normalization statistics
    model.sensor_mean = checkpoint['sensor_mean']
    model.sensor_std = checkpoint['sensor_std']
    model.output_mean = checkpoint['output_mean']
    model.output_std = checkpoint['output_std']
    
    model.eval()
    return model, checkpoint


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_buses: int = 33) -> Dict:
    """
    Compute comprehensive metrics for load flow predictions
    
    Args:
        y_true: Ground truth (n_samples, 66)
        y_pred: Predictions (n_samples, 66)
        n_buses: Number of buses (default 33)
    
    Returns:
        Dictionary of metrics
    """
    # Split into voltage and angle components
    V_true = y_true[:, :n_buses]
    V_pred = y_pred[:, :n_buses]
    
    theta_true = y_true[:, n_buses:]
    theta_pred = y_pred[:, n_buses:]
    
    metrics = {}
    
    # Overall metrics
    metrics['overall_mae'] = mean_absolute_error(y_true, y_pred)
    metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['overall_r2'] = r2_score(y_true, y_pred)
    
    # Voltage metrics
    metrics['voltage_mae'] = mean_absolute_error(V_true, V_pred)
    metrics['voltage_rmse'] = np.sqrt(mean_squared_error(V_true, V_pred))
    metrics['voltage_mape'] = np.mean(np.abs((V_true - V_pred) / V_true)) * 100
    metrics['voltage_max_error'] = np.max(np.abs(V_true - V_pred))
    metrics['voltage_r2'] = r2_score(V_true.flatten(), V_pred.flatten())
    
    # Angle metrics
    metrics['angle_mae'] = mean_absolute_error(theta_true, theta_pred)
    metrics['angle_rmse'] = np.sqrt(mean_squared_error(theta_true, theta_pred))
    metrics['angle_max_error'] = np.max(np.abs(theta_true - theta_pred))
    metrics['angle_r2'] = r2_score(theta_true.flatten(), theta_pred.flatten())
    
    # Per-bus metrics
    metrics['per_bus_voltage_mae'] = np.mean(np.abs(V_true - V_pred), axis=0)
    metrics['per_bus_voltage_rmse'] = np.sqrt(np.mean((V_true - V_pred)**2, axis=0))
    metrics['per_bus_angle_mae'] = np.mean(np.abs(theta_true - theta_pred), axis=0)
    metrics['per_bus_angle_rmse'] = np.sqrt(np.mean((theta_true - theta_pred)**2, axis=0))
    
    return metrics


def analyze_sparsity_impact(
    model,
    X_sensor: np.ndarray,
    X_fuzzy: np.ndarray,
    y: np.ndarray,
    is_neurofuzzy: bool = True,
    n_bins: int = 5
) -> Dict:
    """
    Analyze how prediction accuracy varies with data sparsity
    """
    # Calculate sparsity for each sample
    sparsity_levels = np.isnan(X_sensor).sum(axis=1) / X_sensor.shape[1]
    
    # Bin by sparsity
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    results = {
        'sparsity_bins': bin_centers,
        'voltage_mae': [],
        'angle_mae': [],
        'sample_counts': []
    }
    
    for i in range(n_bins):
        mask = (sparsity_levels >= bins[i]) & (sparsity_levels < bins[i+1])
        if mask.sum() == 0:
            results['voltage_mae'].append(np.nan)
            results['angle_mae'].append(np.nan)
            results['sample_counts'].append(0)
            continue
        
        X_sensor_bin = X_sensor[mask]
        y_bin = y[mask]
        
        if is_neurofuzzy:
            X_fuzzy_bin = X_fuzzy[mask]
            y_pred_bin = model.predict(X_sensor_bin, X_fuzzy_bin)
        else:
            y_pred_bin = model.predict(X_sensor_bin)
        
        metrics_bin = compute_metrics(y_bin, y_pred_bin)
        
        results['voltage_mae'].append(metrics_bin['voltage_mae'])
        results['angle_mae'].append(metrics_bin['angle_mae'])
        results['sample_counts'].append(mask.sum())
    
    return results


def benchmark_inference_time(
    model,
    X_sensor: np.ndarray,
    X_fuzzy: np.ndarray,
    is_neurofuzzy: bool = True,
    n_trials: int = 1000
) -> Dict:
    """Benchmark inference time"""
    times = []
    
    # Single sample inference
    for _ in range(n_trials):
        start = time.time()
        if is_neurofuzzy:
            _ = model.predict(X_sensor[:1], X_fuzzy[:1])
        else:
            _ = model.predict(X_sensor[:1])
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    results = {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'median_time_ms': np.median(times),
        'meets_requirement': np.mean(times) < 100
    }
    
    # Batch inference
    batch_sizes = [1, 10, 50, 100]
    batch_results = {}
    
    for batch_size in batch_sizes:
        batch_times = []
        for _ in range(100):
            start = time.time()
            if is_neurofuzzy:
                _ = model.predict(X_sensor[:batch_size], X_fuzzy[:batch_size])
            else:
                _ = model.predict(X_sensor[:batch_size])
            end = time.time()
            batch_times.append((end - start) * 1000 / batch_size)
        
        batch_results[f'batch_{batch_size}'] = np.mean(batch_times)
    
    results['batch_performance'] = batch_results
    
    return results


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred_nf: np.ndarray,
    y_pred_bl: np.ndarray,
    n_buses: int = 33,
    save_path: str = 'prediction_comparison.png'
):
    """Plot prediction vs ground truth comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Voltage predictions - Neuro-Fuzzy
    V_true = y_true[:, :n_buses].flatten()
    V_pred_nf = y_pred_nf[:, :n_buses].flatten()
    V_pred_bl = y_pred_bl[:, :n_buses].flatten()
    
    axes[0, 0].scatter(V_true, V_pred_nf, alpha=0.3, s=5, c='blue', label='Neuro-Fuzzy')
    axes[0, 0].plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('True Voltage (pu)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Voltage (pu)', fontsize=11)
    axes[0, 0].set_title('Neuro-Fuzzy: Voltage Predictions', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Voltage predictions - Baseline
    axes[0, 1].scatter(V_true, V_pred_bl, alpha=0.3, s=5, c='orange', label='Baseline ANN')
    axes[0, 1].plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Voltage (pu)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Voltage (pu)', fontsize=11)
    axes[0, 1].set_title('Baseline ANN: Voltage Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Angle predictions - Neuro-Fuzzy
    theta_true = y_true[:, n_buses:].flatten()
    theta_pred_nf = y_pred_nf[:, n_buses:].flatten()
    theta_pred_bl = y_pred_bl[:, n_buses:].flatten()
    
    axes[1, 0].scatter(theta_true, theta_pred_nf, alpha=0.3, s=5, c='blue', label='Neuro-Fuzzy')
    axes[1, 0].plot([theta_true.min(), theta_true.max()], [theta_true.min(), theta_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Angle (degrees)', fontsize=11)
    axes[1, 0].set_ylabel('Predicted Angle (degrees)', fontsize=11)
    axes[1, 0].set_title('Neuro-Fuzzy: Angle Predictions', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Angle predictions - Baseline
    axes[1, 1].scatter(theta_true, theta_pred_bl, alpha=0.3, s=5, c='orange', label='Baseline ANN')
    axes[1, 1].plot([theta_true.min(), theta_true.max()], [theta_true.min(), theta_true.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_xlabel('True Angle (degrees)', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Angle (degrees)', fontsize=11)
    axes[1, 1].set_title('Baseline ANN: Angle Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Comparison: Neuro-Fuzzy vs Baseline', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction comparison saved to {save_path}")
    plt.close()


def plot_error_analysis(
    metrics_nf: Dict,
    metrics_bl: Dict,
    n_buses: int = 33,
    save_path: str = 'error_analysis.png'
):
    """Plot detailed error analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bus_numbers = np.arange(1, n_buses + 1)
    
    # Voltage MAE per bus
    axes[0, 0].bar(bus_numbers - 0.2, metrics_nf['per_bus_voltage_mae'], 
                   width=0.4, label='Neuro-Fuzzy', alpha=0.8, color='blue')
    axes[0, 0].bar(bus_numbers + 0.2, metrics_bl['per_bus_voltage_mae'], 
                   width=0.4, label='Baseline', alpha=0.8, color='orange')
    axes[0, 0].set_xlabel('Bus Number', fontsize=11)
    axes[0, 0].set_ylabel('MAE (pu)', fontsize=11)
    axes[0, 0].set_title('Voltage MAE per Bus', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Voltage RMSE per bus
    axes[0, 1].bar(bus_numbers - 0.2, metrics_nf['per_bus_voltage_rmse'], 
                   width=0.4, label='Neuro-Fuzzy', alpha=0.8, color='blue')
    axes[0, 1].bar(bus_numbers + 0.2, metrics_bl['per_bus_voltage_rmse'], 
                   width=0.4, label='Baseline', alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Bus Number', fontsize=11)
    axes[0, 1].set_ylabel('RMSE (pu)', fontsize=11)
    axes[0, 1].set_title('Voltage RMSE per Bus', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Angle MAE per bus
    axes[1, 0].bar(bus_numbers - 0.2, metrics_nf['per_bus_angle_mae'], 
                   width=0.4, label='Neuro-Fuzzy', alpha=0.8, color='blue')
    axes[1, 0].bar(bus_numbers + 0.2, metrics_bl['per_bus_angle_mae'], 
                   width=0.4, label='Baseline', alpha=0.8, color='orange')
    axes[1, 0].set_xlabel('Bus Number', fontsize=11)
    axes[1, 0].set_ylabel('MAE (degrees)', fontsize=11)
    axes[1, 0].set_title('Angle MAE per Bus', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Angle RMSE per bus
    axes[1, 1].bar(bus_numbers - 0.2, metrics_nf['per_bus_angle_rmse'], 
                   width=0.4, label='Neuro-Fuzzy', alpha=0.8, color='blue')
    axes[1, 1].bar(bus_numbers + 0.2, metrics_bl['per_bus_angle_rmse'], 
                   width=0.4, label='Baseline', alpha=0.8, color='orange')
    axes[1, 1].set_xlabel('Bus Number', fontsize=11)
    axes[1, 1].set_ylabel('RMSE (degrees)', fontsize=11)
    axes[1, 1].set_title('Angle RMSE per Bus', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Per-Bus Error Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis saved to {save_path}")
    plt.close()


def plot_sparsity_impact(
    sparsity_nf: Dict,
    sparsity_bl: Dict,
    save_path: str = 'sparsity_impact.png'
):
    """Plot impact of data sparsity on accuracy"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sparsity_pct = sparsity_nf['sparsity_bins'] * 100
    
    # Voltage MAE vs Sparsity
    axes[0].plot(sparsity_pct, sparsity_nf['voltage_mae'], 
                 'o-', linewidth=2, markersize=8, label='Neuro-Fuzzy', color='blue')
    axes[0].plot(sparsity_pct, sparsity_bl['voltage_mae'], 
                 's-', linewidth=2, markersize=8, label='Baseline ANN', color='orange')
    axes[0].set_xlabel('Data Sparsity (%)', fontsize=11)
    axes[0].set_ylabel('Voltage MAE (pu)', fontsize=11)
    axes[0].set_title('Voltage Accuracy vs Data Sparsity', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Angle MAE vs Sparsity
    axes[1].plot(sparsity_pct, sparsity_nf['angle_mae'], 
                 'o-', linewidth=2, markersize=8, label='Neuro-Fuzzy', color='blue')
    axes[1].plot(sparsity_pct, sparsity_bl['angle_mae'], 
                 's-', linewidth=2, markersize=8, label='Baseline ANN', color='orange')
    axes[1].set_xlabel('Data Sparsity (%)', fontsize=11)
    axes[1].set_ylabel('Angle MAE (degrees)', fontsize=11)
    axes[1].set_title('Angle Accuracy vs Data Sparsity', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sparsity impact plot saved to {save_path}")
    plt.close()


def print_metrics_table(metrics_nf: Dict, metrics_bl: Dict):
    """Print formatted metrics comparison table"""
    
    print("\n" + "="*70)
    print("METRICS COMPARISON: Neuro-Fuzzy vs Baseline ANN")
    print("="*70)
    
    print("\nOVERALL METRICS:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Neuro-Fuzzy':>15} {'Baseline ANN':>15} {'Improvement':>13}")
    print("-" * 70)
    
    overall_metrics = [
        ('Overall MAE', 'overall_mae'),
        ('Overall RMSE', 'overall_rmse'),
        ('Overall R²', 'overall_r2'),
    ]
    
    for name, key in overall_metrics:
        nf_val = metrics_nf[key]
        bl_val = metrics_bl[key]
        if 'r2' in key:
            improvement = ((nf_val - bl_val) / abs(bl_val)) * 100
        else:
            improvement = ((bl_val - nf_val) / bl_val) * 100
        
        print(f"{name:<25} {nf_val:>15.6f} {bl_val:>15.6f} {improvement:>12.2f}%")
    
    print("\nVOLTAGE METRICS:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Neuro-Fuzzy':>15} {'Baseline ANN':>15} {'Improvement':>13}")
    print("-" * 70)
    
    voltage_metrics = [
        ('Voltage MAE (pu)', 'voltage_mae'),
        ('Voltage RMSE (pu)', 'voltage_rmse'),
        ('Voltage MAPE (%)', 'voltage_mape'),
        ('Voltage Max Error (pu)', 'voltage_max_error'),
        ('Voltage R²', 'voltage_r2'),
    ]
    
    for name, key in voltage_metrics:
        nf_val = metrics_nf[key]
        bl_val = metrics_bl[key]
        if 'r2' in key:
            improvement = ((nf_val - bl_val) / abs(bl_val)) * 100
        else:
            improvement = ((bl_val - nf_val) / bl_val) * 100
        
        print(f"{name:<25} {nf_val:>15.6f} {bl_val:>15.6f} {improvement:>12.2f}%")
    
    print("\nANGLE METRICS:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Neuro-Fuzzy':>15} {'Baseline ANN':>15} {'Improvement':>13}")
    print("-" * 70)
    
    angle_metrics = [
        ('Angle MAE (degrees)', 'angle_mae'),
        ('Angle RMSE (degrees)', 'angle_rmse'),
        ('Angle Max Error (deg)', 'angle_max_error'),
        ('Angle R²', 'angle_r2'),
    ]
    
    for name, key in angle_metrics:
        nf_val = metrics_nf[key]
        bl_val = metrics_bl[key]
        if 'r2' in key:
            improvement = ((nf_val - bl_val) / abs(bl_val)) * 100
        else:
            improvement = ((bl_val - nf_val) / bl_val) * 100
        
        print(f"{name:<25} {nf_val:>15.6f} {bl_val:>15.6f} {improvement:>12.2f}%")
    
    print("="*70)


def main():
    """Main evaluation function"""
    
    print("="*70)
    print("PHASE 4: EVALUATION & METRICS")
    print("="*70)
    
    # Load data
    print("\n[1] Loading dataset...")
    X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv').values
    y = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv').values
    
    # Load fuzzy preprocessor
    print("\n[2] Loading fuzzy preprocessor...")
    with open('models/fuzzy_preprocessor.pkl', 'rb') as f:
        fuzzy_processor = pickle.load(f)
    
    X_sensor_df = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv')
    X_fuzzy = fuzzy_processor.transform(X_sensor_df)
    
    # Use validation set (last 20%)
    n_train = int(len(X_sensor) * 0.8)
    np.random.seed(42)
    indices = np.random.permutation(len(X_sensor))
    val_idx = indices[n_train:]
    
    X_sensor_val = X_sensor[val_idx]
    X_fuzzy_val = X_fuzzy[val_idx]
    y_val = y[val_idx]
    
    print(f"   Validation samples: {len(X_sensor_val)}")
    
    # Load trained models
    print("\n[3] Loading trained models...")
    neurofuzzy_model, nf_checkpoint = load_model(
        'models/checkpoints/neurofuzzy_best.pth',
        NeuroFuzzyLoadFlowModel,
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    print(f"   Neuro-Fuzzy loaded (epoch {nf_checkpoint['epoch']}, val_loss: {nf_checkpoint['best_val_loss']:.6f})")
    
    baseline_model, bl_checkpoint = load_model(
        'models/checkpoints/baseline_best.pth',
        BaselineANN,
        n_sensor_features=20,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    print(f"   Baseline loaded (epoch {bl_checkpoint['epoch']}, val_loss: {bl_checkpoint['best_val_loss']:.6f})")
    
    # Generate predictions
    print("\n[4] Generating predictions...")
    y_pred_nf = neurofuzzy_model.predict(X_sensor_val, X_fuzzy_val)
    y_pred_bl = baseline_model.predict(X_sensor_val)
    print(f"   Predictions generated for {len(y_val)} samples")
    
    # Compute metrics
    print("\n[5] Computing detailed metrics...")
    metrics_nf = compute_metrics(y_val, y_pred_nf)
    metrics_bl = compute_metrics(y_val, y_pred_bl)
    
    # Print metrics table
    print_metrics_table(metrics_nf, metrics_bl)
    
    # Sparsity impact analysis
    print("\n[6] Analyzing sparsity impact...")
    sparsity_nf = analyze_sparsity_impact(neurofuzzy_model, X_sensor_val, X_fuzzy_val, y_val, is_neurofuzzy=True)
    sparsity_bl = analyze_sparsity_impact(baseline_model, X_sensor_val, X_fuzzy_val, y_val, is_neurofuzzy=False)
    print(f"   Sparsity analysis complete ({len(sparsity_nf['sparsity_bins'])} bins)")
    
    # Inference time benchmarking
    print("\n[7] Benchmarking inference time...")
    timing_nf = benchmark_inference_time(neurofuzzy_model, X_sensor_val, X_fuzzy_val, is_neurofuzzy=True)
    timing_bl = benchmark_inference_time(baseline_model, X_sensor_val, X_fuzzy_val, is_neurofuzzy=False)
    
    print(f"\n   Neuro-Fuzzy Inference Time:")
    print(f"     Mean: {timing_nf['mean_time_ms']:.4f} ms")
    print(f"     Std:  {timing_nf['std_time_ms']:.4f} ms")
    print(f"     ✓ Meets <100ms: {timing_nf['meets_requirement']}")
    
    print(f"\n   Baseline Inference Time:")
    print(f"     Mean: {timing_bl['mean_time_ms']:.4f} ms")
    print(f"     Std:  {timing_bl['std_time_ms']:.4f} ms")
    print(f"     ✓ Meets <100ms: {timing_bl['meets_requirement']}")
    
    # Generate visualizations
    print("\n[8] Generating visualizations...")
    plot_prediction_comparison(y_val, y_pred_nf, y_pred_bl, save_path='results/prediction_comparison.png')
    plot_error_analysis(metrics_nf, metrics_bl, save_path='results/error_analysis.png')
    plot_sparsity_impact(sparsity_nf, sparsity_bl, save_path='results/sparsity_impact.png')
    
    # Save evaluation results
    print("\n[9] Saving evaluation results...")
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = {
        'metrics_neurofuzzy': convert_to_serializable(metrics_nf),
        'metrics_baseline': convert_to_serializable(metrics_bl),
        'sparsity_analysis_neurofuzzy': convert_to_serializable(sparsity_nf),
        'sparsity_analysis_baseline': convert_to_serializable(sparsity_bl),
        'timing_neurofuzzy': convert_to_serializable(timing_nf),
        'timing_baseline': convert_to_serializable(timing_bl),
        'validation_samples': int(len(y_val))
    }
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   Results saved to 'results/evaluation_results.json'")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 4 EVALUATION SUMMARY")
    print("="*70)
    print(f"\n✓ Evaluated on {len(y_val)} validation samples")
    print(f"✓ Neuro-Fuzzy Voltage MAE: {metrics_nf['voltage_mae']:.6f} pu")
    print(f"✓ Neuro-Fuzzy Angle MAE: {metrics_nf['angle_mae']:.6f} degrees")
    print(f"✓ Improvement over baseline: {((metrics_bl['voltage_mae'] - metrics_nf['voltage_mae']) / metrics_bl['voltage_mae'] * 100):.2f}%")
    print(f"✓ Inference time: {timing_nf['mean_time_ms']:.4f} ms (Target: <100ms)")
    print(f"\n✓ Visualizations saved:")
    print(f"  - results/prediction_comparison.png")
    print(f"  - results/error_analysis.png")
    print(f"  - results/sparsity_impact.png")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ EVALUATION FAILED with error: {e}")
        import traceback
        traceback.print_exc()
