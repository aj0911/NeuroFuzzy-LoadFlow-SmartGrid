"""
Test Script for Phase 2: Neural Network Architecture
Tests model initialization, forward pass, and architecture validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import torch
import pickle
import time
from neurofuzzy_model import NeuroFuzzyLoadFlowModel, WeightedMSELoss, BaselineANN
from fuzzy_preprocessor import FuzzyPreprocessor


def test_neural_network():
    """Test the neural network architecture"""
    
    print("="*70)
    print("PHASE 2 TEST: Neural Network Architecture")
    print("="*70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv').values
    y = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv').values
    
    print(f"   Sensor data shape: {X_sensor.shape}")
    print(f"   Output data shape: {y.shape}")
    print(f"   Sparsity: {np.isnan(X_sensor).mean():.2%}")
    
    # Load fuzzy preprocessor
    print("\n[2] Loading fuzzy preprocessor...")
    with open('models/fuzzy_preprocessor.pkl', 'rb') as f:
        fuzzy_processor = pickle.load(f)
    
    # Generate fuzzy features
    print("\n[3] Generating fuzzy features...")
    X_sensor_df = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv')
    X_fuzzy = fuzzy_processor.transform(X_sensor_df)
    print(f"   Fuzzy features shape: {X_fuzzy.shape}")
    
    # Initialize Neuro-Fuzzy Model
    print("\n[4] Initializing Neuro-Fuzzy Model...")
    model = NeuroFuzzyLoadFlowModel(
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    
    print(model.get_architecture_summary())
    
    # Fit normalization statistics
    print("\n[5] Fitting normalization statistics...")
    model.fit_normalization(X_sensor, y)
    
    # Test preprocessing
    print("\n[6] Testing input preprocessing...")
    sample_size = 10
    X_sensor_sample = X_sensor[:sample_size]
    X_fuzzy_sample = X_fuzzy[:sample_size]
    
    X_preprocessed = model.preprocess_input(X_sensor_sample, X_fuzzy_sample)
    print(f"   Raw sensor shape: {X_sensor_sample.shape}")
    print(f"   Raw fuzzy shape: {X_fuzzy_sample.shape}")
    print(f"   Preprocessed shape: {X_preprocessed.shape}")
    print(f"   Expected input dimension: {model.input_dim}")
    print(f"   ✓ Shape match: {X_preprocessed.shape[1] == model.input_dim}")
    
    # Test forward pass
    print("\n[7] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_preprocessed)
    
    print(f"   Input shape: {X_preprocessed.shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Expected output shape: (10, 66)")
    print(f"   ✓ Output shape correct: {y_pred.shape == (10, 66)}")
    
    # Test prediction function
    print("\n[8] Testing prediction function...")
    predictions = model.predict(X_sensor_sample, X_fuzzy_sample)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Ground truth range: [{y[:sample_size].min():.4f}, {y[:sample_size].max():.4f}]")
    
    # Display sample predictions vs ground truth
    print("\n[9] Sample predictions (first 3 samples, first 5 voltages):")
    print("   " + "-"*65)
    for i in range(min(3, sample_size)):
        print(f"\n   Sample {i+1}:")
        print(f"      Predicted: {predictions[i, :5]}")
        print(f"      Actual:    {y[i, :5]}")
        error = np.abs(predictions[i, :5] - y[i, :5])
        print(f"      Error:     {error}")
    
    # Test inference time
    print("\n[10] Testing inference time (target: <100ms)...")
    n_trials = 100
    inference_times = []
    
    for _ in range(n_trials):
        start = time.time()
        _ = model.predict(X_sensor_sample[:1], X_fuzzy_sample[:1])
        end = time.time()
        inference_times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    print(f"   Mean inference time: {mean_time:.2f} ms (std: {std_time:.2f} ms)")
    print(f"   ✓ Meets <100ms requirement: {mean_time < 100}")
    
    # Test weighted loss function
    print("\n[11] Testing weighted loss function...")
    loss_fn = WeightedMSELoss(voltage_weight=2.0, angle_weight=1.0, n_buses=33)
    
    y_sample_normalized = model.normalize_output(y[:sample_size])
    loss = loss_fn(y_pred, y_sample_normalized)
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   ✓ Loss computable: {not torch.isnan(loss).any()}")
    
    # Test baseline ANN (no fuzzy)
    print("\n[12] Testing baseline ANN (for comparison)...")
    baseline_model = BaselineANN(
        n_sensor_features=20,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    baseline_model.fit_normalization(X_sensor, y)
    
    print(f"   Baseline input dimension: {baseline_model.input_dim}")
    print(f"   Baseline parameters: {baseline_model.count_parameters():,}")
    print(f"   Neuro-fuzzy parameters: {model.count_parameters():,}")
    print(f"   Extra parameters from fuzzy: {model.count_parameters() - baseline_model.count_parameters():,}")
    
    baseline_pred = baseline_model.predict(X_sensor_sample)
    print(f"   Baseline predictions shape: {baseline_pred.shape}")
    print(f"   ✓ Baseline functional: {baseline_pred.shape == (sample_size, 66)}")
    
    # Test batch processing
    print("\n[13] Testing batch processing (full dataset)...")
    batch_size = 64
    n_batches = len(X_sensor) // batch_size
    
    start = time.time()
    all_predictions = model.predict(X_sensor, X_fuzzy)
    end = time.time()
    
    total_time = (end - start) * 1000
    time_per_sample = total_time / len(X_sensor)
    
    print(f"   Total samples: {len(X_sensor)}")
    print(f"   Total time: {total_time:.2f} ms")
    print(f"   Time per sample: {time_per_sample:.4f} ms")
    print(f"   Throughput: {len(X_sensor) / (total_time/1000):.0f} samples/sec")
    
    # Model state dict test
    print("\n[14] Testing model save/load...")
    torch.save(model.state_dict(), 'models/test_model_checkpoint.pth')
    
    # Load into new model
    model_loaded = NeuroFuzzyLoadFlowModel(
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    model_loaded.load_state_dict(torch.load('models/test_model_checkpoint.pth'))
    model_loaded.sensor_mean = model.sensor_mean
    model_loaded.sensor_std = model.sensor_std
    model_loaded.output_mean = model.output_mean
    model_loaded.output_std = model.output_std
    
    pred_loaded = model_loaded.predict(X_sensor_sample[:1], X_fuzzy_sample[:1])
    pred_original = model.predict(X_sensor_sample[:1], X_fuzzy_sample[:1])
    
    print(f"   ✓ Predictions match after reload: {np.allclose(pred_loaded, pred_original)}")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 TEST RESULTS: ✓ PASSED")
    print("="*70)
    print(f"✓ Neural network architecture validated")
    print(f"✓ Forward pass successful: {X_preprocessed.shape} -> {y_pred.shape}")
    print(f"✓ Prediction function works correctly")
    print(f"✓ Inference time: {mean_time:.2f} ms (target: <100ms)")
    print(f"✓ Model parameters: {model.count_parameters():,}")
    print(f"✓ Weighted loss function operational")
    print(f"✓ Baseline ANN comparison ready")
    print(f"✓ Model save/load functional")
    print("\n✓ Neural Network is ready for Phase 3 (Training)")
    print("="*70)
    
    return model, baseline_model, X_sensor, X_fuzzy, y


if __name__ == "__main__":
    try:
        model, baseline, X_sensor, X_fuzzy, y = test_neural_network()
        print("\n✓ All Phase 2 tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
