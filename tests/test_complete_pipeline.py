"""
Complete Pipeline Test - End-to-End Validation
Tests the entire neuro-fuzzy load flow estimation system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import pickle
import time

from inference import LoadFlowPredictor


def test_complete_pipeline():
    """Test the complete neuro-fuzzy pipeline end-to-end"""
    
    print("="*70)
    print("COMPLETE PIPELINE TEST - End-to-End Validation")
    print("="*70)
    
    results = {
        'tests_passed': 0,
        'tests_failed': 0,
        'test_details': []
    }
    
    # Test 1: Check all required files exist
    print("\n[Test 1] Checking required files...")
    required_files = [
        'models/fuzzy_preprocessor.pkl',
        'models/checkpoints/neurofuzzy_best.pth',
        'models/checkpoints/baseline_best.pth',
        'models/neurofuzzy_model.onnx',
        'models/neurofuzzy_model_stats.pkl',
        'output_generation/sensor_inputs_ieee_33-bus.csv',
        'output_generation/grid_states_ieee_33-bus.csv'
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        if not exists:
            print(f"  ✗ Missing: {file}")
            all_exist = False
    
    if all_exist:
        print("  ✓ All required files present")
        results['tests_passed'] += 1
        results['test_details'].append({'name': 'File Check', 'status': 'PASS'})
    else:
        print("  ✗ Some files missing")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'File Check', 'status': 'FAIL'})
    
    # Test 2: Initialize predictor
    print("\n[Test 2] Initializing predictor...")
    try:
        predictor = LoadFlowPredictor(
            model_path='models/checkpoints/neurofuzzy_best.pth',
            fuzzy_processor_path='models/fuzzy_preprocessor.pkl'
        )
        print("  ✓ Predictor initialized successfully")
        results['tests_passed'] += 1
        results['test_details'].append({'name': 'Predictor Init', 'status': 'PASS'})
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Predictor Init', 'status': 'FAIL'})
        return results
    
    # Test 3: Load validation data
    print("\n[Test 3] Loading validation data...")
    try:
        X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv').values
        y_true = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv').values
        
        # Use validation split
        n_train = int(len(X_sensor) * 0.8)
        np.random.seed(42)
        indices = np.random.permutation(len(X_sensor))
        val_idx = indices[n_train:]
        
        X_val = X_sensor[val_idx][:100]  # Test on 100 samples
        y_val = y_true[val_idx][:100]
        
        print(f"  ✓ Loaded {len(X_val)} validation samples")
        results['tests_passed'] += 1
        results['test_details'].append({'name': 'Data Loading', 'status': 'PASS'})
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Data Loading', 'status': 'FAIL'})
        return results
    
    # Test 4: Make predictions
    print("\n[Test 4] Making predictions...")
    try:
        start_time = time.time()
        predictions, confidence = predictor.predict_from_array(X_val, return_confidence=True)
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"  ✓ Predictions generated")
        print(f"    Shape: {predictions.shape}")
        print(f"    Time: {elapsed_time:.2f} ms for {len(X_val)} samples")
        print(f"    Per-sample: {elapsed_time/len(X_val):.4f} ms")
        results['tests_passed'] += 1
        results['test_details'].append({'name': 'Prediction', 'status': 'PASS'})
    except Exception as e:
        print(f"  ✗ Failed to predict: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Prediction', 'status': 'FAIL'})
        return results
    
    # Test 5: Validate prediction accuracy
    print("\n[Test 5] Validating prediction accuracy...")
    try:
        # Compute errors
        voltage_mae = np.abs(predictions[:, :33] - y_val[:, :33]).mean()
        angle_mae = np.abs(predictions[:, 33:] - y_val[:, 33:]).mean()
        
        # Expected thresholds (based on validation results)
        voltage_threshold = 0.001  # 0.001 pu
        angle_threshold = 0.01     # 0.01 degrees
        
        voltage_pass = voltage_mae < voltage_threshold
        angle_pass = angle_mae < angle_threshold
        
        print(f"  Voltage MAE: {voltage_mae:.6f} pu (threshold: {voltage_threshold})")
        print(f"  Angle MAE: {angle_mae:.6f} degrees (threshold: {angle_threshold})")
        
        if voltage_pass and angle_pass:
            print("  ✓ Accuracy within acceptable range")
            results['tests_passed'] += 1
            results['test_details'].append({'name': 'Accuracy', 'status': 'PASS'})
        else:
            print("  ✗ Accuracy below threshold")
            results['tests_failed'] += 1
            results['test_details'].append({'name': 'Accuracy', 'status': 'FAIL'})
    except Exception as e:
        print(f"  ✗ Failed to validate: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Accuracy', 'status': 'FAIL'})
    
    # Test 6: Check inference time requirement
    print("\n[Test 6] Checking inference time requirement (<100ms)...")
    try:
        inference_times = []
        for _ in range(100):
            start = time.time()
            _ = predictor.predict_from_array(X_val[:1], return_confidence=False)
            inference_times.append((time.time() - start) * 1000)
        
        mean_time = np.mean(inference_times)
        meets_requirement = mean_time < 100
        
        print(f"  Mean inference time: {mean_time:.4f} ms")
        print(f"  Requirement: <100 ms")
        
        if meets_requirement:
            print(f"  ✓ Meets real-time requirement ({mean_time:.4f} ms < 100 ms)")
            results['tests_passed'] += 1
            results['test_details'].append({'name': 'Inference Time', 'status': 'PASS'})
        else:
            print(f"  ✗ Does not meet requirement ({mean_time:.4f} ms >= 100 ms)")
            results['tests_failed'] += 1
            results['test_details'].append({'name': 'Inference Time', 'status': 'FAIL'})
    except Exception as e:
        print(f"  ✗ Failed to benchmark: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Inference Time', 'status': 'FAIL'})
    
    # Test 7: Check confidence scores
    print("\n[Test 7] Validating confidence scores...")
    try:
        avg_availability = confidence['availability'].mean()
        avg_confidence = confidence['confidence'].mean()
        avg_quality = confidence['quality'].mean()
        
        # Check if scores are in valid range [0, 1]
        valid_range = (
            0 <= avg_availability <= 1 and
            0 <= avg_confidence <= 1 and
            0 <= avg_quality <= 1
        )
        
        print(f"  Average availability: {avg_availability:.4f}")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Average quality: {avg_quality:.4f}")
        
        if valid_range:
            print("  ✓ Confidence scores in valid range [0, 1]")
            results['tests_passed'] += 1
            results['test_details'].append({'name': 'Confidence Scores', 'status': 'PASS'})
        else:
            print("  ✗ Confidence scores out of range")
            results['tests_failed'] += 1
            results['test_details'].append({'name': 'Confidence Scores', 'status': 'FAIL'})
    except Exception as e:
        print(f"  ✗ Failed to validate confidence: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Confidence Scores', 'status': 'FAIL'})
    
    # Test 8: Test with varying sparsity levels
    print("\n[Test 8] Testing robustness across sparsity levels...")
    try:
        sparsity_ranges = [(0, 30), (30, 60), (60, 100)]
        sparsity_results = []
        
        for low, high in sparsity_ranges:
            # Filter samples by sparsity
            sparsity_pct = np.isnan(X_val).sum(axis=1) / X_val.shape[1] * 100
            mask = (sparsity_pct >= low) & (sparsity_pct < high)
            
            if mask.sum() > 0:
                X_sparse = X_val[mask]
                y_sparse = y_val[mask]
                
                pred_sparse = predictor.predict_from_array(X_sparse, return_confidence=False)
                mae_sparse = np.abs(pred_sparse - y_sparse).mean()
                
                sparsity_results.append({
                    'range': f"{low}-{high}%",
                    'samples': mask.sum(),
                    'mae': mae_sparse
                })
                
                print(f"  Sparsity {low}-{high}%: MAE={mae_sparse:.6f} ({mask.sum()} samples)")
        
        if len(sparsity_results) > 0:
            print("  ✓ Successfully tested across sparsity levels")
            results['tests_passed'] += 1
            results['test_details'].append({'name': 'Sparsity Robustness', 'status': 'PASS'})
        else:
            print("  ⚠ Not enough samples across sparsity levels")
            results['test_details'].append({'name': 'Sparsity Robustness', 'status': 'SKIP'})
    except Exception as e:
        print(f"  ✗ Failed to test sparsity: {e}")
        results['tests_failed'] += 1
        results['test_details'].append({'name': 'Sparsity Robustness', 'status': 'FAIL'})
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    
    total_tests = results['tests_passed'] + results['tests_failed']
    pass_rate = (results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTests Passed: {results['tests_passed']}/{total_tests} ({pass_rate:.1f}%)")
    print(f"Tests Failed: {results['tests_failed']}/{total_tests}")
    
    print("\nDetailed Results:")
    for test in results['test_details']:
        status_symbol = "✓" if test['status'] == 'PASS' else "✗" if test['status'] == 'FAIL' else "⚠"
        print(f"  {status_symbol} {test['name']}: {test['status']}")
    
    if results['tests_failed'] == 0:
        print("\n🎉 ALL TESTS PASSED - Pipeline is production-ready!")
    else:
        print(f"\n⚠ {results['tests_failed']} test(s) failed - Review required")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    try:
        results = test_complete_pipeline()
        
        # Save test results
        import json
        with open('results/pipeline_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Test results saved to 'results/pipeline_test_results.json'")
        
    except Exception as e:
        print(f"\n✗ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
