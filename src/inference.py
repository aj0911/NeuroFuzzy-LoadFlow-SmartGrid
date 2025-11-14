"""
Real-Time Inference Script for Neuro-Fuzzy Load Flow Estimation
Production-ready script for predicting grid states from sparse sensor data
"""

import numpy as np
import pandas as pd
import torch
import pickle
import argparse
import time
import json
from pathlib import Path
from typing import Union, Dict, Tuple

from neurofuzzy_model import NeuroFuzzyLoadFlowModel
from fuzzy_preprocessor import FuzzyPreprocessor


class LoadFlowPredictor:
    """
    Production-ready predictor for load flow estimation
    Handles loading models and making real-time predictions
    """
    
    def __init__(
        self, 
        model_path: str = 'models/checkpoints/neurofuzzy_best.pth',
        fuzzy_processor_path: str = 'models/fuzzy_preprocessor.pkl',
        device: str = 'cpu'
    ):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model checkpoint
            fuzzy_processor_path: Path to fuzzy preprocessor
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load fuzzy preprocessor
        print(f"Loading fuzzy preprocessor from {fuzzy_processor_path}...")
        with open(fuzzy_processor_path, 'rb') as f:
            self.fuzzy_processor = pickle.load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = NeuroFuzzyLoadFlowModel(
            n_sensor_features=20,
            n_fuzzy_features=12,
            n_outputs=66,
            hidden_dims=[128, 256, 128],
            dropout_rate=0.2
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore normalization statistics
        self.model.sensor_mean = checkpoint['sensor_mean']
        self.model.sensor_std = checkpoint['sensor_std']
        self.model.output_mean = checkpoint['output_mean']
        self.model.output_std = checkpoint['output_std']
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
    
    def predict_from_array(
        self, 
        sensor_measurements: np.ndarray,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Predict grid state from sensor measurements
        
        Args:
            sensor_measurements: Sensor data array (n_samples, 20) with NaN for missing values
            return_confidence: Whether to return fuzzy confidence scores
        
        Returns:
            predictions: Grid state predictions (n_samples, 66)
            confidence (optional): Dictionary with confidence metrics
        """
        start_time = time.time()
        
        # Convert to DataFrame for fuzzy processing
        sensor_df = pd.DataFrame(
            sensor_measurements, 
            columns=[f'meas_{i}' for i in range(20)]
        )
        
        # Generate fuzzy features
        fuzzy_features = self.fuzzy_processor.transform(sensor_df)
        
        # Make predictions
        predictions = self.model.predict(sensor_measurements, fuzzy_features)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if return_confidence:
            # Extract confidence metrics from fuzzy features
            confidence = {
                'availability': fuzzy_features[:, 0],  # Data availability score
                'confidence': fuzzy_features[:, 1],     # Fuzzy confidence
                'quality': fuzzy_features[:, 2],        # Data quality score
                'inference_time_ms': inference_time,
                'sparsity_pct': np.isnan(sensor_measurements).mean(axis=1) * 100
            }
            return predictions, confidence
        
        return predictions
    
    def predict_from_csv(
        self, 
        csv_path: str,
        output_path: str = None,
        save_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Predict grid states from CSV file of sensor measurements
        
        Args:
            csv_path: Path to CSV file with sensor measurements
            output_path: Path to save predictions (optional)
            save_confidence: Whether to include confidence scores
        
        Returns:
            DataFrame with predictions
        """
        print(f"\nReading sensor data from {csv_path}...")
        sensor_data = pd.read_csv(csv_path)
        
        print(f"Processing {len(sensor_data)} samples...")
        sensor_array = sensor_data.values
        
        # Make predictions
        predictions, confidence = self.predict_from_array(
            sensor_array, 
            return_confidence=True
        )
        
        # Create output DataFrame
        n_buses = 33
        columns = [f'V_{i}' for i in range(n_buses)] + [f'theta_{i}' for i in range(n_buses)]
        results_df = pd.DataFrame(predictions, columns=columns)
        
        if save_confidence:
            results_df['data_availability'] = confidence['availability']
            results_df['confidence_score'] = confidence['confidence']
            results_df['quality_score'] = confidence['quality']
            results_df['sparsity_pct'] = confidence['sparsity_pct']
        
        # Save to file
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        print(f"Inference completed in {confidence['inference_time_ms']:.2f} ms")
        print(f"Average inference time per sample: {confidence['inference_time_ms']/len(sensor_data):.4f} ms")
        
        return results_df
    
    def predict_single(
        self, 
        sensor_dict: Dict[str, float],
        verbose: bool = True
    ) -> Dict:
        """
        Predict grid state from a single set of sensor measurements
        
        Args:
            sensor_dict: Dictionary of sensor measurements (key: sensor_name, value: measurement)
                        Use np.nan or None for missing values
            verbose: Whether to print detailed output
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Convert dict to array
        sensor_array = np.full((1, 20), np.nan)
        
        for key, value in sensor_dict.items():
            if key.startswith('meas_'):
                idx = int(key.split('_')[1])
                if idx < 20:
                    sensor_array[0, idx] = value if value is not None else np.nan
        
        # Make prediction
        predictions, confidence = self.predict_from_array(sensor_array, return_confidence=True)
        
        # Format output
        n_buses = 33
        result = {
            'voltages': {f'bus_{i}': float(predictions[0, i]) for i in range(n_buses)},
            'angles': {f'bus_{i}': float(predictions[0, n_buses + i]) for i in range(n_buses)},
            'metadata': {
                'data_availability': float(confidence['availability'][0]),
                'confidence_score': float(confidence['confidence'][0]),
                'quality_score': float(confidence['quality'][0]),
                'sparsity_pct': float(confidence['sparsity_pct'][0]),
                'inference_time_ms': float(confidence['inference_time_ms'])
            }
        }
        
        if verbose:
            print("\n" + "="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"\nMetadata:")
            print(f"  Data Availability: {result['metadata']['data_availability']:.2%}")
            print(f"  Confidence Score: {result['metadata']['confidence_score']:.2%}")
            print(f"  Quality Score: {result['metadata']['quality_score']:.2%}")
            print(f"  Sparsity: {result['metadata']['sparsity_pct']:.2f}%")
            print(f"  Inference Time: {result['metadata']['inference_time_ms']:.4f} ms")
            
            print(f"\nVoltages (first 5 buses):")
            for i in range(min(5, n_buses)):
                print(f"  Bus {i}: {result['voltages'][f'bus_{i}']:.6f} pu")
            
            print(f"\nAngles (first 5 buses):")
            for i in range(min(5, n_buses)):
                print(f"  Bus {i}: {result['angles'][f'bus_{i}']:.6f} degrees")
            print("="*70)
        
        return result


def export_to_onnx(
    model_path: str = 'models/checkpoints/neurofuzzy_best.pth',
    output_path: str = 'models/neurofuzzy_model.onnx',
    opset_version: int = 12
):
    """
    Export trained model to ONNX format for deployment
    
    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    print(f"\nExporting model to ONNX format...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = NeuroFuzzyLoadFlowModel(
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input (preprocessed format)
    dummy_input = torch.randn(1, 52)  # 20 sensor + 20 mask + 12 fuzzy
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to {output_path}")
    print(f"  Input shape: (batch_size, 52)")
    print(f"  Output shape: (batch_size, 66)")
    
    # Save normalization statistics separately
    stats_path = output_path.replace('.onnx', '_stats.pkl')
    stats = {
        'sensor_mean': checkpoint['sensor_mean'],
        'sensor_std': checkpoint['sensor_std'],
        'output_mean': checkpoint['output_mean'],
        'output_std': checkpoint['output_std']
    }
    
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"✓ Normalization statistics saved to {stats_path}")


def main():
    """Main inference function with CLI"""
    
    parser = argparse.ArgumentParser(
        description='Real-Time Load Flow Estimation Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from CSV file
  python inference.py --input sensor_data.csv --output predictions.csv
  
  # Export model to ONNX
  python inference.py --export-onnx
  
  # Interactive mode (test with validation data)
  python inference.py --demo
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file with sensor measurements'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save predictions CSV file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/neurofuzzy_best.pth',
        help='Path to model checkpoint (default: models/checkpoints/neurofuzzy_best.pth)'
    )
    
    parser.add_argument(
        '--fuzzy-processor',
        type=str,
        default='models/fuzzy_preprocessor.pkl',
        help='Path to fuzzy preprocessor (default: models/fuzzy_preprocessor.pkl)'
    )
    
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='Export model to ONNX format'
    )
    
    parser.add_argument(
        '--onnx-path',
        type=str,
        default='models/neurofuzzy_model.onnx',
        help='Path to save ONNX model (default: models/neurofuzzy_model.onnx)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with validation data'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Export to ONNX
    if args.export_onnx:
        export_to_onnx(args.model, args.onnx_path)
        return
    
    # Initialize predictor
    predictor = LoadFlowPredictor(
        model_path=args.model,
        fuzzy_processor_path=args.fuzzy_processor,
        device=args.device
    )
    
    # Demo mode
    if args.demo:
        print("\n" + "="*70)
        print("DEMO MODE: Testing with validation data")
        print("="*70)
        
        # Load validation data
        X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv').values
        y_true = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv').values
        
        # Use last 20% as validation
        n_train = int(len(X_sensor) * 0.8)
        np.random.seed(42)
        indices = np.random.permutation(len(X_sensor))
        val_idx = indices[n_train:]
        
        X_val = X_sensor[val_idx][:5]  # First 5 validation samples
        y_val = y_true[val_idx][:5]
        
        # Make predictions
        predictions = predictor.predict_from_array(X_val)
        
        # Show results
        print(f"\nComparing predictions with ground truth (first 5 samples):")
        print("-"*70)
        
        for i in range(5):
            voltage_error = np.abs(predictions[i, :33] - y_val[i, :33]).mean()
            angle_error = np.abs(predictions[i, 33:] - y_val[i, 33:]).mean()
            sparsity = np.isnan(X_val[i]).sum() / 20 * 100
            
            print(f"\nSample {i+1} (Sparsity: {sparsity:.1f}%):")
            print(f"  Voltage MAE: {voltage_error:.6f} pu")
            print(f"  Angle MAE: {angle_error:.6f} degrees")
        
        return
    
    # CSV file prediction
    if args.input:
        if not args.output:
            args.output = args.input.replace('.csv', '_predictions.csv')
        
        predictor.predict_from_csv(args.input, args.output)
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
