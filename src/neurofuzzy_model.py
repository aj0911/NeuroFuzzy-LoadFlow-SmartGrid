"""
Neuro-Fuzzy Model for Load Flow Estimation
Combines fuzzy preprocessing with deep neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class NeuroFuzzyLoadFlowModel(nn.Module):
    """
    Hybrid Neuro-Fuzzy Model for Load Flow Estimation
    
    Architecture:
    - Input: Sparse sensor measurements (20) + Fuzzy features (12) = 32 total
    - Hidden: 128 -> 256 -> 128 neurons with ReLU + Dropout
    - Output: 66 grid states (33 voltages + 33 angles)
    """
    
    def __init__(
        self, 
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    ):
        super(NeuroFuzzyLoadFlowModel, self).__init__()
        
        self.n_sensor_features = n_sensor_features
        self.n_fuzzy_features = n_fuzzy_features
        self.n_outputs = n_outputs
        
        # Input dimension: sensor features + binary mask + fuzzy features
        self.input_dim = n_sensor_features + n_sensor_features + n_fuzzy_features
        
        # Build neural network layers
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # Add dropout to all but last hidden layer
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (linear activation for regression)
        layers.append(nn.Linear(prev_dim, n_outputs))
        
        self.network = nn.Sequential(*layers)
        
        # Statistics for normalization (to be fitted)
        self.sensor_mean = None
        self.sensor_std = None
        self.output_mean = None
        self.output_std = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def fit_normalization(self, X_sensor: np.ndarray, y: np.ndarray):
        """
        Fit normalization statistics on training data
        
        Args:
            X_sensor: Sensor data (n_samples, n_sensor_features) with NaN for missing
            y: Output labels (n_samples, n_outputs)
        """
        # Compute sensor statistics (ignoring NaN)
        self.sensor_mean = np.nanmean(X_sensor, axis=0)
        self.sensor_std = np.nanstd(X_sensor, axis=0) + 1e-8
        
        # Compute output statistics
        self.output_mean = np.mean(y, axis=0)
        self.output_std = np.std(y, axis=0) + 1e-8
        
        print(f"Normalization statistics fitted:")
        print(f"  Sensor mean range: [{np.min(self.sensor_mean):.4f}, {np.max(self.sensor_mean):.4f}]")
        print(f"  Sensor std range: [{np.min(self.sensor_std):.4f}, {np.max(self.sensor_std):.4f}]")
        print(f"  Output mean range: [{np.min(self.output_mean):.4f}, {np.max(self.output_mean):.4f}]")
        print(f"  Output std range: [{np.min(self.output_std):.4f}, {np.max(self.output_std):.4f}]")
    
    def preprocess_input(self, X_sensor: np.ndarray, X_fuzzy: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data: impute missing values, create masks, normalize
        
        Args:
            X_sensor: Sensor data (n_samples, n_sensor_features) with NaN for missing
            X_fuzzy: Fuzzy features (n_samples, n_fuzzy_features)
        
        Returns:
            Preprocessed tensor ready for forward pass
        """
        if self.sensor_mean is None or self.sensor_std is None:
            raise ValueError("Model not fitted. Call fit_normalization() first.")
        
        # Create binary mask (1 = present, 0 = missing)
        mask = (~np.isnan(X_sensor)).astype(np.float32)
        
        # Impute missing values with mean
        X_sensor_imputed = X_sensor.copy()
        for i in range(X_sensor.shape[1]):
            X_sensor_imputed[np.isnan(X_sensor[:, i]), i] = self.sensor_mean[i]
        
        # Normalize sensor data
        X_sensor_normalized = (X_sensor_imputed - self.sensor_mean) / self.sensor_std
        
        # Concatenate: [sensor_normalized, mask, fuzzy_features]
        X_combined = np.concatenate([X_sensor_normalized, mask, X_fuzzy], axis=1)
        
        return torch.FloatTensor(X_combined)
    
    def normalize_output(self, y: np.ndarray) -> torch.Tensor:
        """Normalize output labels"""
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Model not fitted. Call fit_normalization() first.")
        
        y_normalized = (y - self.output_mean) / self.output_std
        return torch.FloatTensor(y_normalized)
    
    def denormalize_output(self, y_normalized: torch.Tensor) -> np.ndarray:
        """Denormalize network predictions"""
        if self.output_mean is None or self.output_std is None:
            raise ValueError("Model not fitted. Call fit_normalization() first.")
        
        if isinstance(y_normalized, torch.Tensor):
            y_normalized = y_normalized.detach().cpu().numpy()
        
        return y_normalized * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Preprocessed input tensor (batch_size, input_dim)
        
        Returns:
            Network output (batch_size, n_outputs)
        """
        return self.network(x)
    
    def predict(self, X_sensor: np.ndarray, X_fuzzy: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X_sensor: Sensor data (n_samples, n_sensor_features)
            X_fuzzy: Fuzzy features (n_samples, n_fuzzy_features)
        
        Returns:
            Predictions (n_samples, n_outputs) in original scale
        """
        self.eval()
        with torch.no_grad():
            X = self.preprocess_input(X_sensor, X_fuzzy)
            y_pred_normalized = self.forward(X)
            y_pred = self.denormalize_output(y_pred_normalized)
        return y_pred
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> str:
        """Get a string summary of the architecture"""
        summary = []
        summary.append("="*70)
        summary.append("Neuro-Fuzzy Load Flow Model Architecture")
        summary.append("="*70)
        summary.append(f"Input Features:")
        summary.append(f"  - Sensor features: {self.n_sensor_features}")
        summary.append(f"  - Binary mask: {self.n_sensor_features}")
        summary.append(f"  - Fuzzy features: {self.n_fuzzy_features}")
        summary.append(f"  - Total input dimension: {self.input_dim}")
        summary.append(f"\nNetwork Architecture:")
        
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                summary.append(f"  Layer {i}: Linear({module.in_features} -> {module.out_features})")
            elif isinstance(module, nn.ReLU):
                summary.append(f"  Layer {i}: ReLU()")
            elif isinstance(module, nn.Dropout):
                summary.append(f"  Layer {i}: Dropout(p={module.p})")
        
        summary.append(f"\nOutput:")
        summary.append(f"  - Grid states: {self.n_outputs} (33 voltages + 33 angles)")
        summary.append(f"\nTotal Parameters: {self.count_parameters():,}")
        summary.append("="*70)
        
        return "\n".join(summary)


class WeightedMSELoss(nn.Module):
    """
    Custom loss function with higher weight on voltage predictions
    since voltage magnitudes are more critical than angles for grid stability
    """
    
    def __init__(self, voltage_weight=2.0, angle_weight=1.0, n_buses=33):
        super(WeightedMSELoss, self).__init__()
        self.voltage_weight = voltage_weight
        self.angle_weight = angle_weight
        self.n_buses = n_buses
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss
        
        Args:
            predictions: Model predictions (batch_size, 66)
            targets: Ground truth (batch_size, 66)
        
        Returns:
            Weighted MSE loss (scalar)
        """
        # Split into voltage and angle components
        pred_voltages = predictions[:, :self.n_buses]
        pred_angles = predictions[:, self.n_buses:]
        
        target_voltages = targets[:, :self.n_buses]
        target_angles = targets[:, self.n_buses:]
        
        # Compute MSE for each component
        voltage_mse = F.mse_loss(pred_voltages, target_voltages)
        angle_mse = F.mse_loss(pred_angles, target_angles)
        
        # Weighted combination
        total_loss = self.voltage_weight * voltage_mse + self.angle_weight * angle_mse
        
        return total_loss


class BaselineANN(nn.Module):
    """
    Baseline ANN without fuzzy preprocessing (for comparison)
    """
    
    def __init__(
        self, 
        n_sensor_features=20,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    ):
        super(BaselineANN, self).__init__()
        
        self.n_sensor_features = n_sensor_features
        self.n_outputs = n_outputs
        
        # Input: sensor features + binary mask
        self.input_dim = n_sensor_features + n_sensor_features
        
        # Build network
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_outputs))
        
        self.network = nn.Sequential(*layers)
        
        self.sensor_mean = None
        self.sensor_std = None
        self.output_mean = None
        self.output_std = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def fit_normalization(self, X_sensor: np.ndarray, y: np.ndarray):
        self.sensor_mean = np.nanmean(X_sensor, axis=0)
        self.sensor_std = np.nanstd(X_sensor, axis=0) + 1e-8
        self.output_mean = np.mean(y, axis=0)
        self.output_std = np.std(y, axis=0) + 1e-8
    
    def preprocess_input(self, X_sensor: np.ndarray) -> torch.Tensor:
        mask = (~np.isnan(X_sensor)).astype(np.float32)
        X_sensor_imputed = X_sensor.copy()
        for i in range(X_sensor.shape[1]):
            X_sensor_imputed[np.isnan(X_sensor[:, i]), i] = self.sensor_mean[i]
        X_sensor_normalized = (X_sensor_imputed - self.sensor_mean) / self.sensor_std
        X_combined = np.concatenate([X_sensor_normalized, mask], axis=1)
        return torch.FloatTensor(X_combined)
    
    def normalize_output(self, y: np.ndarray) -> torch.Tensor:
        y_normalized = (y - self.output_mean) / self.output_std
        return torch.FloatTensor(y_normalized)
    
    def denormalize_output(self, y_normalized: torch.Tensor) -> np.ndarray:
        if isinstance(y_normalized, torch.Tensor):
            y_normalized = y_normalized.detach().cpu().numpy()
        return y_normalized * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict(self, X_sensor: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X = self.preprocess_input(X_sensor)
            y_pred_normalized = self.forward(X)
            y_pred = self.denormalize_output(y_pred_normalized)
        return y_pred
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
