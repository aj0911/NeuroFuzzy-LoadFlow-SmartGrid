"""
Training Pipeline for Neuro-Fuzzy Load Flow Estimation
Trains both neuro-fuzzy and baseline models with evaluation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from neurofuzzy_model import NeuroFuzzyLoadFlowModel, BaselineANN, WeightedMSELoss
from fuzzy_preprocessor import FuzzyPreprocessor


class LoadFlowDataset(Dataset):
    """PyTorch Dataset for load flow estimation"""
    
    def __init__(self, X_sensor: np.ndarray, X_fuzzy: np.ndarray, y: np.ndarray):
        self.X_sensor = X_sensor
        self.X_fuzzy = X_fuzzy
        self.y = y
    
    def __len__(self):
        return len(self.X_sensor)
    
    def __getitem__(self, idx):
        return self.X_sensor[idx], self.X_fuzzy[idx], self.y[idx]


class BaselineDataset(Dataset):
    """PyTorch Dataset for baseline model (no fuzzy features)"""
    
    def __init__(self, X_sensor: np.ndarray, y: np.ndarray):
        self.X_sensor = X_sensor
        self.y = y
    
    def __len__(self):
        return len(self.X_sensor)
    
    def __getitem__(self, idx):
        return self.X_sensor[idx], self.y[idx]


class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=1e-6, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Training manager for load flow estimation models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        model_name: str = "model",
        save_dir: str = "models/checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, is_neurofuzzy=True) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_data in self.train_loader:
            if is_neurofuzzy:
                X_sensor_batch, X_fuzzy_batch, y_batch = batch_data
                X_sensor_batch = X_sensor_batch.to(self.device)
                X_fuzzy_batch = X_fuzzy_batch.to(self.device)
                
                # Preprocess inputs
                X_batch = self.model.preprocess_input(
                    X_sensor_batch.cpu().numpy(),
                    X_fuzzy_batch.cpu().numpy()
                ).to(self.device)
            else:
                X_sensor_batch, y_batch = batch_data
                X_sensor_batch = X_sensor_batch.to(self.device)
                
                # Preprocess inputs
                X_batch = self.model.preprocess_input(
                    X_sensor_batch.cpu().numpy()
                ).to(self.device)
            
            y_batch = y_batch.to(self.device)
            
            # Normalize outputs
            y_batch_normalized = self.model.normalize_output(y_batch.cpu().numpy()).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            
            # Compute loss
            loss = self.criterion(y_pred, y_batch_normalized)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, is_neurofuzzy=True) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                if is_neurofuzzy:
                    X_sensor_batch, X_fuzzy_batch, y_batch = batch_data
                    X_sensor_batch = X_sensor_batch.to(self.device)
                    X_fuzzy_batch = X_fuzzy_batch.to(self.device)
                    
                    X_batch = self.model.preprocess_input(
                        X_sensor_batch.cpu().numpy(),
                        X_fuzzy_batch.cpu().numpy()
                    ).to(self.device)
                else:
                    X_sensor_batch, y_batch = batch_data
                    X_sensor_batch = X_sensor_batch.to(self.device)
                    
                    X_batch = self.model.preprocess_input(
                        X_sensor_batch.cpu().numpy()
                    ).to(self.device)
                
                y_batch = y_batch.to(self.device)
                y_batch_normalized = self.model.normalize_output(y_batch.cpu().numpy()).to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch_normalized)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, n_epochs: int, early_stopping: EarlyStopping = None, is_neurofuzzy=True):
        """Full training loop"""
        print(f"\nTraining {self.model_name}...")
        print("="*70)
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(is_neurofuzzy)
            
            # Validate
            val_loss = self.validate(is_neurofuzzy)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Track history
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint(f'{self.model_name}_best.pth', epoch)
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Save final model
        self.save_checkpoint(f'{self.model_name}_final.pth', n_epochs)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        print("="*70)
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'sensor_mean': self.model.sensor_mean,
            'sensor_std': self.model.sensor_std,
            'output_mean': self.model.output_mean,
            'output_std': self.model.output_std,
        }
        
        torch.save(checkpoint, self.save_dir / filename)


def plot_training_history(
    neurofuzzy_history: Dict,
    baseline_history: Dict,
    save_path: str = 'training_history.png'
):
    """Plot training curves for both models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    axes[0, 0].plot(neurofuzzy_history['train_loss'], label='Neuro-Fuzzy', linewidth=2)
    axes[0, 0].plot(baseline_history['train_loss'], label='Baseline ANN', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[0, 1].plot(neurofuzzy_history['val_loss'], label='Neuro-Fuzzy', linewidth=2)
    axes[0, 1].plot(baseline_history['val_loss'], label='Baseline ANN', linewidth=2)
    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(neurofuzzy_history['learning_rate'], label='Neuro-Fuzzy', linewidth=2)
    axes[1, 0].plot(baseline_history['learning_rate'], label='Baseline ANN', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch time
    axes[1, 1].plot(neurofuzzy_history['epoch_time'], label='Neuro-Fuzzy', linewidth=2, alpha=0.7)
    axes[1, 1].plot(baseline_history['epoch_time'], label='Baseline ANN', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Comparison: Neuro-Fuzzy vs Baseline ANN', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    
    print("="*70)
    print("PHASE 3: TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    config = {
        'train_split': 0.8,
        'batch_size': 64,
        'n_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 15,
        'voltage_weight': 2.0,
        'angle_weight': 1.0,
        'random_seed': 42
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n[1] Loading dataset...")
    X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv').values
    y = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv').values
    
    print(f"   Dataset size: {len(X_sensor)} samples")
    print(f"   Sensor features: {X_sensor.shape[1]}")
    print(f"   Output targets: {y.shape[1]}")
    
    # Load fuzzy preprocessor and generate fuzzy features
    print("\n[2] Loading fuzzy preprocessor and generating features...")
    with open('models/fuzzy_preprocessor.pkl', 'rb') as f:
        fuzzy_processor = pickle.load(f)
    
    X_sensor_df = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv')
    X_fuzzy = fuzzy_processor.transform(X_sensor_df)
    print(f"   Fuzzy features shape: {X_fuzzy.shape}")
    
    # Train/validation split
    print("\n[3] Splitting into train/validation sets...")
    n_train = int(len(X_sensor) * config['train_split'])
    indices = np.random.permutation(len(X_sensor))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_sensor_train, X_sensor_val = X_sensor[train_idx], X_sensor[val_idx]
    X_fuzzy_train, X_fuzzy_val = X_fuzzy[train_idx], X_fuzzy[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"   Training samples: {len(X_sensor_train)}")
    print(f"   Validation samples: {len(X_sensor_val)}")
    
    # Create data loaders
    print("\n[4] Creating data loaders...")
    train_dataset = LoadFlowDataset(X_sensor_train, X_fuzzy_train, y_train)
    val_dataset = LoadFlowDataset(X_sensor_val, X_fuzzy_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    baseline_train_dataset = BaselineDataset(X_sensor_train, y_train)
    baseline_val_dataset = BaselineDataset(X_sensor_val, y_val)
    
    baseline_train_loader = DataLoader(baseline_train_dataset, batch_size=config['batch_size'], shuffle=True)
    baseline_val_loader = DataLoader(baseline_val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Initialize models
    print("\n[5] Initializing models...")
    
    # Neuro-Fuzzy Model
    neurofuzzy_model = NeuroFuzzyLoadFlowModel(
        n_sensor_features=20,
        n_fuzzy_features=12,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    neurofuzzy_model.fit_normalization(X_sensor_train, y_train)
    print(f"   Neuro-Fuzzy parameters: {neurofuzzy_model.count_parameters():,}")
    
    # Baseline Model
    baseline_model = BaselineANN(
        n_sensor_features=20,
        n_outputs=66,
        hidden_dims=[128, 256, 128],
        dropout_rate=0.2
    )
    baseline_model.fit_normalization(X_sensor_train, y_train)
    print(f"   Baseline parameters: {baseline_model.count_parameters():,}")
    
    # Loss function
    criterion = WeightedMSELoss(
        voltage_weight=config['voltage_weight'],
        angle_weight=config['angle_weight'],
        n_buses=33
    )
    
    # Optimizers
    neurofuzzy_optimizer = optim.Adam(
        neurofuzzy_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    baseline_optimizer = optim.Adam(
        baseline_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate schedulers (reduce on plateau)
    neurofuzzy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        neurofuzzy_optimizer, mode='min', factor=0.5, patience=5
    )
    baseline_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        baseline_optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    neurofuzzy_early_stop = EarlyStopping(patience=config['patience'])
    baseline_early_stop = EarlyStopping(patience=config['patience'])
    
    # Train Neuro-Fuzzy Model
    print("\n[6] Training Neuro-Fuzzy Model...")
    neurofuzzy_trainer = Trainer(
        model=neurofuzzy_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=neurofuzzy_optimizer,
        scheduler=neurofuzzy_scheduler,
        device=device,
        model_name='neurofuzzy',
        save_dir='checkpoints'
    )
    
    neurofuzzy_history = neurofuzzy_trainer.train(
        n_epochs=config['n_epochs'],
        early_stopping=neurofuzzy_early_stop,
        is_neurofuzzy=True
    )
    
    # Train Baseline Model
    print("\n[7] Training Baseline ANN...")
    baseline_trainer = Trainer(
        model=baseline_model,
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        criterion=criterion,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        device=device,
        model_name='baseline',
        save_dir='checkpoints'
    )
    
    baseline_history = baseline_trainer.train(
        n_epochs=config['n_epochs'],
        early_stopping=baseline_early_stop,
        is_neurofuzzy=False
    )
    
    # Plot training history
    print("\n[8] Generating training visualizations...")
    plot_training_history(neurofuzzy_history, baseline_history, 'results/training_history.png')
    
    # Save training histories
    with open('results/training_histories.json', 'w') as f:
        json.dump({
            'neurofuzzy': neurofuzzy_history,
            'baseline': baseline_history,
            'config': config
        }, f, indent=2)
    
    print("\n[9] Training Summary:")
    print("="*70)
    print(f"Neuro-Fuzzy Model:")
    print(f"  Best validation loss: {neurofuzzy_trainer.best_val_loss:.6f}")
    print(f"  Best epoch: {neurofuzzy_trainer.best_epoch}")
    print(f"  Total training time: {sum(neurofuzzy_history['epoch_time']):.2f}s")
    print(f"\nBaseline ANN:")
    print(f"  Best validation loss: {baseline_trainer.best_val_loss:.6f}")
    print(f"  Best epoch: {baseline_trainer.best_epoch}")
    print(f"  Total training time: {sum(baseline_history['epoch_time']):.2f}s")
    print(f"\nImprovement: {(1 - neurofuzzy_trainer.best_val_loss / baseline_trainer.best_val_loss) * 100:.2f}%")
    print("="*70)
    
    print("\n✓ Phase 3 complete! Models saved to 'models/checkpoints/' directory")
    print("✓ Training history saved to 'results/training_histories.json'")
    print("✓ Training plot saved to 'results/training_history.png'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ TRAINING FAILED with error: {e}")
        import traceback
        traceback.print_exc()
