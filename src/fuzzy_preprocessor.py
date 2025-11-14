"""
Fuzzy Logic Preprocessor for Load Flow Estimation
Handles uncertainty and missing data through fuzzy inference
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from typing import Tuple, Dict


class FuzzyPreprocessor:
    """
    Fuzzy logic system for preprocessing sparse sensor data.
    Generates confidence scores and enhanced features for neural network.
    """
    
    def __init__(self):
        """Initialize fuzzy membership functions and rule base"""
        
        # Define fuzzy variables
        # Voltage magnitude (pu units: 0.90 to 1.00)
        self.voltage = ctrl.Antecedent(np.arange(0.85, 1.05, 0.01), 'voltage')
        self.voltage['low'] = fuzz.trapmf(self.voltage.universe, [0.85, 0.85, 0.93, 0.96])
        self.voltage['normal'] = fuzz.trimf(self.voltage.universe, [0.94, 0.97, 1.00])
        self.voltage['high'] = fuzz.trapmf(self.voltage.universe, [0.98, 1.00, 1.05, 1.05])
        
        # Current magnitude (normalized 0-1, will scale based on data)
        self.current = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'current')
        self.current['low'] = fuzz.trapmf(self.current.universe, [0, 0, 0.2, 0.4])
        self.current['medium'] = fuzz.trimf(self.current.universe, [0.3, 0.5, 0.7])
        self.current['high'] = fuzz.trapmf(self.current.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Power (normalized 0-1)
        self.power = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'power')
        self.power['low'] = fuzz.trapmf(self.power.universe, [0, 0, 0.2, 0.4])
        self.power['medium'] = fuzz.trimf(self.power.universe, [0.3, 0.5, 0.7])
        self.power['high'] = fuzz.trapmf(self.power.universe, [0.6, 0.8, 1.0, 1.0])
        
        # Data availability (percentage of non-missing values)
        self.availability = ctrl.Antecedent(np.arange(0, 101, 1), 'availability')
        self.availability['sparse'] = fuzz.trapmf(self.availability.universe, [0, 0, 20, 40])
        self.availability['medium'] = fuzz.trimf(self.availability.universe, [30, 50, 70])
        self.availability['dense'] = fuzz.trapmf(self.availability.universe, [60, 80, 100, 100])
        
        # Output: Confidence score (0-100)
        self.confidence = ctrl.Consequent(np.arange(0, 101, 1), 'confidence')
        self.confidence['low'] = fuzz.trimf(self.confidence.universe, [0, 0, 40])
        self.confidence['medium'] = fuzz.trimf(self.confidence.universe, [30, 50, 70])
        self.confidence['high'] = fuzz.trimf(self.confidence.universe, [60, 100, 100])
        
        # Output: Quality indicator
        self.quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')
        self.quality['poor'] = fuzz.trimf(self.quality.universe, [0, 0, 40])
        self.quality['fair'] = fuzz.trimf(self.quality.universe, [30, 50, 70])
        self.quality['good'] = fuzz.trimf(self.quality.universe, [60, 100, 100])
        
        # Define fuzzy rules
        self.rules = self._create_rules()
        
        # Create control systems
        self.confidence_system = ctrl.ControlSystem(self.rules['confidence'])
        self.quality_system = ctrl.ControlSystem(self.rules['quality'])
        
        # Statistics for normalization (will be fitted on training data)
        self.current_stats = {'min': 0, 'max': 150}  # Default range in Amps
        self.power_stats = {'min': 0, 'max': 0.5}    # Default range in MW
        
    def _create_rules(self) -> Dict:
        """Create fuzzy rule base for inference"""
        
        confidence_rules = [
            # High confidence rules
            ctrl.Rule(self.voltage['normal'] & self.availability['dense'], 
                     self.confidence['high']),
            ctrl.Rule(self.voltage['high'] & self.availability['dense'], 
                     self.confidence['high']),
            ctrl.Rule(self.voltage['normal'] & self.availability['medium'], 
                     self.confidence['medium']),
            
            # Medium confidence rules
            ctrl.Rule(self.voltage['low'] & self.availability['dense'], 
                     self.confidence['medium']),
            ctrl.Rule(self.voltage['normal'] & self.availability['sparse'], 
                     self.confidence['medium']),
            
            # Low confidence rules
            ctrl.Rule(self.voltage['low'] & self.availability['sparse'], 
                     self.confidence['low']),
            ctrl.Rule(self.voltage['low'] & self.availability['medium'], 
                     self.confidence['low']),
        ]
        
        quality_rules = [
            # Good quality
            ctrl.Rule(self.current['medium'] & self.power['medium'] & 
                     self.availability['dense'], self.quality['good']),
            ctrl.Rule(self.current['low'] & self.power['low'] & 
                     self.availability['dense'], self.quality['good']),
            
            # Fair quality
            ctrl.Rule(self.current['high'] & self.availability['medium'], 
                     self.quality['fair']),
            ctrl.Rule(self.power['high'] & self.availability['medium'], 
                     self.quality['fair']),
            
            # Poor quality
            ctrl.Rule(self.current['high'] & self.availability['sparse'], 
                     self.quality['poor']),
            ctrl.Rule(self.availability['sparse'], self.quality['poor']),
        ]
        
        return {'confidence': confidence_rules, 'quality': quality_rules}
    
    def fit(self, X: pd.DataFrame):
        """Fit preprocessor statistics on training data"""
        # Extract current and power measurements (columns with I_, P_, Q_)
        current_cols = [c for c in X.columns if 'meas_' in c]
        
        # Estimate ranges from data (excluding NaN)
        all_values = X[current_cols].values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        
        # Separate voltage, current, power based on typical ranges
        voltage_values = all_values[(all_values >= 0.8) & (all_values <= 1.1)]
        current_values = all_values[(all_values > 1.1) & (all_values < 200)]
        power_values = all_values[(all_values >= 0) & (all_values < 2)]
        
        if len(current_values) > 0:
            self.current_stats = {
                'min': np.percentile(current_values, 1),
                'max': np.percentile(current_values, 99)
            }
        
        if len(power_values) > 0:
            self.power_stats = {
                'min': np.percentile(power_values, 1),
                'max': np.percentile(power_values, 99)
            }
        
        print(f"Fitted fuzzy preprocessor:")
        print(f"  Current range: {self.current_stats['min']:.2f} - {self.current_stats['max']:.2f}")
        print(f"  Power range: {self.power_stats['min']:.4f} - {self.power_stats['max']:.4f}")
        
        return self
    
    def _normalize_value(self, value, stats):
        """Normalize value to 0-1 range"""
        if np.isnan(value):
            return np.nan
        return np.clip((value - stats['min']) / (stats['max'] - stats['min'] + 1e-8), 0, 1)
    
    def _compute_membership_degrees(self, value, fuzzy_var) -> Dict[str, float]:
        """Compute membership degrees for all fuzzy sets"""
        if np.isnan(value) or value < fuzzy_var.universe.min() or value > fuzzy_var.universe.max():
            return {term: 0.0 for term in fuzzy_var.terms.keys()}
        
        degrees = {}
        for term_name in fuzzy_var.terms.keys():
            degrees[term_name] = fuzz.interp_membership(
                fuzzy_var.universe, 
                fuzzy_var[term_name].mf, 
                value
            )
        return degrees
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform sparse sensor data to fuzzy-enhanced features
        
        Returns:
            Enhanced feature array with shape (n_samples, n_features + fuzzy_features)
        """
        n_samples = len(X)
        fuzzy_features = []
        
        for idx in range(n_samples):
            row = X.iloc[idx].values
            
            # Calculate data availability
            availability_pct = (1 - np.isnan(row).sum() / len(row)) * 100
            
            # Extract representative values (first non-NaN of each type)
            voltage_vals = []
            current_vals = []
            power_vals = []
            
            for val in row:
                if not np.isnan(val):
                    if 0.8 <= val <= 1.1:
                        voltage_vals.append(val)
                    elif val > 1.1 and val < 200:
                        current_vals.append(val)
                    elif 0 <= val < 2:
                        power_vals.append(val)
            
            # Use mean values or defaults
            voltage_val = np.mean(voltage_vals) if voltage_vals else 0.95
            current_val = self._normalize_value(
                np.mean(current_vals) if current_vals else 50, 
                self.current_stats
            )
            if np.isnan(current_val):
                current_val = 0.5
                
            power_val = self._normalize_value(
                np.mean(power_vals) if power_vals else 0.2, 
                self.power_stats
            )
            if np.isnan(power_val):
                power_val = 0.5
            
            # Compute membership degrees
            voltage_memberships = self._compute_membership_degrees(voltage_val, self.voltage)
            current_memberships = self._compute_membership_degrees(current_val, self.current)
            power_memberships = self._compute_membership_degrees(power_val, self.power)
            availability_memberships = self._compute_membership_degrees(availability_pct, self.availability)
            
            # Compute fuzzy inference outputs (confidence and quality)
            try:
                conf_sim = ctrl.ControlSystemSimulation(self.confidence_system)
                conf_sim.input['voltage'] = voltage_val
                conf_sim.input['availability'] = availability_pct
                conf_sim.compute()
                confidence_score = conf_sim.output['confidence']
            except:
                confidence_score = 50.0  # Default medium confidence
            
            try:
                qual_sim = ctrl.ControlSystemSimulation(self.quality_system)
                qual_sim.input['current'] = current_val
                qual_sim.input['power'] = power_val
                qual_sim.input['availability'] = availability_pct
                qual_sim.compute()
                quality_score = qual_sim.output['quality']
            except:
                quality_score = 50.0  # Default fair quality
            
            # Combine all fuzzy features
            sample_fuzzy_features = [
                availability_pct / 100.0,  # Normalized availability
                confidence_score / 100.0,   # Confidence score
                quality_score / 100.0,      # Quality score
                voltage_memberships['low'],
                voltage_memberships['normal'],
                voltage_memberships['high'],
                current_memberships['low'],
                current_memberships['medium'],
                current_memberships['high'],
                power_memberships['low'],
                power_memberships['medium'],
                power_memberships['high'],
            ]
            
            fuzzy_features.append(sample_fuzzy_features)
        
        return np.array(fuzzy_features)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def visualize_membership_functions(self, save_path='fuzzy_membership_functions.png'):
        """Visualize fuzzy membership functions"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fuzzy Membership Functions for Load Flow Estimation', fontsize=16)
        
        # Voltage
        axes[0, 0].plot(self.voltage.universe, self.voltage['low'].mf, 'b', linewidth=2, label='Low')
        axes[0, 0].plot(self.voltage.universe, self.voltage['normal'].mf, 'g', linewidth=2, label='Normal')
        axes[0, 0].plot(self.voltage.universe, self.voltage['high'].mf, 'r', linewidth=2, label='High')
        axes[0, 0].set_title('Voltage Magnitude (pu)')
        axes[0, 0].set_xlabel('Voltage (pu)')
        axes[0, 0].set_ylabel('Membership Degree')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Current
        axes[0, 1].plot(self.current.universe, self.current['low'].mf, 'b', linewidth=2, label='Low')
        axes[0, 1].plot(self.current.universe, self.current['medium'].mf, 'g', linewidth=2, label='Medium')
        axes[0, 1].plot(self.current.universe, self.current['high'].mf, 'r', linewidth=2, label='High')
        axes[0, 1].set_title('Current (Normalized)')
        axes[0, 1].set_xlabel('Current (normalized)')
        axes[0, 1].set_ylabel('Membership Degree')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Availability
        axes[1, 0].plot(self.availability.universe, self.availability['sparse'].mf, 'b', linewidth=2, label='Sparse')
        axes[1, 0].plot(self.availability.universe, self.availability['medium'].mf, 'g', linewidth=2, label='Medium')
        axes[1, 0].plot(self.availability.universe, self.availability['dense'].mf, 'r', linewidth=2, label='Dense')
        axes[1, 0].set_title('Data Availability (%)')
        axes[1, 0].set_xlabel('Availability (%)')
        axes[1, 0].set_ylabel('Membership Degree')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence Output
        axes[1, 1].plot(self.confidence.universe, self.confidence['low'].mf, 'b', linewidth=2, label='Low')
        axes[1, 1].plot(self.confidence.universe, self.confidence['medium'].mf, 'g', linewidth=2, label='Medium')
        axes[1, 1].plot(self.confidence.universe, self.confidence['high'].mf, 'r', linewidth=2, label='High')
        axes[1, 1].set_title('Confidence Score (Output)')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Membership Degree')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Membership functions saved to {save_path}")
        plt.close()
