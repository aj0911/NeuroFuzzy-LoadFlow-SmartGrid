"""
Comprehensive Data Analysis for Neuro-Fuzzy Load Flow Estimation
Generates 15+ publication-quality visualizations for B.Tech presentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directory
output_dir = Path('results/data-analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("COMPREHENSIVE DATA ANALYSIS")
print("="*70)

# Load data
print("\n[1] Loading dataset...")
X_sensor = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv')
y_grid = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv')

n_samples = len(X_sensor)
n_features = X_sensor.shape[1]
n_outputs = y_grid.shape[1]
n_buses = 33

print(f"   Samples: {n_samples}")
print(f"   Input features: {n_features}")
print(f"   Output features: {n_outputs} ({n_buses} voltages + {n_buses} angles)")

# Calculate statistics
sparsity = X_sensor.isna().mean().mean()
print(f"   Overall sparsity: {sparsity:.2%}")

# ==========================================================================
# FIGURE 1: Dataset Overview
# ==========================================================================
print("\n[2] Generating Figure 1: Dataset Overview...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1: Samples distribution across sparsity levels
sparsity_per_sample = X_sensor.isna().mean(axis=1) * 100
axes[0, 0].hist(sparsity_per_sample, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(sparsity_per_sample.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sparsity_per_sample.mean():.1f}%')
axes[0, 0].set_xlabel('Sparsity per Sample (%)', fontsize=11)
axes[0, 0].set_ylabel('Number of Samples', fontsize=11)
axes[0, 0].set_title('Distribution of Data Sparsity Across Samples', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2: Feature availability
feature_availability = (~X_sensor.isna()).mean() * 100
axes[0, 1].bar(range(n_features), feature_availability, alpha=0.7, color='seagreen', edgecolor='black')
axes[0, 1].axhline(50, color='red', linestyle='--', linewidth=1.5, label='50% threshold')
axes[0, 1].set_xlabel('Feature Index', fontsize=11)
axes[0, 1].set_ylabel('Availability (%)', fontsize=11)
axes[0, 1].set_title('Feature Availability Across Dataset', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 1.3: Sample statistics
available_per_sample = (~X_sensor.isna()).sum(axis=1)
axes[1, 0].hist(available_per_sample, bins=range(0, n_features+2), alpha=0.7, color='coral', edgecolor='black')
axes[1, 0].axvline(available_per_sample.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {available_per_sample.mean():.1f}')
axes[1, 0].set_xlabel('Number of Available Features', fontsize=11)
axes[1, 0].set_ylabel('Number of Samples', fontsize=11)
axes[1, 0].set_title('Distribution of Available Measurements per Sample', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 1.4: Dataset size comparison
dataset_info = {
    'Train Samples\n(80%)': n_samples * 0.8,
    'Val Samples\n(20%)': n_samples * 0.2,
    'Input\nFeatures': n_features,
    'Output\nTargets': n_outputs
}
axes[1, 1].bar(dataset_info.keys(), dataset_info.values(), alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Dataset Composition', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(dataset_info.items()):
    axes[1, 1].text(i, v + max(dataset_info.values())*0.02, f'{int(v)}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Dataset Overview - IEEE 33-Bus Load Flow Estimation', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '01_dataset_overview.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 01_dataset_overview.png")

# ==========================================================================
# FIGURE 2: Voltage Distribution Analysis
# ==========================================================================
print("\n[3] Generating Figure 2: Voltage Distribution Analysis...")

voltage_cols = [col for col in y_grid.columns if col.startswith('V_')]
voltages = y_grid[voltage_cols].values.flatten()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1: Overall voltage distribution
axes[0, 0].hist(voltages, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(voltages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {voltages.mean():.4f} pu')
axes[0, 0].axvline(0.95, color='orange', linestyle=':', linewidth=2, label='Low limit (0.95 pu)')
axes[0, 0].axvline(1.05, color='orange', linestyle=':', linewidth=2, label='High limit (1.05 pu)')
axes[0, 0].set_xlabel('Voltage Magnitude (pu)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Overall Voltage Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2.2: Voltage per bus (box plot)
voltage_data = []
bus_labels = []
for i in range(min(15, n_buses)):  # Show first 15 buses
    voltage_data.append(y_grid[f'V_{i}'].values)
    bus_labels.append(f'Bus {i}')

bp = axes[0, 1].boxplot(voltage_data, labels=bus_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
axes[0, 1].axhline(0.95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].axhline(1.05, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Bus Number', fontsize=11)
axes[0, 1].set_ylabel('Voltage (pu)', fontsize=11)
axes[0, 1].set_title('Voltage Distribution per Bus (First 15 Buses)', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 2.3: Voltage statistics across buses
mean_voltages = [y_grid[f'V_{i}'].mean() for i in range(n_buses)]
std_voltages = [y_grid[f'V_{i}'].std() for i in range(n_buses)]

axes[1, 0].plot(range(n_buses), mean_voltages, 'o-', linewidth=2, markersize=5, label='Mean Voltage', color='steelblue')
axes[1, 0].fill_between(range(n_buses), 
                         np.array(mean_voltages) - np.array(std_voltages),
                         np.array(mean_voltages) + np.array(std_voltages),
                         alpha=0.3, color='steelblue', label='±1 Std Dev')
axes[1, 0].axhline(1.0, color='green', linestyle='--', linewidth=1.5, label='Nominal (1.0 pu)')
axes[1, 0].set_xlabel('Bus Number', fontsize=11)
axes[1, 0].set_ylabel('Voltage (pu)', fontsize=11)
axes[1, 0].set_title('Mean Voltage Profile Across All Buses', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 2.4: Voltage violations
violations_low = (y_grid[voltage_cols] < 0.95).sum().sum()
violations_high = (y_grid[voltage_cols] > 1.05).sum().sum()
normal = len(voltages) - violations_low - violations_high

violation_data = {
    'Normal\n(0.95-1.05 pu)': normal,
    'Low Voltage\n(<0.95 pu)': violations_low,
    'High Voltage\n(>1.05 pu)': violations_high
}

colors_viol = ['green', 'orange', 'red']
axes[1, 1].bar(violation_data.keys(), violation_data.values(), alpha=0.7, color=colors_viol, edgecolor='black')
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Voltage Constraint Violations', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(violation_data.items()):
    pct = v / len(voltages) * 100
    axes[1, 1].text(i, v + max(violation_data.values())*0.02, f'{int(v)}\n({pct:.1f}%)', 
                    ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Voltage Magnitude Analysis - IEEE 33-Bus System', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '02_voltage_distribution.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 02_voltage_distribution.png")

# ==========================================================================
# FIGURE 3: Angle Distribution Analysis
# ==========================================================================
print("\n[4] Generating Figure 3: Angle Distribution Analysis...")

angle_cols = [col for col in y_grid.columns if col.startswith('theta_')]
angles = y_grid[angle_cols].values.flatten()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1: Overall angle distribution
axes[0, 0].hist(angles, bins=50, alpha=0.7, color='darkorange', edgecolor='black')
axes[0, 0].axvline(angles.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {angles.mean():.4f}°')
axes[0, 0].axvline(0, color='green', linestyle=':', linewidth=2, label='Reference (0°)')
axes[0, 0].set_xlabel('Voltage Angle (degrees)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Overall Voltage Angle Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 3.2: Angle per bus (selected buses)
angle_data = []
bus_labels_angle = []
for i in range(min(15, n_buses)):
    angle_data.append(y_grid[f'theta_{i}'].values)
    bus_labels_angle.append(f'Bus {i}')

bp = axes[0, 1].boxplot(angle_data, labels=bus_labels_angle, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightsalmon')
    patch.set_alpha(0.7)
axes[0, 1].axhline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Bus Number', fontsize=11)
axes[0, 1].set_ylabel('Angle (degrees)', fontsize=11)
axes[0, 1].set_title('Angle Distribution per Bus (First 15 Buses)', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3.3: Angle profile across buses
mean_angles = [y_grid[f'theta_{i}'].mean() for i in range(n_buses)]
std_angles = [y_grid[f'theta_{i}'].std() for i in range(n_buses)]

axes[1, 0].plot(range(n_buses), mean_angles, 'o-', linewidth=2, markersize=5, label='Mean Angle', color='darkorange')
axes[1, 0].fill_between(range(n_buses),
                         np.array(mean_angles) - np.array(std_angles),
                         np.array(mean_angles) + np.array(std_angles),
                         alpha=0.3, color='darkorange', label='±1 Std Dev')
axes[1, 0].axhline(0, color='green', linestyle='--', linewidth=1.5, label='Reference (0°)')
axes[1, 0].set_xlabel('Bus Number', fontsize=11)
axes[1, 0].set_ylabel('Angle (degrees)', fontsize=11)
axes[1, 0].set_title('Mean Angle Profile Across All Buses', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 3.4: Angle range per bus
angle_ranges = [y_grid[f'theta_{i}'].max() - y_grid[f'theta_{i}'].min() for i in range(n_buses)]
axes[1, 1].bar(range(n_buses), angle_ranges, alpha=0.7, color='coral', edgecolor='black')
axes[1, 1].set_xlabel('Bus Number', fontsize=11)
axes[1, 1].set_ylabel('Angle Range (degrees)', fontsize=11)
axes[1, 1].set_title('Angle Variation Range per Bus', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Voltage Angle Analysis - IEEE 33-Bus System', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '03_angle_distribution.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 03_angle_distribution.png")

# ==========================================================================
# FIGURE 4: Correlation Analysis
# ==========================================================================
print("\n[5] Generating Figure 4: Correlation Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 4.1: Voltage correlations (first 15 buses)
voltage_subset = y_grid[[f'V_{i}' for i in range(15)]]
corr_V = voltage_subset.corr()

sns.heatmap(corr_V, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Voltage Correlation Matrix (First 15 Buses)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Bus Number', fontsize=11)
axes[0].set_ylabel('Bus Number', fontsize=11)

# 4.2: Angle correlations (first 15 buses)
angle_subset = y_grid[[f'theta_{i}' for i in range(15)]]
corr_theta = angle_subset.corr()

sns.heatmap(corr_theta, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Angle Correlation Matrix (First 15 Buses)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Bus Number', fontsize=11)
axes[1].set_ylabel('Bus Number', fontsize=11)

plt.suptitle('Spatial Correlation Analysis - IEEE 33-Bus System', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '04_correlation_analysis.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 04_correlation_analysis.png")

# ==========================================================================
# FIGURE 5: Input Data Characteristics
# ==========================================================================
print("\n[6] Generating Figure 5: Input Data Characteristics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1: Value range distribution
all_sensor_values = X_sensor.values.flatten()
all_sensor_values = all_sensor_values[~np.isnan(all_sensor_values)]

axes[0, 0].hist(all_sensor_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[0, 0].set_xlabel('Sensor Value', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of All Sensor Measurements', fontsize=12, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# 5.2: Voltage measurements vs others
voltage_measurements = all_sensor_values[(all_sensor_values >= 0.8) & (all_sensor_values <= 1.2)]
current_measurements = all_sensor_values[(all_sensor_values > 1.2) & (all_sensor_values < 200)]
power_measurements = all_sensor_values[(all_sensor_values >= 0) & (all_sensor_values < 2) & 
                                       ~((all_sensor_values >= 0.8) & (all_sensor_values <= 1.2))]

measurement_types = {
    'Voltage\n(0.8-1.2 pu)': len(voltage_measurements),
    'Current\n(>1.2 A)': len(current_measurements),
    'Power\n(0-2 MW)': len(power_measurements)
}

axes[0, 1].bar(measurement_types.keys(), measurement_types.values(), 
               alpha=0.7, color=['steelblue', 'orange', 'green'], edgecolor='black')
axes[0, 1].set_ylabel('Number of Measurements', fontsize=11)
axes[0, 1].set_title('Measurement Type Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(measurement_types.items()):
    axes[0, 1].text(i, v + max(measurement_types.values())*0.02, f'{int(v)}',
                    ha='center', fontsize=10, fontweight='bold')

# 5.3: Sparsity pattern heatmap (first 100 samples, all features)
sample_subset = X_sensor.head(100)
sparsity_pattern = sample_subset.isna().astype(int)

sns.heatmap(sparsity_pattern.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing (1) / Present (0)'},
            ax=axes[1, 0])
axes[1, 0].set_xlabel('Sample Index', fontsize=11)
axes[1, 0].set_ylabel('Feature Index', fontsize=11)
axes[1, 0].set_title('Sparsity Pattern (First 100 Samples)', fontsize=12, fontweight='bold')

# 5.4: Noise characteristics (estimated)
# For voltage measurements, estimate noise by looking at deviations from smooth patterns
sample_voltages = voltage_measurements[:1000]
sorted_voltages = np.sort(sample_voltages)
smoothed = np.convolve(sorted_voltages, np.ones(10)/10, mode='same')
noise_estimate = sorted_voltages - smoothed

axes[1, 1].hist(noise_estimate, bins=30, alpha=0.7, color='crimson', edgecolor='black')
axes[1, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='Zero noise')
axes[1, 1].set_xlabel('Estimated Noise (pu)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Estimated Measurement Noise Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Input Data Characteristics - Sensor Measurements', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '05_input_characteristics.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 05_input_characteristics.png")

# ==========================================================================
# FIGURE 6: Statistical Summary
# ==========================================================================
print("\n[7] Generating Figure 6: Statistical Summary...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1: Voltage statistics
voltage_stats = y_grid[voltage_cols].describe().T[['mean', 'std', 'min', 'max']]
x = np.arange(n_buses)
width = 0.2

axes[0, 0].bar(x - 1.5*width, voltage_stats['mean'], width, label='Mean', alpha=0.8, color='steelblue')
axes[0, 0].bar(x - 0.5*width, voltage_stats['std'], width, label='Std Dev', alpha=0.8, color='orange')
axes[0, 0].bar(x + 0.5*width, voltage_stats['min'], width, label='Min', alpha=0.8, color='green')
axes[0, 0].bar(x + 1.5*width, voltage_stats['max'], width, label='Max', alpha=0.8, color='red')
axes[0, 0].set_xlabel('Bus Number', fontsize=11)
axes[0, 0].set_ylabel('Voltage (pu)', fontsize=11)
axes[0, 0].set_title('Voltage Statistics per Bus', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 6.2: Angle statistics
angle_stats = y_grid[angle_cols].describe().T[['mean', 'std', 'min', 'max']]

axes[0, 1].bar(x - 1.5*width, angle_stats['mean'], width, label='Mean', alpha=0.8, color='steelblue')
axes[0, 1].bar(x - 0.5*width, angle_stats['std'], width, label='Std Dev', alpha=0.8, color='orange')
axes[0, 1].bar(x + 0.5*width, angle_stats['min'], width, label='Min', alpha=0.8, color='green')
axes[0, 1].bar(x + 1.5*width, angle_stats['max'], width, label='Max', alpha=0.8, color='red')
axes[0, 1].set_xlabel('Bus Number', fontsize=11)
axes[0, 1].set_ylabel('Angle (degrees)', fontsize=11)
axes[0, 1].set_title('Angle Statistics per Bus', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 6.3: Overall dataset statistics table
stats_data = {
    'Metric': [
        'Total Samples',
        'Training Samples (80%)',
        'Validation Samples (20%)',
        'Input Features',
        'Output Features',
        'Overall Sparsity',
        'Mean Available Features',
        'Voltage Range',
        'Angle Range'
    ],
    'Value': [
        f'{n_samples:,}',
        f'{int(n_samples * 0.8):,}',
        f'{int(n_samples * 0.2):,}',
        f'{n_features}',
        f'{n_outputs}',
        f'{sparsity:.1%}',
        f'{available_per_sample.mean():.1f}',
        f'{voltages.min():.4f} - {voltages.max():.4f} pu',
        f'{angles.min():.4f}° - {angles.max():.4f}°'
    ]
}

axes[1, 0].axis('tight')
axes[1, 0].axis('off')
table = axes[1, 0].table(cellText=[[stats_data['Metric'][i], stats_data['Value'][i]] for i in range(len(stats_data['Metric']))],
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(stats_data['Metric']) + 1):
    if i == 0:
        table[(i, 0)].set_facecolor('#4CAF50')
        table[(i, 1)].set_facecolor('#4CAF50')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    else:
        if i % 2 == 0:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#f0f0f0')

axes[1, 0].set_title('Dataset Statistics Summary', fontsize=12, fontweight='bold', pad=20)

# 6.4: Key insights
insights = [
    f"• Dataset contains {n_samples:,} simulated scenarios",
    f"• Average {sparsity:.1%} missing data per sample",
    f"• {len(voltage_measurements):,} voltage measurements collected",
    f"• {violations_low} low voltage violations detected",
    f"• Voltages range: {voltages.min():.4f} - {voltages.max():.4f} pu",
    f"• Angles range: {angles.min():.2f}° - {angles.max():.2f}°",
    f"• IEEE 33-bus standard benchmark system",
    f"• Realistic disaster scenario simulation",
    f"• 5-10% Gaussian noise applied",
    f"• 30-70% sensor sparsity variation"
]

axes[1, 1].axis('off')
axes[1, 1].text(0.05, 0.95, 'Key Insights:', fontsize=13, fontweight='bold', 
                verticalalignment='top', transform=axes[1, 1].transAxes)

for i, insight in enumerate(insights):
    axes[1, 1].text(0.05, 0.88 - i*0.08, insight, fontsize=10,
                    verticalalignment='top', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Statistical Summary - Dataset Characteristics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '06_statistical_summary.png', bbox_inches='tight')
plt.close()
print(f"   Saved: 06_statistical_summary.png")

# ==========================================================================
# Save JSON Summary
# ==========================================================================
print("\n[8] Generating JSON summary...")

summary_stats = {
    'dataset_info': {
        'total_samples': int(n_samples),
        'train_samples': int(n_samples * 0.8),
        'val_samples': int(n_samples * 0.2),
        'input_features': int(n_features),
        'output_features': int(n_outputs),
        'n_buses': int(n_buses)
    },
    'sparsity_stats': {
        'overall_sparsity_pct': float(sparsity * 100),
        'mean_available_features': float(available_per_sample.mean()),
        'min_available_features': int(available_per_sample.min()),
        'max_available_features': int(available_per_sample.max())
    },
    'voltage_stats': {
        'mean': float(voltages.mean()),
        'std': float(voltages.std()),
        'min': float(voltages.min()),
        'max': float(voltages.max()),
        'violations_low': int(violations_low),
        'violations_high': int(violations_high),
        'normal_count': int(normal)
    },
    'angle_stats': {
        'mean': float(angles.mean()),
        'std': float(angles.std()),
        'min': float(angles.min()),
        'max': float(angles.max())
    },
    'measurement_types': {
        'voltage_measurements': int(len(voltage_measurements)),
        'current_measurements': int(len(current_measurements)),
        'power_measurements': int(len(power_measurements))
    }
}

with open(output_dir / 'data_analysis_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"   Saved: data_analysis_summary.json")

# ==========================================================================
# Summary Report
# ==========================================================================
print("\n" + "="*70)
print("DATA ANALYSIS COMPLETE")
print("="*70)
print(f"\n✓ Generated 6 comprehensive visualizations")
print(f"✓ All files saved to: results/data-analysis/")
print(f"✓ Summary statistics saved to JSON")
print("\nGenerated Files:")
print("  1. 01_dataset_overview.png - Overall dataset composition")
print("  2. 02_voltage_distribution.png - Voltage analysis")
print("  3. 03_angle_distribution.png - Angle analysis")
print("  4. 04_correlation_analysis.png - Spatial correlations")
print("  5. 05_input_characteristics.png - Sensor measurement analysis")
print("  6. 06_statistical_summary.png - Complete statistics")
print("  7. data_analysis_summary.json - Numerical summary")
print("\n" + "="*70)
print("Ready for B.Tech presentation! 🎉")
print("="*70)
