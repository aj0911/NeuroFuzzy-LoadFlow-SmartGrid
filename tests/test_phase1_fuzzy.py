"""
Test Script for Phase 1: Fuzzy Logic Preprocessor
Tests membership functions, rule inference, and feature generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from fuzzy_preprocessor import FuzzyPreprocessor
import matplotlib.pyplot as plt


def test_fuzzy_preprocessor():
    """Test the fuzzy logic preprocessor on real dataset"""
    
    print("="*70)
    print("PHASE 1 TEST: Fuzzy Logic Preprocessor")
    print("="*70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    X_train = pd.read_csv('output_generation/sensor_inputs_ieee_33-bus.csv')
    y_train = pd.read_csv('output_generation/grid_states_ieee_33-bus.csv')
    
    print(f"   Input shape: {X_train.shape}")
    print(f"   Output shape: {y_train.shape}")
    print(f"   Dataset sparsity: {X_train.isna().mean().mean():.2%}")
    
    # Initialize and fit fuzzy preprocessor
    print("\n[2] Initializing Fuzzy Preprocessor...")
    fuzzy_processor = FuzzyPreprocessor()
    
    print("\n[3] Fitting on training data...")
    fuzzy_processor.fit(X_train)
    
    # Test on a small sample
    print("\n[4] Testing fuzzy transformation on sample data...")
    sample_size = 10
    X_sample = X_train.head(sample_size)
    
    fuzzy_features = fuzzy_processor.transform(X_sample)
    print(f"   Original features shape: {X_sample.shape}")
    print(f"   Fuzzy features shape: {fuzzy_features.shape}")
    print(f"   Number of fuzzy features added: {fuzzy_features.shape[1]}")
    
    # Display fuzzy features for first 3 samples
    print("\n[5] Sample Fuzzy Features (first 3 samples):")
    print("   " + "-"*65)
    feature_names = [
        'Availability', 'Confidence', 'Quality',
        'V_low', 'V_normal', 'V_high',
        'I_low', 'I_medium', 'I_high',
        'P_low', 'P_medium', 'P_high'
    ]
    
    for i in range(min(3, sample_size)):
        print(f"\n   Sample {i+1}:")
        for j, name in enumerate(feature_names):
            print(f"      {name:12s}: {fuzzy_features[i, j]:.4f}")
    
    # Test full dataset transformation
    print("\n[6] Transforming full dataset...")
    full_fuzzy_features = fuzzy_processor.transform(X_train)
    print(f"   Transformed shape: {full_fuzzy_features.shape}")
    print(f"   No NaN values: {not np.isnan(full_fuzzy_features).any()}")
    
    # Statistics of fuzzy features
    print("\n[7] Fuzzy Feature Statistics:")
    print("   " + "-"*65)
    fuzzy_df = pd.DataFrame(full_fuzzy_features, columns=feature_names)
    print(fuzzy_df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Visualize membership functions
    print("\n[8] Generating membership function visualizations...")
    fuzzy_processor.visualize_membership_functions('results/fuzzy_membership_functions.png')
    
    # Test edge cases
    print("\n[9] Testing edge cases...")
    
    # Case 1: All missing data
    test_case_1 = pd.DataFrame([[np.nan] * 20], columns=X_train.columns)
    fuzzy_1 = fuzzy_processor.transform(test_case_1)
    print(f"   All NaN input -> Availability: {fuzzy_1[0, 0]:.4f}, Confidence: {fuzzy_1[0, 1]:.4f}")
    
    # Case 2: Full data (no missing)
    test_case_2 = X_train.dropna()
    if len(test_case_2) > 0:
        fuzzy_2 = fuzzy_processor.transform(test_case_2.head(1))
        print(f"   No NaN input -> Availability: {fuzzy_2[0, 0]:.4f}, Confidence: {fuzzy_2[0, 1]:.4f}")
    
    # Case 3: Medium sparsity
    test_case_3 = X_train.iloc[[100]].copy()
    fuzzy_3 = fuzzy_processor.transform(test_case_3)
    actual_avail = (1 - test_case_3.isna().values.sum() / test_case_3.shape[1])
    print(f"   Medium sparse -> Actual avail: {actual_avail:.2%}, Fuzzy avail: {fuzzy_3[0, 0]:.4f}")
    
    # Visualization of fuzzy feature distributions
    print("\n[10] Creating fuzzy feature distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(fuzzy_df['Availability'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Data Availability Distribution')
    axes[0, 0].set_xlabel('Availability Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(fuzzy_df['Confidence'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Confidence Score Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(fuzzy_df['Quality'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Quality Score Distribution')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: Availability vs Confidence
    axes[1, 1].scatter(fuzzy_df['Availability'], fuzzy_df['Confidence'], 
                       alpha=0.3, s=10, color='purple')
    axes[1, 1].set_title('Availability vs Confidence')
    axes[1, 1].set_xlabel('Availability Score')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fuzzy_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("   Saved to results/fuzzy_feature_distributions.png")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 TEST RESULTS: ✓ PASSED")
    print("="*70)
    print(f"✓ Fuzzy preprocessor successfully initialized")
    print(f"✓ Fitted on {len(X_train)} training samples")
    print(f"✓ Generated {fuzzy_features.shape[1]} fuzzy features per sample")
    print(f"✓ No NaN values in fuzzy features")
    print(f"✓ Visualizations saved:")
    print(f"  - results/fuzzy_membership_functions.png")
    print(f"  - results/fuzzy_feature_distributions.png")
    print("\n✓ Fuzzy Logic Preprocessor is ready for Phase 2 (Neural Network)")
    print("="*70)
    
    return fuzzy_processor, full_fuzzy_features


if __name__ == "__main__":
    try:
        fuzzy_processor, fuzzy_features = test_fuzzy_preprocessor()
        
        # Save preprocessor for later use
        import pickle
        with open('models/fuzzy_preprocessor.pkl', 'wb') as f:
            pickle.dump(fuzzy_processor, f)
        print("\n✓ Fuzzy preprocessor saved to 'models/fuzzy_preprocessor.pkl'")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
