# ⚡ Neuro-Fuzzy Load Flow Estimation for Disaster-Resilient Smart Grids

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)]()

**B.Tech Final Year Project** | Delhi Technological University  
**Team:** Abhinav Jha, Akshin Saxena, Akshat Garg | **Date:** November 2025

---

## 🎯 Project Overview

This project implements a **hybrid neuro-fuzzy system** for real-time load flow estimation in smart grids using **sparse mobile sensor data**. The system combines fuzzy logic preprocessing with deep neural networks to predict complete grid states (voltage magnitudes and angles) from limited, noisy sensor measurements—critical for disaster scenarios where traditional SCADA systems may fail.

### Key Achievements
- ✅ **Real-time Performance:** 0.089ms inference time (1120× faster than target)
- ✅ **High Accuracy:** Voltage MAE of 0.000337 pu (0.03% error)
- ✅ **Robustness:** Handles 75% data sparsity with minimal degradation
- ✅ **18.38% Improvement** over baseline ANN without fuzzy logic

---

## 📁 Project Structure

```
DTU Project/
│
├── 📄 README.md                    # This file - Project overview & usage guide
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 src/                         # Core Implementation (5 files)
│   ├── fuzzy_preprocessor.py      # Fuzzy logic system (membership functions, rules)
│   ├── neurofuzzy_model.py        # Neural network architecture (hybrid + baseline)
│   ├── train.py                   # Training pipeline with early stopping
│   ├── evaluate.py                # Comprehensive evaluation metrics
│   └── inference.py               # Production-ready inference script
│
├── 📂 tests/                       # Test Scripts (3 files)
│   ├── test_phase1_fuzzy.py       # Fuzzy preprocessor validation
│   ├── test_phase2_neural_network.py  # Neural network architecture tests
│   └── test_complete_pipeline.py  # End-to-end system validation
│
├── 📂 models/                      # Trained Models & Exports
│   ├── checkpoints/               # PyTorch model checkpoints
│   │   ├── neurofuzzy_best.pth    # ✨ BEST MODEL (966 KB)
│   │   ├── neurofuzzy_final.pth
│   │   ├── baseline_best.pth
│   │   └── baseline_final.pth
│   ├── fuzzy_preprocessor.pkl     # Fitted fuzzy preprocessor
│   ├── neurofuzzy_model.onnx      # Exported ONNX model (for deployment)
│   └── neurofuzzy_model_stats.pkl # Normalization statistics
│
├── 📂 results/                     # Visualizations & Results
│   ├── training_history.png       # Training/validation loss curves
│   ├── prediction_comparison.png  # Scatter plots (predicted vs actual)
│   ├── error_analysis.png         # Per-bus error analysis (33 buses)
│   ├── sparsity_impact.png        # Accuracy vs data sparsity
│   ├── fuzzy_membership_functions.png  # Fuzzy logic visualization
│   ├── fuzzy_feature_distributions.png # Fuzzy feature histograms
│   ├── training_histories.json    # Training curves data
│   ├── evaluation_results.json    # Complete evaluation metrics
│   └── pipeline_test_results.json # End-to-end test results
│
├── 📂 data_generation/             # Dataset Generation Scripts
│   ├── main.py                    # Pandapower data generation
│   └── ieee_33_bus_system.py      # IEEE 33-bus system definition
│
├── 📂 output_generation/           # Generated Dataset
│   ├── sensor_inputs_ieee_33-bus.csv   # 5,000 sparse sensor samples
│   └── grid_states_ieee_33-bus.csv     # 5,000 full grid state labels
│
└── 📂 docs/                        # Documentation
    ├── PROJECT_SUMMARY.md          # Complete project documentation
    └── ieee_33_bus_system.pdf      # IEEE 33-bus system reference
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd "/Users/abhinavjha/Drive/DTU Project"

# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline Test

```bash
# Test the entire system (all phases)
python tests/test_complete_pipeline.py
```

**Expected Output:**
```
Tests Passed: 8/8 (100.0%)
🎉 ALL TESTS PASSED - Pipeline is production-ready!
```

### 3. Make Real-Time Predictions

#### Option A: Demo Mode (uses validation data)
```bash
python src/inference.py --demo
```

#### Option B: Predict from CSV file
```bash
python src/inference.py --input your_sensor_data.csv --output predictions.csv
```

#### Option C: Python API
```python
import sys
sys.path.append('src')
from inference import LoadFlowPredictor

# Initialize predictor
predictor = LoadFlowPredictor()

# Make prediction
predictions, confidence = predictor.predict_from_array(
    sensor_measurements,  # numpy array with NaN for missing
    return_confidence=True
)

print(f"Voltages: {predictions[:, :33]}")
print(f"Confidence: {confidence['confidence_score']}")
```

---

## 🧪 Testing & Validation

### Phase 1: Fuzzy Logic Preprocessor
```bash
python tests/test_phase1_fuzzy.py
```
**Validates:**
- ✅ Membership functions (voltage, current, power, availability)
- ✅ Fuzzy rule inference (13 rules for confidence/quality)
- ✅ Feature generation (12 fuzzy features per sample)

### Phase 2: Neural Network Architecture
```bash
python tests/test_phase2_neural_network.py
```
**Validates:**
- ✅ Architecture (52 inputs → 128 → 256 → 128 → 66 outputs)
- ✅ Forward pass and gradient flow
- ✅ Inference time (<100ms requirement)

### Phase 3: Training Pipeline
```bash
python src/train.py
```
**Trains:**
- Neuro-fuzzy model (81,218 parameters)
- Baseline ANN (79,682 parameters)
- Saves best checkpoints to `models/checkpoints/`

### Phase 4: Evaluation & Metrics
```bash
python src/evaluate.py
```
**Generates:**
- Detailed metrics (MAE, RMSE, R², MAPE)
- Per-bus error analysis
- Sparsity impact analysis
- 3 visualization plots

### Phase 5: Complete System Test
```bash
python tests/test_complete_pipeline.py
```
**End-to-end validation:**
- ✅ File integrity check
- ✅ Model loading
- ✅ Prediction accuracy
- ✅ Inference time
- ✅ Confidence scores
- ✅ Sparsity robustness

---

## 📊 Performance Metrics

### Overall Performance (1,000 validation samples)

| Metric | Neuro-Fuzzy | Baseline ANN | Improvement |
|--------|-------------|--------------|-------------|
| **Voltage MAE (pu)** | **0.000337** | 0.000373 | **9.88%** ✅ |
| **Voltage RMSE (pu)** | 0.003092 | 0.003662 | **15.58%** ✅ |
| **Voltage R²** | **0.475** | 0.263 | **80.52%** ✅ |
| **Angle MAE (°)** | **0.002281** | 0.002402 | **5.03%** ✅ |
| **Angle RMSE (°)** | 0.025456 | 0.029795 | **14.56%** ✅ |
| **Inference Time (ms)** | **0.089** | 0.087 | Similar ✅ |

### Key Highlights
- 🎯 **0.03% voltage error** - Highly accurate predictions
- ⚡ **0.089ms inference** - 1120× faster than 100ms target
- 🛡️ **Handles 75% sparsity** - Robust to massive sensor loss
- 🚀 **1,049,521 samples/sec** - Production-ready throughput

---

## 🎓 System Architecture

### 1. Fuzzy Logic Preprocessor
- **Membership Functions:** Voltage (Low/Normal/High), Current (Low/Medium/High), Power (Low/Medium/High), Availability (Sparse/Medium/Dense)
- **Fuzzy Rules:** 13 IF-THEN rules for confidence and quality inference
- **Output:** 12 fuzzy features (availability, confidence, quality, membership degrees)

### 2. Neural Network
- **Architecture:** 52 inputs → 128 → 256 → 128 → 66 outputs
- **Input:** 20 sensor features + 20 binary masks + 12 fuzzy features
- **Output:** 33 voltages (pu) + 33 angles (degrees)
- **Parameters:** 81,218 trainable weights
- **Loss:** Weighted MSE (2× voltage, 1× angle)

### 3. Training Details
- **Dataset:** IEEE 33-bus system, 5,000 scenarios
- **Split:** 80/20 train/validation (4,000 / 1,000)
- **Sparsity:** 53.67% average missing data
- **Noise:** 5-10% Gaussian noise
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping:** Patience=15 epochs

---

## 📈 Usage Examples

### Example 1: Train New Model
```bash
# Train both neuro-fuzzy and baseline models
python src/train.py

# Output:
# - models/checkpoints/neurofuzzy_best.pth
# - models/checkpoints/baseline_best.pth
# - results/training_history.png
# - results/training_histories.json
```

### Example 2: Evaluate Trained Model
```bash
# Comprehensive evaluation on validation set
python src/evaluate.py

# Output:
# - results/prediction_comparison.png
# - results/error_analysis.png
# - results/sparsity_impact.png
# - results/evaluation_results.json
```

### Example 3: Export to ONNX
```bash
# Export model for deployment
python src/inference.py --export-onnx

# Output:
# - models/neurofuzzy_model.onnx
# - models/neurofuzzy_model_stats.pkl
```

### Example 4: Single Sample Prediction
```python
import sys
sys.path.append('src')
from inference import LoadFlowPredictor
import numpy as np

# Initialize predictor
predictor = LoadFlowPredictor()

# Create sensor measurements (NaN for missing values)
sensor_dict = {
    'meas_0': 0.97,    # Voltage at sensor 0
    'meas_1': 0.95,    # Voltage at sensor 1
    'meas_5': 15.3,    # Current at sensor 5
    'meas_8': 0.25,    # Power at sensor 8
    # ... remaining sensors are missing (NaN)
}

# Make prediction
result = predictor.predict_single(sensor_dict, verbose=True)

# Access results
voltages = result['voltages']  # Dict: 'bus_0': 0.98, 'bus_1': 0.97, ...
angles = result['angles']       # Dict: 'bus_0': 0.0, 'bus_1': -0.15, ...
metadata = result['metadata']   # Confidence scores, inference time
```

---

## 📚 Documentation

### Detailed Documentation
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Complete technical documentation
  - Architecture details
  - Mathematical foundations
  - Training procedures
  - Evaluation metrics
  - Research contributions

### IEEE 33-Bus System
- **[ieee_33_bus_system.pdf](docs/ieee_33_bus_system.pdf)** - Reference document
  - System topology
  - Bus/line parameters
  - Load specifications

---

## 🔧 Dependencies

### Core Libraries
```
torch==2.9.1                # Deep learning framework
numpy==2.3.1                # Numerical computing
pandas==2.3.1               # Data manipulation
scikit-fuzzy==0.5.0         # Fuzzy logic
scikit-learn==1.7.2         # Machine learning utilities
matplotlib==3.10.3          # Visualization
```

### Additional Requirements
```
pandapower==3.1.2           # Power system simulation
scipy==1.16.0               # Scientific computing
networkx==3.5               # Graph algorithms
onnx==1.19.1                # Model export
onnxscript==0.5.6           # ONNX utilities
```

### Installation
```bash
pip install -r requirements.txt
```

---

## 🎯 Project Milestones

- [x] **Phase 1:** Literature Review
- [x] **Phase 2:** Data Generation (5,000 scenarios)
- [x] **Phase 3:** Fuzzy Logic Implementation
- [x] **Phase 4:** Neural Network Development
- [x] **Phase 5:** Training & Optimization
- [x] **Phase 6:** Evaluation & Validation
- [x] **Phase 7:** Documentation & Deployment

---

## 🏆 Key Contributions

### Novel Aspects
1. **First application** of neuro-fuzzy architecture for sparse load flow estimation
2. **Disaster resilience** focus - optimized for 50%+ sensor loss
3. **Real-time capability** - sub-millisecond inference on CPU
4. **IEEE 33-bus validation** - complete benchmark on standard system

### Comparison with Traditional Methods

| Method | Voltage MAE | Handles Sparsity | Real-Time | Robustness |
|--------|-------------|------------------|-----------|------------|
| Newton-Raphson | N/A* | ❌ No | ⚠️ Slow | ❌ Low |
| Gauss-Seidel | N/A* | ❌ No | ⚠️ Slow | ❌ Low |
| Simple ANN | 0.000373 pu | ⚠️ Partial | ✅ Fast | ⚠️ Medium |
| **Neuro-Fuzzy** | **0.000337 pu** | ✅ **Yes** | ✅ **Fast** | ✅ **High** |

*Traditional methods require full observability (cannot handle missing data)

---

## 👥 Team

**B.Tech Final Year Project - Electrical Engineering**  
**Delhi Technological University**

- **Abhinav Jha (2K22/EE/10)** - Lead Developer, Power System Modeling
- **Akshin Saxena (2K22/EE/36)** - Data Preprocessing, Digital Logic
- **Akshat Garg (2K22/EE/35)** - Signal Processing, Dataset Validation

**Project Duration:** July 2025 - November 2025

---

## 📞 Contact & Support

For questions, issues, or collaboration:
- **GitHub Issues:** (if applicable)
- **Email:** Contact through Delhi Technological University

---

## 📄 License

This project is developed for academic purposes as part of the B.Tech curriculum at Delhi Technological University. All rights reserved.

---

## 🙏 Acknowledgments

- IEEE for providing standard test systems (IEEE 33-bus)
- Pandapower development team for power system simulation tools
- PyTorch and scikit-fuzzy communities for frameworks
- Delhi Technological University - Department of Electrical Engineering

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@project{jha2025neurofuzzy,
  title={A Neuro-Fuzzy System for Real-Time Load Flow Estimation},
  author={Jha, Abhinav and Saxena, Akshin and Garg, Akshat},
  year={2025},
  institution={Delhi Technological University},
  department={Electrical Engineering},
  type={B.Tech Final Year Project}
}
```

---

<div align="center">

**⚡ Built with Python, PyTorch, and Fuzzy Logic ⚡**

Made with ❤️ by Team DTU EE 2025

</div>
