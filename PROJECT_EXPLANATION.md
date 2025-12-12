# Neuro-Fuzzy Load Flow Estimation: Complete Project Guide

## Complete Project Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA GENERATION                         │
│                  (data_generation/main.py)                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  IEEE 33-Bus System  │
                    │  (Pandapower)        │
                    │  - 33 buses          │
                    │  - 12.66 kV network  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │  Generate 5,000 Scenarios   │
                    │  • Vary loads (±20%)        │
                    │  • Random line outages      │
                    │  • Run power flow solution  │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────┐
                    │  Add Realistic Conditions   │
                    │  • 30-70% sensors missing   │
                    │  • 5-10% Gaussian noise     │
                    │  • Simulate disaster damage │
                    └──────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │  Save Datasets                      │
                    │  ✓ sensor_inputs_ieee_33-bus.csv    │
                    │    (5000×20: sparse measurements)   │
                    │  ✓ grid_states_ieee_33-bus.csv      │
                    │    (5000×66: complete grid states)  │
                    └──────────┬──────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────────┐
│                     PHASE 2: PREPROCESSING                           │
│              (src/fuzzy_preprocessor.py + src/train.py)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
        ┌───────────────────┐   ┌────────────────────┐
        │  Fuzzy Logic      │   │  Baseline ANN Path │
        │  Preprocessing    │   │  (for comparison)  │
        └─────────┬─────────┘   └──────────┬─────────┘
                  │                        │
                  ▼                        │
    ┌──────────────────────────┐          │
    │  Define 4 Fuzzy Variables│          │
    │  • Voltage (low/normal/  │          │
    │    high)                 │          │
    │  • Current (low/med/high)│          │
    │  • Power (low/med/high)  │          │
    │  • Availability (sparse/ │          │
    │    medium/dense)         │          │
    └─────────┬────────────────┘          │
              │                            │
              ▼                            │
    ┌──────────────────────────┐          │
    │  Apply 13 Fuzzy Rules    │          │
    │  Examples:               │          │
    │  IF voltage=normal AND   │          │
    │     avail=dense          │          │
    │  THEN confidence=high    │          │
    │                          │          │
    │  IF avail=sparse         │          │
    │  THEN quality=poor       │          │
    └─────────┬────────────────┘          │
              │                            │
              ▼                            │
    ┌──────────────────────────┐          │
    │  Generate 12 Fuzzy       │          │
    │  Features per Sample:    │          │
    │  1. V_confidence         │          │
    │  2. I_confidence         │          │
    │  3. P_confidence         │          │
    │  4. V_quality            │          │
    │  5. I_quality            │          │
    │  6. P_quality            │          │
    │  7. V_statistical        │          │
    │  8. I_statistical        │          │
    │  9. P_statistical        │          │
    │  10. Availability %      │          │
    │  11. Noise estimate      │          │
    │  12. Consistency         │          │
    └─────────┬────────────────┘          │
              │                            │
              ▼                            │
    ┌──────────────────────────┐          │
    │  Save Fuzzy Preprocessor │          │
    │  fuzzy_preprocessor.pkl  │          │
    └─────────┬────────────────┘          │
              │                            │
              └────────────┬───────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                     PHASE 3: MODEL PREPARATION                       │
│                      (src/train.py)                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  NEURO-FUZZY MODEL   │  │  BASELINE ANN MODEL  │
        │                      │  │                      │
        │  Input: 52 features  │  │  Input: 40 features  │
        │  • 20 sensors        │  │  • 20 sensors        │
        │  • 20 masks          │  │  • 20 masks          │
        │  • 12 fuzzy ✨       │  │  • (no fuzzy)        │
        │                      │  │                      │
        │  Architecture:       │  │  Architecture:       │
        │  52→128→256→128→66   │  │  40→128→256→128→66   │
        │                      │  │                      │
        │  Parameters: 81,218  │  │  Parameters: 78,592  │
        └──────────┬───────────┘  └──────────┬───────────┘
                   │                         │
                   └──────────┬──────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Data Preparation    │
                   │  • KNN imputation    │
                   │    (fill missing)    │
                   │  • Binary masks      │
                   │    (track present)   │
                   │  • Normalization     │
                   │    (standardize)     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Train/Val Split     │
                   │  • Train: 4,000 (80%)│
                   │  • Val: 1,000 (20%)  │
                   └──────────┬───────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                     PHASE 4: TRAINING                                │
│                      (src/train.py)                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Train Neuro-Fuzzy   │  │  Train Baseline      │
        └──────────┬───────────┘  └──────────┬───────────┘
                   │                         │
                   │  Configuration:         │
                   │  • Adam optimizer       │
                   │  • LR: 0.001           │
                   │  • Batch size: 64       │
                   │  • Max epochs: 100      │
                   │  • Early stopping: 15   │
                   │  • Weighted MSE Loss:   │
                   │    (voltage×2 + angle×1)│
                   │                         │
                   └──────────┬──────────────┘
                              │
                              ▼
                   ┌──────────────────────────┐
                   │  Training Loop           │
                   │  FOR each epoch:         │
                   │    1. Forward pass       │
                   │    2. Compute loss       │
                   │    3. Backward pass      │
                   │    4. Update weights     │
                   │    5. Validate           │
                   │    6. Save if best       │
                   │    7. Check early stop   │
                   └──────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────────┐
                   │  Training Results        │
                   │                          │
                   │  Neuro-Fuzzy:            │
                   │  ✓ Best val loss: 1.548  │
                   │  ✓ Best epoch: 33        │
                   │  ✓ Time: ~12 minutes     │
                   │                          │
                   │  Baseline:               │
                   │  ✓ Best val loss: 1.896  │
                   │  ✓ Best epoch: 21        │
                   │  ✓ Time: ~10 minutes     │
                   │                          │
                   │  🎯 Improvement: 18.38%  │
                   └──────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────────┐
                   │  Save Trained Models     │
                   │  ✓ neurofuzzy_best.pth   │
                   │  ✓ baseline_best.pth     │
                   │  ✓ training_history.json │
                   └──────────┬───────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                     PHASE 5: EVALUATION                              │
│                      (src/evaluate.py)                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                   ┌──────────────────────────┐
                   │  Load Best Models        │
                   │  • Neuro-fuzzy model     │
                   │  • Baseline model        │
                   │  • Fuzzy preprocessor    │
                   └──────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────────┐
                   │  Run Inference on        │
                   │  Validation Set          │
                   │  (1,000 samples)         │
                   └──────────┬───────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────┐
                   │  Compute Metrics                 │
                   │                                  │
                   │  Overall Performance:            │
                   │  • Voltage MAE: 0.000337 pu     │
                   │  • Angle MAE: 0.002281°         │
                   │  • Voltage RMSE: 0.000521 pu    │
                   │  • Inference time: 0.089 ms     │
                   │                                  │
                   │  Per-Bus Analysis:               │
                   │  • Best buses: 0-5              │
                   │  • Worst buses: 28-32           │
                   │                                  │
                   │  Sparsity Impact:                │
                   │  • 30% missing: 0.000280 pu     │
                   │  • 50% missing: 0.000337 pu     │
                   │  • 70% missing: 0.000450 pu     │
                   └──────────┬───────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────┐
                   │  Generate Visualizations         │
                   │  ✓ Training curves              │
                   │  ✓ Error analysis               │
                   │  ✓ Sparsity impact              │
                   │  ✓ Model comparison             │
                   │  ✓ Feature importance           │
                   └──────────┬───────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────┐
                   │  Save Results                    │
                   │  ✓ evaluation_results.json       │
                   │  ✓ prediction_comparison.png     │
                   │  ✓ error_analysis.png            │
                   │  ✓ sparsity_impact.png           │
                   └──────────┬───────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                     PHASE 6: DEPLOYMENT (Optional)                   │
│                      (server.py)                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                   ┌──────────────────────────────────┐
                   │  FastAPI Server                  │
                   │  • Load trained model            │
                   │  • Load fuzzy preprocessor       │
                   │  • Create REST endpoints:        │
                   │    - GET /health                 │
                   │    - POST /predict               │
                   │    - POST /predict/batch         │
                   │    - GET /stats                  │
                   └──────────┬───────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────┐
                   │  Deploy to Cloud                 │
                   │  • Platform: Vercel              │
                   │  • URL: https://...              │
                   │  • API Docs: /docs               │
                   └──────────────────────────────────┘
```

---

## Simple Step-by-Step Explanation

### 📊 PHASE 1: Data Generation (Creating the Dataset)

**What happens:** You create artificial training data since you don't have real disaster scenarios.

**Steps:**

1. **Load IEEE 33-bus system** - A standard test power grid with 33 buses (connection points)
2. **Generate 5,000 different scenarios** by:
   - Randomly varying loads (±20%) - simulating different times of day
   - Randomly removing lines - simulating damaged equipment
   - Running power flow calculations to get the "true" grid state
3. **Simulate disaster conditions:**
   - Randomly remove 30-70% of sensor measurements (NaN values)
   - Add 5-10% noise to remaining measurements
4. **Save two files:**
   - `sensor_inputs_ieee_33-bus.csv` - Sparse, noisy measurements (5000 rows × 20 columns)
   - `grid_states_ieee_33-bus.csv` - Complete true states (5000 rows × 66 columns: 33 voltages + 33 angles)

**Simple analogy:** Like creating a practice exam where you have incomplete answers (inputs) and the full correct answers (outputs).

---

### 🔮 PHASE 2: Fuzzy Logic Preprocessing

**What happens:** You analyze the quality and reliability of your sparse sensor data.

**Steps:**

1. **Define fuzzy membership functions** - Create "fuzzy categories":
   - Voltage: "low" (0.85-0.93), "normal" (0.94-1.00), "high" (1.00-1.05)
   - Availability: "sparse" (<40%), "medium" (40-70%), "dense" (>70%)
   
2. **Create 13 fuzzy rules** - Expert knowledge like:
   - "IF voltage is normal AND availability is dense THEN confidence is high"
   - "IF availability is sparse THEN quality is poor"

3. **Generate 12 fuzzy features for each sample:**
   - 3 confidence scores (voltage, current, power)
   - 3 quality indicators
   - 3 statistical features
   - 3 metadata features (availability, noise, consistency)

4. **Save the fuzzy preprocessor** so you can use it later

**Simple analogy:** Like a quality inspector checking each piece of data and stamping it with a "quality rating" before it goes to the factory (neural network).

---

### 🧠 PHASE 3: Model Building

**What happens:** You create two neural network models to compare.

**Two models:**

**Model 1: Neuro-Fuzzy (Your innovation)**
- Input: 52 features
  - 20 sensor measurements
  - 20 binary masks (which sensors are present)
  - 12 fuzzy quality features ✨ (the secret sauce!)
- Architecture: 52 → 128 → 256 → 128 → 66
- Parameters: 81,218

**Model 2: Baseline ANN (For comparison)**
- Input: 40 features
  - 20 sensor measurements
  - 20 binary masks
  - (No fuzzy features)
- Architecture: 40 → 128 → 256 → 128 → 66
- Parameters: 78,592

**Data preparation:**
- Split data: 80% training (4,000), 20% validation (1,000)
- Fill missing values with KNN imputation (find 5 similar samples, use their average)
- Normalize all features (make them 0-mean, 1-std)

**Simple analogy:** Building two students - one gets extra tutoring (fuzzy features), one doesn't. Let's see who performs better!

---

### 🏋️ PHASE 4: Training

**What happens:** You teach both models to predict grid states from sparse sensors.

**Training configuration:**
- **Optimizer:** Adam (smart gradient descent)
- **Learning rate:** 0.001 (step size)
- **Batch size:** 64 (process 64 samples at once)
- **Loss function:** Weighted MSE (voltage errors count 2× more than angle errors)
- **Early stopping:** Stop if no improvement for 15 epochs

**Training loop (100 epochs max):**

```
For each epoch:
  1. Shuffle training data
  2. For each batch:
     - Forward pass: predict outputs
     - Compute loss: compare predictions vs truth
     - Backward pass: calculate gradients
     - Update weights: improve the model
  3. Validate on validation set
  4. Save if best model so far
  5. Reduce learning rate if stuck
  6. Stop early if no progress
```

**Results:**
- **Neuro-Fuzzy:** Best validation loss = 1.548 at epoch 33 (took ~12 min)
- **Baseline:** Best validation loss = 1.896 at epoch 21 (took ~10 min)
- **Improvement:** 18.38% better! 🎉

**Simple analogy:** Training is like studying for an exam - you practice problems (forward pass), check answers (loss), and learn from mistakes (backward pass). The neuro-fuzzy student does 18% better because of the fuzzy "study guide"!

---

### 📈 PHASE 5: Evaluation

**What happens:** You test how well your trained models perform.

**Metrics computed:**

**1. Accuracy metrics:**
- **Voltage MAE:** 0.000337 pu (0.03% error) - How close are voltage predictions?
- **Angle MAE:** 0.002281° - How close are angle predictions?
- **RMSE:** Root mean squared error (penalizes large errors more)

**2. Per-bus analysis:**
- Buses near the substation (0-5): Lowest error (easy to predict)
- Buses far away (28-32): Higher error (harder to predict)

**3. Sparsity impact:**
- 30% missing data → 0.000280 pu error
- 50% missing data → 0.000337 pu error
- 70% missing data → 0.000450 pu error
- The model gracefully degrades as more sensors fail!

**4. Inference speed:**
- 0.089 milliseconds per prediction
- Fast enough for real-time use!

**Visualizations generated:**
- Training curves (loss over time)
- Error distribution histograms
- Sparsity vs accuracy graphs
- Model comparison bar charts
- Feature importance analysis

**Simple analogy:** Final exam time! You test the student on new problems they haven't seen, grade them on multiple criteria, and make pretty graphs showing their performance.

---

### 🚀 PHASE 6: Deployment (Optional)

**What happens:** You make your model available online so others can use it.

**FastAPI Server:**
- Loads the trained model and fuzzy preprocessor
- Creates web endpoints:
  - `GET /health` - Check if server is running
  - `POST /predict` - Send sensor data, get grid state prediction
  - `POST /predict/batch` - Predict multiple samples at once
  - `GET /stats` - Get model information

**Example API call:**

```
POST /predict
{
  "measurements": [0.98, null, 1.5, null, 2.3, ...]  # 20 values, some null
}

Response:
{
  "voltages": {"bus_0": 1.000, "bus_1": 0.997, ...},
  "angles": {"bus_0": 0.0, "bus_1": -0.152, ...},
  "metadata": {
    "inference_time_ms": 0.089,
    "confidence_score": 0.845,
    "sparsity_percent": 55.0
  }
}
```

**Deploy to cloud:** Upload to Vercel/Render so anyone can access it via URL

**Simple analogy:** Publishing your app on the App Store so others can download and use it!

---

## 🎯 Summary: The Big Picture

1. **Generate data** (5,000 disaster scenarios)
2. **Add fuzzy intelligence** (12 quality features)
3. **Build two models** (with/without fuzzy)
4. **Train both** (teach them to predict)
5. **Evaluate results** (neuro-fuzzy wins by 18.38%!)
6. **Deploy online** (make it usable)

**The key innovation:** Adding fuzzy logic preprocessing gives the neural network extra "context" about data quality, leading to 18.38% better predictions with minimal overhead. This is crucial for disaster scenarios where you need to know which sensors to trust!

---

**End of Project Explanation**
