# ⚡ Neuro-Fuzzy Load Flow Estimation for Disaster-Resilient Smart Grids

This repository contains the implementation of a B.Tech project focused on **neuro-fuzzy load flow estimation** for **disaster-resilient smart grids**, simulating mobile sensor data for the IEEE 33-bus distribution systems.

The project develops a neuro-fuzzy model to estimate grid states (voltage magnitudes and angles) using sparse, noisy measurements from **mobile sensors** (e.g., IoT devices or drones) in **post-disaster scenarios** (e.g., earthquakes), where fixed sensors may fail. The dataset generation script simulates realistic sensor behavior with **5–10% noise** and **30–50% missing data**, enabling robust training of the model.

---

## 👥 Contributors

- **Abhinav Jha (2K22/EE/10)** — Led data generation, power system modeling, and hardware design  
- **Akshin Saxena (2K22/EE/36)** — Focused on digital logic and data preprocessing  
- **Akshat Garg (2K22/EE/35)** — Contributed to signal processing and dataset validation  

📅 **Date**: July 2025

---

## 📌 Project Overview

The project addresses the challenge of load flow estimation in smart grids under **disaster conditions**, where traditional SCADA systems may be unreliable due to physical damage or communication failures.

Mobile sensors are simulated to collect **sparse measurements** (voltage, current, power flow) at a subset of buses or lines in the IEEE 33-bus systems (12.66 kV). A **neuro-fuzzy model** (combining fuzzy logic and ANNs) processes these measurements to predict full grid states, even under **30–50% data sparsity**.

---

## 🎯 Objectives

- Simulate mobile sensor data for IEEE 33-bus systems
- Generate 5,000 scenarios with sparse, noisy measurements  
  - 5–10 sensors  
  - 5–10% Gaussian noise  
  - 30–50% missing data  
- Develop a neuro-fuzzy model to estimate:
  - Voltage magnitudes \((V_i)\) in pu  
  - Voltage angles \((\theta_i)\) in degrees  
- Validate robustness under disaster conditions

---

## 📐 Mathematical Foundation

The load flow equations at bus \(i\) are:

\[
P_i = V_i \sum_{j=1}^N V_j (G_{ij} \cos(\theta_i - \theta_j) + B_{ij} \sin(\theta_i - \theta_j))
\]

\[
Q_i = V_i \sum_{j=1}^N V_j (G_{ij} \sin(\theta_i - \theta_j) - B_{ij} \cos(\theta_i - \theta_j))
\]

Where:
- \(N = 33\)
- \(G_{ij}, B_{ij}\): Admittance matrix values
- \(P_i, Q_i\): Active/reactive power injections

Sensor measurements are modeled as:

\[
\tilde{x}_k =
\begin{cases}
x_k (1 + \epsilon_k), & \text{if measured} \\
\text{NaN}, & \text{if missing}
\end{cases},
\quad \epsilon_k \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.05–0.1
\]

The neuro-fuzzy model maps sparse inputs to full grid states:

\[
f(\tilde{x}_1, \ldots, \tilde{x}_m) \rightarrow (V_1, \theta_1, \ldots, V_N, \theta_N)
\]

---

## 🧪 Data Generation

The script `generate_ieee_bus_data.py` uses the `pandapower` library to simulate sensor data for the IEEE 33-bus systems.

### Inputs (Sparse Measurements):
- Voltage magnitudes \((V_i)\) in pu
- Current magnitudes \((I_{ij})\) in amps
- Power flows \((P_{ij}, Q_{ij})\) in MW/MVAr

### Outputs (Full Grid States):
- 66 values for IEEE 33-bus (33 voltages, 33 angles)

### Simulation Features:
- Load scaling between **50–150%**
- Noise: **5–10% Gaussian**
- Sparsity: **30–50%**

---

## 📂 Output Files

### IEEE 33-bus:
- `sensor_inputs_ieee_33-bus.csv` — ~5,000 rows, ~20–40 columns
- `grid_states_ieee_33-bus.csv` — ~5,000 rows, 66 columns