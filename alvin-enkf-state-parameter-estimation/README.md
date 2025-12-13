# Alvin AUV — Ensemble Kalman Filter (EnKF) for Depth and Parameter Estimation

This project implements an Ensemble Kalman Filter (EnKF) to estimate the vertical state and key physical parameters of the DSV Alvin using real dive telemetry data.

The objective is to demonstrate nonlinear state estimation, system identification, and physics-aware filtering on real-world underwater vehicle data.

The filter jointly estimates:
- Depth (z)
- Vertical velocity (v)
- Vehicle mass offset (Δm)
- Drag coefficient (Cd)
- Reference area (Aref)
- Thruster effectiveness (k_thr)

---

## Problem Setup

The vertical dynamics of the vehicle are modeled as a 1D nonlinear system including:
- Gravity
- Quadratic hydrodynamic drag
- Thruster-induced force

Depth measurements are assimilated online using an Ensemble Kalman Filter with optional **state–parameter augmentation**.

---

## Project Structure

    alvin-enkf-state-parameter-estimation/
    │
    ├── src/
    │   └── alvin_enkf.py
    │
    ├── data/
    │   └── alvin_all.csv
    │
    ├── outputs/
    │   ├── plots/
    │   └── logs/
    │
    ├── config/
    │   └── default.yaml
    │
    ├── docs/
    │   └── enkf_overview.md
    │
    ├── requirements.txt
    └── README.md

---

## Methodology

- Nonlinear continuous-time dynamics integrated using Runge–Kutta (RK4)
- Ensemble Kalman Filter with perturbed observations
- Optional offline least-squares initialization
- Augmented state estimation (states + physical parameters)
- Physically constrained parameters and numerical safety clamps
- Robust handling of noisy real-world sensor data

---

## How to Run

Install dependencies:

    pip install -r requirements.txt

Run the filter:

    python src/alvin_enkf.py

---

## Outputs

- Estimated depth versus observed depth
- Posterior evolution of physical parameters
- Diagnostic plots for convergence and stability analysis

---

## Applications

- Autonomous and Human-Occupied Vehicle (AUV/HOV) modeling
- Digital twin calibration
- Physics-informed state estimation
- Underwater vehicle system identification

---

## Author

This project is part of a data science and applied machine learning portfolio focusing on Bayesian filtering, system identification, and real-world sensor data modeling.
