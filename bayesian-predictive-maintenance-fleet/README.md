# Bayesian Predictive Maintenance Framework for Fleet Vehicles

This project implements a **Bayesian predictive maintenance framework** for fleet vehicles using probabilistic modeling, uncertainty quantification, and business-oriented analytics.

The goal is to move beyond point predictions and enable **risk-aware maintenance decisions** using real-world fleet telematics data.

The project combines:
- Bayesian inference
- Fleet health analytics
- Predictive maintenance modeling
- Business ROI evaluation
- Power BI dashboards

---

## Project Overview

Fleet maintenance is inherently uncertain due to noisy sensor signals, diverse driving patterns, and incomplete failure observations.

This framework uses **Bayesian predictive modeling** to:
- Estimate failure risk probabilistically
- Quantify uncertainty in predictions
- Identify high-risk vehicles early
- Support proactive and cost-efficient maintenance decisions

---

## Project Structure

    bayesian-predictive-maintenance-fleet/
    │
    ├── notebooks/
    │   └── bayesian_predictive_maintenance.ipynb
    │
    ├── powerbi/
    │   └── (pbix not included due to size constraints)
    │
    ├── data/
    │   └── README.md
    │
    ├── docs/
    │   └── methodology.md
    │
    ├── outputs/
    │   ├── powerbi_report.pdf
    │   └── screenshots/
    │       ├── fleet_overview.png
    │       ├── bayesian_insights.png
    │       └── roi_analysis.png
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Methodology

The predictive maintenance framework is built using Bayesian inference techniques.

Key components include:
- Probabilistic failure modeling
- Posterior distributions over component health
- Uncertainty-aware risk thresholds
- Scenario-based maintenance planning
- Expected cost and ROI estimation

Bayesian modeling enables transparent decision-making by explicitly accounting for uncertainty.

---

## Analytics and Dashboards

The Power BI analysis includes three dashboard views:

### Fleet Overview
- Overall fleet health distribution
- Vehicle risk segmentation
- Identification of high-risk assets

### Bayesian Predictive Insights
- Failure probability distributions
- Credible intervals for predictions
- Risk ranking under uncertainty

### ROI and Business Impact
- Preventive vs reactive maintenance cost comparison
- Maintenance optimization scenarios
- Expected savings using Bayesian decision rules

---

## Power BI Report Access

Due to file size limitations, the interactive Power BI `.pbix` file is not included in this repository.

A full **PDF export of the dashboard** is available here:

    outputs/powerbi_report.pdf

Representative screenshots are included in:

    outputs/screenshots/

This allows reviewers to explore the insights without requiring Power BI Desktop.

---

## Tools and Technologies

- Python (Pandas, NumPy)
- PyMC for Bayesian modeling
- ArviZ for posterior analysis
- Parquet-based data storage
- Power BI for visualization

---

## Data Availability

Raw fleet telematics data is not included due to size and confidentiality constraints.

The notebook and analytics pipeline are reusable for any fleet dataset following the documented schema.

---

## Applications

- Fleet maintenance optimization
- Predictive reliability engineering
- Risk-aware operational decision support
- Bayesian analytics for industrial systems

---

## Author

This project is part of a data analytics and applied machine learning portfolio focused on Bayesian modeling, uncertainty quantification, and business-aligned predictive systems.

