# Customer Segmentation & Retention Analysis

A full data science pipeline that segments telecom customers using RFM + KMeans clustering, predicts churn per segment using XGBoost, and simulates retention campaign ROI to prioritize intervention strategy.

## Business Problem
Telecom companies lose 15–25% of customers annually. Blanket retention campaigns are expensive and inefficient. This project identifies **which customer segments are most at-risk** and which interventions deliver the **highest ROI**.

## Approach

| Stage | Method |
|---|---|
| Segmentation | RFM feature engineering + KMeans (k=4) |
| Churn Prediction | XGBoost per segment (handles class imbalance) |
| Explainability | SHAP feature importance |
| Retention Strategy | Simulation: revenue retained vs. campaign cost |

## Key Results

**Segmentation**
- Identified 4 distinct customer personas with churn rates ranging from 40% to 65%
- Most at-risk: **Low-Engagement Churner** (65% churn, recent customers with high monthly spend but low service adoption)
- Most stable: **High-Value Loyal** (40% churn, longest tenure, highest service breadth)

**Churn Model**
- Overall XGBoost AUC: **0.60** (limited by small per-segment test sets)
- Best segment model: **High-Value Loyal — AUC 0.81**
- Overall top SHAP drivers: `Contract`, `tenure`, `MonthlyCharges`, `OnlineSecurity`, `TechSupport`
- High-Value Loyal top drivers: `TotalCharges`, `tenure`, `MonthlyCharges`, `Contract` — spend accumulation matters more than contract type for loyal customers

**Retention ROI Simulation**
| Segment | Churn Prob | Customers Saved | ROI |
|---|---|---|---|
| At-Risk Mid-Tenure | 75.3% | 56 | **689.9%** |
| Low-Engagement Churner | 49.5% | 25 | 436.8% |
| New & Exploring | 48.0% | 22 | 391.5% |
| High-Value Loyal | 29.3% | 23 | 385.0% |

- **Priority target: At-Risk Mid-Tenure** — 690% ROI, $41,120 estimated net gain
- All 4 segments ROI-positive; prioritize At-Risk for highest return per campaign dollar

## Project Structure
```
├── data/                  # Raw dataset (not committed)
├── notebooks/             # Pipeline scripts
│   └── run_pipeline.py
├── src/
│   ├── preprocess.py      # Data cleaning & RFM feature engineering
│   ├── segmentation.py    # KMeans clustering & persona labeling
│   ├── churn_model.py     # XGBoost churn classifier + SHAP
│   └── retention_simulation.py  # Campaign ROI simulation
├── outputs/               # Charts & scored customer data
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
# Download telco_churn.csv from Kaggle and place in data/
python notebooks/run_pipeline.py
```

## Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Skills Demonstrated
`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `KMeans Clustering` `RFM Analysis` `Churn Modeling` `Business Simulation`