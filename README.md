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
- Identified 4 distinct customer personas with churn rates ranging from X% to X%
- Segment-level XGBoost AUC: 0.XX – 0.XX
- Highest-ROI retention target: **[Segment Name]** — XX% ROI, $XX,XXX estimated net value

*(Update with actual results after running)*

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
python main/run_pipeline.py
```

## Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Skills Demonstrated
`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `KMeans Clustering` `RFM Analysis` `Churn Modeling` `Business Simulation`
