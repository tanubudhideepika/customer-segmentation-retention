"""
run_pipeline.py
---------------
Orchestrates the full Customer Segmentation & Retention pipeline using OOP.
Run this after downloading telco_churn.csv.

Steps:
  1. Load, clean, and engineer features (TelcoDataPreprocessor)
  2. Segment customers via KMeans (CustomerSegmentation)
  3. Train XGBoost churn models and score (ChurnPredictor)
  4. Simulate ROI and prioritize campaigns (RetentionSimulator)
"""

import sys
import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

# Import the refactored classes
from preprocess import TelcoDataPreprocessor
from segmentation import CustomerSegmentation
from churn_model import ChurnPredictor
from retention_simulation import RetentionSimulator

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # Update this path if needed
    data_path = r"/Users/nikitha/Desktop/customer-segmentation-retention/data/telco_churn.csv"
    
    print("="*50)
    print(" STARTING CUSTOMER RETENTION PIPELINE")
    print("="*50)

    # Step 1: Preprocessing & Feature Engineering 
    print("\n[1/4] PREPROCESSING DATA...")
    preprocessor = TelcoDataPreprocessor(data_path)
    
    # Run the internal preprocessing pipeline
    df, _ = preprocessor.run_pipeline()
    
    # Extract specifically the RFM features for clustering
    X_rfm = preprocessor.scale_features(["Recency", "Frequency", "Monetary"])
    print(f"Data preprocessed. Shape: {df.shape}")


    # Step 2: Customer Segmentation 
    print("\n[2/4] SEGMENTING CUSTOMERS...")
    # Initialize with 4 clusters based on previous elbow method findings
    segmenter = CustomerSegmentation(df=df, X_scaled=X_rfm, n_clusters=4)
    
    # Runs KMeans, assigns personas, and generates plots
    df, cluster_profiles = segmenter.run_pipeline()
    print("Segmentation complete. Cluster profiles generated.")


    # Step 3: Churn Prediction & Scoring 
    print("\n[3/4] TRAINING CHURN MODELS...")
    predictor = ChurnPredictor(df=df)
    
    # Trains overall model, segment models, plots SHAP, and assigns ChurnScores
    df = predictor.run_pipeline()
    print("Churn modeling complete. Customers scored.")


    # Step 4: Retention ROI Simulation 
    print("\n[4/4] SIMULATING RETENTION ROI...")
    simulator = RetentionSimulator(
        df=df, 
        intervention_lift=0.20,       # 20% of predicted churners saved
        campaign_cost=15.0,           # $15 per customer contacted
        months_retained=6             # Saved customers stay 6 months
    )
    
    # Calculates ROI, generates plots, and prints the executive recommendation
    summary = simulator.run_pipeline()


    # Step 5: Exporting Final Deliverables 
    print("\n EXPORTING FINAL DATASETS...")
    summary.to_csv("outputs/retention_summary.csv", index=False)
    df.to_csv("outputs/scored_customers.csv", index=False)
    
    print("Done! All plots and datasets have been saved to the /outputs/ directory.")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("\n ERROR: Could not find the dataset.")
        print("Please ensure 'telco_churn.csv' is located at:")
        print(r"/Users/nikitha/Desktop/customer-segmentation-retention/data/telco_churn.csv")
    except Exception as e:
        print(f"\n AN ERROR OCCURRED: {str(e)}")