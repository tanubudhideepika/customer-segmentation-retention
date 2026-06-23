import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class ChurnPredictor:
    def __init__(self, df: pd.DataFrame, feature_cols: list = None, target: str = "Churn"):
        """Initialize the ChurnPredictor with data, features, and target."""
        self.df = df.copy()
        self.target = target
        
        # Default feature list if none provided
        self.feature_cols = feature_cols or [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", "ServiceCount"
        ]
        
        # Ensure only columns actually present in df are used
        self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]
        
        self.overall_model = None
        self.segment_models = {}
        
        # Ensure output directory exists for plots
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def train_overall_model(self) -> tuple:
        """Trains a single XGBoost model on the entire dataset."""
        print("Training overall churn model...")
        X = self.df[self.feature_cols]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Calculate imbalance ratio
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.overall_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,  
            random_state=42,
            eval_metric="auc",
            early_stopping_rounds=20,
            verbosity=0
        )

        self.overall_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_prob = self.overall_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Overall AUC: {auc:.4f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, self.overall_model.predict(X_test)))

        return X_test, y_test, y_prob

    def train_per_segment(self) -> dict:
        """Trains a separate XGBoost model for each customer persona."""
        print("\nTraining per-segment models...")
        
        if "Persona" not in self.df.columns:
            print("'Persona' column missing. Run segmentation first.")
            return {}

        self.segment_models = {}
        
        for segment in self.df["Persona"].unique():
            seg_df = self.df[self.df["Persona"] == segment]
            
            # Ensure we have enough data and both churned/retained customers
            if seg_df[self.target].nunique() < 2 or len(seg_df) < 50:
                print(f"Skipping '{segment}' — insufficient data or single class.")
                continue

            X = seg_df[self.feature_cols]
            y = seg_df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, verbosity=0, eval_metric="auc"
            )
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            self.segment_models[segment] = {
                "model": model, 
                "auc": auc, 
                "X_test": X_test, 
                "y_test": y_test
            }
            print(f"Segment: {segment:25s} | AUC: {auc:.4f} | Size: {len(seg_df)}")

        return self.segment_models

    def plot_shap_summary(self, model, X_test: pd.DataFrame, title="Overall"):
        """Generates and saves a SHAP feature importance plot."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {title}")
        plt.tight_layout()
        
        safe_title = title.replace(' ', '_').replace('&', 'and')
        filename = f"{self.output_dir}/shap_{safe_title}.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved SHAP plot to {filename}")
        plt.close()

    def assign_churn_scores(self) -> pd.DataFrame:
        """Appends overall churn probability and risk buckets to the dataframe."""
        if self.overall_model is None:
            print("Train the overall model first.")
            return self.df
            
        self.df["ChurnScore"] = self.overall_model.predict_proba(self.df[self.feature_cols])[:, 1]
        self.df["ChurnRisk"] = pd.cut(
            self.df["ChurnScore"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
        return self.df

    def run_pipeline(self) -> pd.DataFrame:
        """Executes the full modeling, explainability, and scoring pipeline."""
        # 1. Train Overall Model
        X_test_overall, _, _ = self.train_overall_model()
        self.plot_shap_summary(self.overall_model, X_test_overall, title="Overall")
        
        # 2. Train Segment Models
        if "Persona" in self.df.columns:
            self.train_per_segment()
            # Plot SHAP for the first successfully trained segment as an example
            if self.segment_models:
                first_seg = list(self.segment_models.keys())[0]
                self.plot_shap_summary(
                    self.segment_models[first_seg]["model"], 
                    self.segment_models[first_seg]["X_test"], 
                    title=first_seg
                )

        # 3. Assign Scores
        scored_df = self.assign_churn_scores()
        return scored_df


if __name__ == "__main__":

    np.random.seed(42)
    n_samples = 500
    mock_df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n_samples),
        "MonthlyCharges": np.random.uniform(20, 120, n_samples),
        "Contract": np.random.randint(0, 3, n_samples),
        "ServiceCount": np.random.randint(1, 6, n_samples),
        "Persona": np.random.choice(["High-Value Loyal", "New & Exploring", "At-Risk"], n_samples),
        "Churn": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]) # Imbalanced class
    })
    
    mock_features = ["tenure", "MonthlyCharges", "Contract", "ServiceCount"]
    
    predictor = ChurnPredictor(df=mock_df, feature_cols=mock_features)
    final_scored_df = predictor.run_pipeline()
    
    print("\ First 5 Rows of Final Scored Dataset:")
    print(final_scored_df[["Persona", "Churn", "ChurnScore", "ChurnRisk"]].head())