"""

Data loading and cleaning for Telco Churn dataset.
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TelcoDataPreprocessor:
    def __init__(self, path: str):
        """Initialize the preprocessor with the dataset path."""
        self.path = path
        self.df = None
        self.scaler = StandardScaler()
        # Storing LabelEncoders is best practice so you can inverse_transform later if needed
        self.label_encoders = {} 

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.path)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        # Fix TotalCharges (whitespace -> NaN -> drop or fill)
        self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors="coerce")
        self.df["TotalCharges"].fillna(self.df["TotalCharges"].median(), inplace=True)

        # Drop customerID (not a feature)
        if "customerID" in self.df.columns:
            self.df.drop(columns=["customerID"], inplace=True)

        # Encode binary target
        if "Churn" in self.df.columns:
            self.df["Churn"] = self.df["Churn"].map({"Yes": 1, "No": 0})

        return self.df

    def encode_categoricals(self) -> pd.DataFrame:
        """Label encode all object columns except target."""
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        
        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le # Save the fitted encoder
            
        return self.df

    def get_rfm_features(self) -> pd.DataFrame:
        """
        Approximate RFM from Telco dataset.
        Returns a separate DataFrame of RFM features.
        """
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        
        # Count services (assumes encoded: > 0 means subscribed)
        self.df["ServiceCount"] = self.df[service_cols].apply(
            lambda row: (row > 0).sum(), axis=1
        )

        rfm = pd.DataFrame({
            "Recency": self.df["tenure"].max() - self.df["tenure"], 
            "Frequency": self.df["ServiceCount"],
            "Monetary": self.df["MonthlyCharges"]
        })
        return rfm

    def scale_features(self, feature_cols: list) -> np.ndarray:
        """Scales specified features and returns the scaled array."""
        X_scaled = self.scaler.fit_transform(self.df[feature_cols])
        return X_scaled

    def run_pipeline(self) -> tuple:
        """A convenience method to run the entire preprocessing pipeline at once."""
        self.load_data()
        self.clean_data()
        self.encode_categoricals()
        
        # Merge RFM features into the main dataframe
        rfm_features = self.get_rfm_features()
        self.df = pd.concat([self.df, rfm_features], axis=1)
        
        # Example of scaling numerical columns
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Recency", "Frequency", "Monetary"]
        X_scaled = self.scale_features(numerical_cols)
        
        return self.df, X_scaled


if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = "/Users/nikitha/Desktop/customer-segmentation-retention/data/telco_churn.csv" 
    
    try:
        # Instantiate the class
        preprocessor = TelcoDataPreprocessor(dataset_path)
        
        # Run the entire pipeline
        final_df, scaled_matrix = preprocessor.run_pipeline()
        
        print("Data preprocessing complete!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{dataset_path}'. Please ensure the path is correct.")