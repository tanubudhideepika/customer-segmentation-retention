"""
KMeans + DBSCAN customer segmentation on RFM features.
Assigns human-readable persona labels to each cluster.
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSegmentation:
    def __init__(self, df: pd.DataFrame, X_scaled: np.ndarray, n_clusters: int = 4):
        """Initialize the segmentation pipeline with data and desired cluster count."""
        self.df = df.copy()
        self.X_scaled = X_scaled
        self.n_clusters = n_clusters
        self.model = None
        self.k_results = None
        
        # Ensure output directory exists for plots
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default map (Should be updated after inspecting cluster profiles)
        self.persona_map = {
            0: "High-Value Loyal",
            1: "At-Risk Mid-Tenure",
            2: "New & Exploring",
            3: "Low-Engagement Churner"
        }

    def find_optimal_k(self, k_range=range(2, 9)) -> dict:
        """Calculates inertia and silhouette scores to evaluate optimal k."""
        print("Calculating optimal K...")
        results = {"k": [], "inertia": [], "silhouette": []}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.X_scaled)
            results["k"].append(k)
            results["inertia"].append(km.inertia_)
            results["silhouette"].append(silhouette_score(self.X_scaled, labels))
            
        self.k_results = results
        return self.k_results

    def plot_elbow(self):
        """Plots the Elbow Curve and Silhouette Scores."""
        if not self.k_results:
            print("Run find_optimal_k() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.k_results["k"], self.k_results["inertia"], marker="o")
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")

        axes[1].plot(self.k_results["k"], self.k_results["silhouette"], marker="o", color="green")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Score")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/elbow_silhouette.png", dpi=150)
        print(f"Saved optimal K plots to {self.output_dir}/elbow_silhouette.png")
        plt.close() # Use plt.show() if running in Jupyter

    def run_kmeans(self) -> np.ndarray:
        """Fits KMeans and returns the cluster labels."""
        print(f"Running KMeans with {self.n_clusters} clusters...")
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.model.fit_predict(self.X_scaled)
        return labels

    def assign_personas(self, labels: np.ndarray) -> pd.DataFrame:
        """Maps cluster integers to human-readable string personas."""
        self.df["Cluster"] = labels
        self.df["Persona"] = self.df["Cluster"].map(self.persona_map)
        return self.df

    def profile_clusters(self, rfm_cols=["Recency", "Frequency", "Monetary"]) -> pd.DataFrame:
        """Returns summary stats per cluster to help validate persona names."""
        cols_to_check = [col for col in rfm_cols + ["Churn"] if col in self.df.columns]
        profile = self.df.groupby("Cluster")[cols_to_check].agg(["mean", "count"])
        return profile

    def plot_segment_distribution(self):
        """Plots the count of customers in each persona segment."""
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x="Persona", order=self.df["Persona"].value_counts().index)
        plt.title("Customer Segments Distribution")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/segment_distribution.png", dpi=150)
        print(f"Saved distribution plot to {self.output_dir}/segment_distribution.png")
        plt.close()

    def plot_churn_by_segment(self):
        """Plots the average churn rate for each persona."""
        if "Churn" not in self.df.columns:
            print("'Churn' column missing. Cannot plot churn by segment.")
            return
            
        churn_rate = self.df.groupby("Persona")["Churn"].mean().sort_values(ascending=False)
        churn_rate.plot(kind="bar", color="salmon", figsize=(8, 5))
        plt.title("Churn Rate by Customer Segment")
        plt.ylabel("Churn Rate")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/churn_by_segment.png", dpi=150)
        print(f"Saved churn plot to {self.output_dir}/churn_by_segment.png")
        plt.close()

    def run_pipeline(self):
        """Executes the full clustering and visualization pipeline."""
        # 1. Evaluate K
        self.find_optimal_k()
        self.plot_elbow()
        
        # 2. Run Modeling
        labels = self.run_kmeans()
        
        # 3. Labeling and Profiling
        self.assign_personas(labels)
        profile = self.profile_clusters()
        
        # 4. Final Visuals
        self.plot_segment_distribution()
        self.plot_churn_by_segment()
        
        return self.df, profile


if __name__ == "__main__":
    np.random.seed(42)
    mock_df = pd.DataFrame({
        "Recency": np.random.randint(1, 72, 100),
        "Frequency": np.random.randint(1, 10, 100),
        "Monetary": np.random.uniform(20, 120, 100),
        "Churn": np.random.choice([0, 1], 100)
    })
    
    # Mocking scaled features
    from sklearn.preprocessing import StandardScaler
    mock_scaled = StandardScaler().fit_transform(mock_df[["Recency", "Frequency", "Monetary"]])
    
    segmenter = CustomerSegmentation(df=mock_df, X_scaled=mock_scaled, n_clusters=4)
    
    final_df, cluster_profiles = segmenter.run_pipeline()
    
    print("\nCluster Profiles (Mean Values):")
    print(cluster_profiles)