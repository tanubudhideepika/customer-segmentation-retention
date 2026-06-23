import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class RetentionSimulator:
    def __init__(self, df: pd.DataFrame, intervention_lift: float = 0.20, 
                campaign_cost: float = 15.0, months_retained: int = 6):
        """
        Initialize the simulator with data and business assumptions.
        """
        self.df = df.copy()
        self.intervention_lift = intervention_lift
        self.campaign_cost = campaign_cost
        self.months_retained = months_retained
        self.summary = None
        
        # Ensure output directory exists for plots
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def simulate(self) -> pd.DataFrame:
        """
        Simulates the retention campaign and calculates ROI per segment.
        """
        print("Running retention ROI simulation...")
        
        required_cols = ["Persona", "ChurnScore", "MonthlyCharges"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        results = []

        for segment in self.df["Persona"].unique():
            seg = self.df[self.df["Persona"] == segment]

            n_customers = len(seg)
            avg_churn_prob = seg["ChurnScore"].mean()
            avg_monthly_charges = seg["MonthlyCharges"].mean()

            # Expected churners without intervention
            expected_churners = n_customers * avg_churn_prob

            # Customers saved with intervention
            customers_saved = expected_churners * self.intervention_lift

            # Revenue retained
            revenue_retained = customers_saved * avg_monthly_charges * self.months_retained

            # Campaign cost (applied to full segment, assuming we target everyone in it)
            total_campaign_cost = n_customers * self.campaign_cost

            # ROI Calculation
            net_value = revenue_retained - total_campaign_cost
            roi = (net_value / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0

            results.append({
                "Segment": segment,
                "N_Customers": n_customers,
                "Avg_Churn_Prob": round(avg_churn_prob, 3),
                "Expected_Churners": round(expected_churners),
                "Customers_Saved": round(customers_saved),
                "Revenue_Retained_$": round(revenue_retained, 2),
                "Campaign_Cost_$": round(total_campaign_cost, 2),
                "Net_Value_$": round(net_value, 2),
                "ROI_%": round(roi, 1)
            })

        # Sort by best ROI
        self.summary = pd.DataFrame(results).sort_values("ROI_%", ascending=False)
        return self.summary

    def plot_roi(self):
        """Generates and saves the ROI and Revenue visualizations."""
        if self.summary is None:
            print("Please run simulate() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROI by segment
        axes[0].barh(self.summary["Segment"], self.summary["ROI_%"], color="steelblue")
        axes[0].set_title("Retention Campaign ROI by Segment (%)")
        axes[0].set_xlabel("ROI %")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].invert_yaxis() # Highest ROI at the top

        # Revenue retained vs campaign cost
        x = np.arange(len(self.summary))
        width = 0.35
        
        axes[1].bar(x - width/2, self.summary["Revenue_Retained_$"], width, label="Revenue Retained", color="green", alpha=0.7)
        axes[1].bar(x + width/2, self.summary["Campaign_Cost_$"], width, label="Campaign Cost", color="orange", alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.summary["Segment"], rotation=15)
        axes[1].set_title("Revenue Retained vs Campaign Cost ($)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/retention_roi.png", dpi=150)
        print(f"Saved ROI plots to {self.output_dir}/retention_roi.png")
        plt.close()

    def print_recommendation(self):
        """Prints an executive summary of the simulation results."""
        if self.summary is None or self.summary.empty:
            print("No simulation data to recommend from.")
            return
            
        top = self.summary.iloc[0]
        print("\n" + "="*40)
        print(" Retention Strategy Recommendation")
        print("="*40)
        
        if top["ROI_%"] <= 0:
            print("WARNING: No segments yield a positive ROI under current assumptions.")
        else:
            print(f"Priority Segment : {top['Segment']}")
            print(f"Expected ROI     : {top['ROI_%']}%")
            print(f"Customers Saved  : ~{top['Customers_Saved']}")
            print(f"Net Revenue Gain : ${top['Net_Value_$']:,.0f}")
            
        print("\nFull Segment Ranking:")
        print(self.summary[["Segment", "Avg_Churn_Prob", "Customers_Saved", "ROI_%"]].to_string(index=False))
        print("="*40 + "\n")

    def run_pipeline(self) -> pd.DataFrame:
        """Runs the math, plots the charts, and prints the recommendation."""
        self.simulate()
        self.plot_roi()
        self.print_recommendation()
        return self.summary


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    
    mock_scored_df = pd.DataFrame({
        "Persona": np.random.choice(
            ["High-Value Loyal", "New & Exploring", "At-Risk Mid-Tenure", "Low-Engagement Churner"], 
            n_samples, p=[0.3, 0.2, 0.3, 0.2]
        ),
        "ChurnScore": np.random.uniform(0.05, 0.9, n_samples),
        "MonthlyCharges": np.random.uniform(20, 120, n_samples)
    })
    
    # Artificially tweak mock data to make it realistic
    mock_scored_df.loc[mock_scored_df["Persona"] == "High-Value Loyal", "MonthlyCharges"] += 40
    mock_scored_df.loc[mock_scored_df["Persona"] == "High-Value Loyal", "ChurnScore"] -= 0.2
    mock_scored_df.loc[mock_scored_df["Persona"] == "At-Risk Mid-Tenure", "ChurnScore"] += 0.3
    
    # Clip scores to stay between 0 and 1
    mock_scored_df["ChurnScore"] = mock_scored_df["ChurnScore"].clip(0, 1)

    simulator = RetentionSimulator(
        df=mock_scored_df, 
        intervention_lift=0.25, # Assuming a 25% save rate
        campaign_cost=20.0,     # Assuming a $20 cost per customer
        months_retained=12      # Assuming saved customers stay for a full year
    )
    
    final_summary = simulator.run_pipeline()