import shap
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import load_params

class ModelExplainer:
    """
    Generates SHAP values and visual summaries for model interpretability,
    crucial for loan officer auditing and compliance.
    """
    def __init__(self, model, X_sample: pd.DataFrame):
        self.model = model
        self.X_sample = X_sample
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self):
        """Computes TreeExplainer SHAP values for tree-based models (CatBoost/XGB/LGBM)."""
        # Using TreeExplainer optimized for tree ensembles
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_sample)
        return self.shap_values

    def plot_summary(self, save_path: str = "reports/shap_summary.png"):
        """Generates and saves a global SHAP beeswarm summary plot."""
        if self.shap_values is None:
            self.compute_shap_values()
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_sample, show=False)
        plt.title("SHAP Global Feature Importance Summary", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot saved successfully to {save_path}")