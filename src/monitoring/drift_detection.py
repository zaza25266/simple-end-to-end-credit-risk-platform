import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src.utils.config import load_params

class ModelDriftMonitor:
    """
    Automated drift detection engine utilizing Evidently AI to audit
    incoming production telemetry against baseline training distributions.
    """
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.params = load_params()

    def run_drift_analysis(self, current_data: pd.DataFrame, report_output_path: str = "reports/drift_report.html"):
        """
        Executes a comprehensive data drift and target drift analysis report.
        """
        print("Initializing Evidently AI drift report generation...")
        
        drift_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset()
        ])

        drift_report.run(reference_data=self.reference_data, current_data=current_data)
        
        # Save interactive HTML audit report for compliance and ML engineers
        drift_report.save_html(report_output_path)
        print(f"Drift monitoring report generated successfully and saved to {report_output_path}")
        
        # Extract summary metrics for programmatic alerting
        report_dict = drift_report.as_dict()
        dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
        
        if dataset_drift:
            print("WARNING: Significant data drift detected in production telemetry compared to training baseline!")
        else:
            print("INFO: No significant data drift detected. Production data distribution is stable.")

        return dataset_drift

if __name__ == "__main__":
    # Example execution scaffolding stub
    print("Drift monitor initialized and ready for production telemetry integration.")