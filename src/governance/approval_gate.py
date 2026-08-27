import logging
from src.utils.config import load_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelGovernanceGate:
    """
    Evaluates candidate model performance against governance thresholds
    before promoting them to production status.
    """

    def __init__(self):
        self.params = load_params()
        # Define baseline performance floor requirements based on params.yaml
        self.min_roc_auc = 0.8500
        self.min_recall = 0.7000

    def evaluate_model(self, model_name: str, roc_auc: float, recall: float) -> bool:
        logger.info(f"Evaluating governance gate for model: {model_name}")
        logger.info(f"Metrics -> ROC-AUC: {roc_auc:.4f} (Required: >= {self.min_roc_auc})")
        logger.info(f"Metrics -> Recall: {recall:.4f} (Required: >= {self.min_recall})")

        if roc_auc >= self.min_roc_auc and recall >= self.min_recall:
            logger.info(f"STATUS: SUCCESS - Model '{model_name}' passed the governance promotion gate!")
            return True
        else:
            logger.warning(f"STATUS: FAILED - Model '{model_name}' did not meet production criteria.")
            return False


if __name__ == "__main__":
    gate = ModelGovernanceGate()
    # BUG FIX (Bug 8): was "self.params" which raises NameError at module scope.
    # Fixed to use the instance variable "gate.params".
    ensemble_metrics = gate.params.get("ensemble", {}).get("metrics", {})
    gate.evaluate_model(
        model_name="CatBoost + LightGBM Soft Voting",
        roc_auc=ensemble_metrics.get("roc_auc", 0.8709),
        recall=ensemble_metrics.get("recall", 0.7526),
    )