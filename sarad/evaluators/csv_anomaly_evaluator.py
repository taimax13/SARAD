import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class CSVAnomalyEvaluator:
    def __init__(self, calibration_csv: str, metric_col="mse_loss", z_threshold=3):
        self.metric_col = metric_col
        self.calib_df = pd.read_csv(calibration_csv)
        self.z_threshold = z_threshold

        self.train_mean = self.calib_df[metric_col].mean()
        self.train_std = self.calib_df[metric_col].std()
        self.threshold_95 = self.train_mean + 2 * self.train_std

        print(f"✅ Loaded calibration: mean={self.train_mean:.4f}, std={self.train_std:.4f}, 95% threshold={self.threshold_95:.4f}")

    def evaluate_csv(self, test_csv: str, output_path="models/output/anomaly_eval.csv") -> pd.DataFrame:
        df = pd.read_csv(test_csv)
        df["z_score"] = (df[self.metric_col] - self.train_mean) / (self.train_std + 1e-6)
        df["is_anomalous_z"] = df["z_score"] > self.z_threshold
        df["is_anomalous_95"] = df[self.metric_col] > self.threshold_95

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"📊 Evaluation complete. Saved to {output_path}")
        return df


def main():
    evaluator = CSVAnomalyEvaluator(
        calibration_csv = os.path.join(get_project_root(), "models", "output", "validation_metrics.csv"),
        metric_col="mse_loss",
        z_threshold=3
    )

    df = evaluator.evaluate_csv(
        test_csv = os.path.join(get_project_root(), "models", "output", "validation_metrics.csv"),
        output_path="output/test_anomaly_results.csv"
    )

    print(df[["index", "mse_loss", "z_score", "is_anomalous_z", "is_anomalous_95"]].head())


if __name__ == "__main__":
    main()