import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from typing import Union
import matplotlib.pyplot as plt

class JointAnomalyClassifier:
    def __init__(self, score_csv_path: Union[str, Path], eps=0.5, min_samples=3):
        self.score_csv = Path(score_csv_path)
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.data = None
        self.labels = None

    def load_scores(self):
        if not self.score_csv.exists():
            raise FileNotFoundError(f"No CSV found at {self.score_csv}")
        df = pd.read_csv(self.score_csv)
        self.data = df
        self.X = df[["Max_RX_Score", "AE_Score"]].values
        print(f"📊 Loaded {len(self.X)} score pairs for clustering.")

    def cluster(self):
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(self.X)
        self.data["Cluster"] = self.labels
        print("✅ Clustering complete.")
        return self.data

    def plot_clusters(self):
        plt.figure(figsize=(8, 6))
        colors = {label: f"C{label}" for label in np.unique(self.labels)}
        for label in np.unique(self.labels):
            points = self.X[self.labels == label]
            plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {label}", alpha=0.7)

        plt.xlabel("RX Score")
        plt.ylabel("AE Score")
        plt.title("DBSCAN Clustering of Anomaly Scores")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_result(self, out_path: str = "clustered_anomalies.csv"):
        self.data.to_csv(out_path, index=False)
        print(f"💾 Clustering results saved to {out_path}")

def main():
    classifier = JointAnomalyClassifier("rx_ae_scores.csv", eps=0.6, min_samples=2)
    classifier.load_scores()
    results = classifier.cluster()
    classifier.plot_clusters()
    classifier.save_result("clustered_anomalies.csv")


if __name__ == '__main__':
    main()