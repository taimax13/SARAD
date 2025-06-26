import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def extract_gt_label(filename):
    """
    Extracts ground truth label from filename.
    Returns 1 if filename ends with '_A.npy' (anomaly), otherwise 0 (normal).
    """
    return 1 if filename.endswith('_A.npy') else 0

def evaluate(csv_path):
    """
    Evaluate predictions stored in a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing predictions.

    Returns:
        None. Prints evaluation metrics to stdout.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {'filename', 'predicted_label', 'anomaly_score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Extract ground truth from filename
    df['gt_label'] = df['filename'].apply(extract_gt_label)

    y_true = df['gt_label']
    y_pred = df['predicted_label']
    y_score = df['anomaly_score']

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    print("📊 Evaluation Metrics:")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  ROC-AUC        : {auc:.4f}")
    print("  Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    """
    evaluate_predictions.py

    This script evaluates binary classification metrics from a CSV file that contains
    model predictions and filenames encoding the ground truth (GT).

    The expected CSV format includes the following columns:
        - filename: the filename of the patch (e.g., '46_patch_2207.npy' or '46_patch_2208_A.npy')
        - predicted_label: model's predicted label (0 = normal, 1 = anomaly)
        - anomaly_score: model's anomaly probability score (float between 0 and 1)

    Ground truth is inferred from the filename:
        - If filename ends with '_A.npy' → ground truth = 1 (anomaly)
        - Otherwise → ground truth = 0 (normal)

    Metrics reported:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - Confusion Matrix
    """
    parser = argparse.ArgumentParser(description="Evaluate anomaly predictions from CSV.")
    parser.add_argument("--csv_path", required=True, help="Path to prediction CSV file")
    args = parser.parse_args()

    evaluate(args.csv_path)
