import os
import argparse
import numpy as np
import logging
import pickle
import csv
from collections import Counter
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
from sarad.models.sar_siames.sar_siames_network import l2_norm


def majority_vote(nearest_labels):
    return Counter(nearest_labels).most_common(1)[0][0]

def evaluate_patches(patch_folder, model_path, db_path, k=5):
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path, custom_objects={"l2_norm": l2_norm})

    logging.info(f"Loading DB from {db_path}")
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    db_embeddings = np.array(db["embeddings"])
    db_labels = np.array(db["labels"])

    patch_paths = [os.path.join(patch_folder, f) for f in os.listdir(patch_folder) if f.endswith(('.npy'))]
    logging.info(f"Found {len(patch_paths)} patches for inference")

    predictions = []

    for path in patch_paths:
        if path.endswith(".npy"):
            patch = np.load(path)
        patch = np.expand_dims(patch, axis=0)

        embedding = model.predict(patch, verbose=0)
        dists = euclidean_distances(embedding, db_embeddings)
        nearest_indices = np.argsort(dists[0])[:k]
        nearest_labels = db_labels[nearest_indices]
        # Weight is how many neighbors are labeled as anomaly (1) divided by k
        anomaly_wheight = np.sum(nearest_labels) / k
        predicted_label = int(anomaly_wheight >= 0.5)  # or keep as soft weight if needed
        predictions.append((os.path.basename(path), predicted_label, float(anomaly_wheight)))
        logging.debug(f"{os.path.basename(path)} → Label: {predicted_label}, Weight: {anomaly_wheight:.2f}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", required=True, help="Folder with test patches")
    parser.add_argument("--model_path", required=True, help="Path to trained base CNN (.keras)")
    parser.add_argument("--db_path", required=True, help="Pickle file with {'embeddings': ..., 'labels': ...}")
    parser.add_argument("--predictions_csv", required=False, help="Path to save predictions.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), "INFO"),
        format="%(asctime)s — %(levelname)s — %(message)s"
    )

    preds = evaluate_patches(
        patch_folder=args.patch_folder,
        model_path=args.model_path,
        db_path=args.db_path,
        k=args.k
    )

    anomaly_count = sum(1 for _, label, _ in preds if label == 1)
    logging.info(f"Total patches: {len(preds)}, Detected anomalies: {anomaly_count}")
    #TO-DO: Apply desicion rule for anomaly image from patches scores? set threshold?

    if args.predictions_csv:
        with open(args.predictions_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted_label", "anomaly_weight"])
            writer.writerows(preds)

        logging.info(f"Saved predictions to {args.predictions_csv}")
