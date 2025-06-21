import os
import argparse
import numpy as np
import tensorflow as tf
import pickle
import logging
from tensorflow.keras.models import load_model
from sar_siames_network import l2_norm
import keras
keras.config.enable_unsafe_deserialization()


def extract_label_from_filename(filename):
    """
    Labels: 1 if '_A.npy' (anomalous), else 0 (normal).
    """
    return 1 if filename.endswith("_A.npy") else 0


def build_db(patch_folder, model_path, db_output_path):
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path, custom_objects={"l2_norm": l2_norm})

    patch_paths = [
        os.path.join(patch_folder, f) 
        for f in os.listdir(patch_folder) 
        if f.endswith(".npy")
    ]
    logging.info(f"Found {len(patch_paths)} patches to process")

    embeddings = []
    labels = []

    for path in patch_paths:
        patch = np.load(path)
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        embedding = model.predict(patch, verbose=0)
        label = extract_label_from_filename(os.path.basename(path))

        embeddings.append(embedding[0])
        labels.append(label)

        logging.debug(f"{os.path.basename(path)} → Label: {label}")

    db = {
        "embeddings": np.array(embeddings),
        "labels": np.array(labels)
    }

    with open(db_output_path, "wb") as f:
        pickle.dump(db, f)

    logging.info(f"Saved DB pickle to {db_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", required=True, help="Folder with patches to add to DB")
    parser.add_argument("--model_path", required=True, help="Path to trained base CNN (.keras)")
    parser.add_argument("--db_path", required=True, help="Output path for DB pickle file")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), "INFO"),
        format="%(asctime)s — %(levelname)s — %(message)s"
    )

    build_db(
        patch_folder=args.patch_folder,
        model_path=args.model_path,
        db_output_path=args.db_path
    )

    #TO-DO: What samples we want our embedding DB to contain? 50%-50% anomaly-normal patches of diff images?
    #Curentlly contain 192 patches of 4 images - 155 normal - 37 anomaly