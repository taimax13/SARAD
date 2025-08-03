from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Reshape
import tensorflow.keras.backend as K
import os.path
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Dense, Input, Lambda, Reshape
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Dropout, Input, Reshape, Conv2DTranspose, Layer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class Sampling(Layer):
    """Sampling layer using (mean, log_var)"""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class Utils:
    def __init__(self):
        pass

    def plot_history(self, history):
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Autoencoder Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    reconstructions = []

    def evaluate_set(self, dataset=None, set_name="validation", valid_items=None, threshold=None, model=None, patches = None):
        global reconstructions
        # if dataset is None:
        #     dataset = X_val
        #     set_name = "validation"

        if valid_items is None:
            raise ValueError("⚠️ Must provide `valid_items` to match patch metadata.")

        print(f"📊 Evaluating {set_name} set reconstructions...")
        reconstructions = model.predict(dataset)

        stats = []

        for idx in range(len(dataset)):
            original = dataset[idx]
            recon = reconstructions[idx]
            meta = patches[idx]
            patch_id = meta.get("patch_id", f"{set_name}_{idx}")
            true_label = int(meta.get("is_anomaly", False))

            # Compute loss metrics
            bce = tf.keras.losses.binary_crossentropy(
                tf.convert_to_tensor(original.flatten()),
                tf.convert_to_tensor(recon.flatten())
            ).numpy().mean()

            mse = np.mean((original - recon) ** 2)
            mae = np.mean(np.abs(original - recon))
            diff = np.abs(original - recon)

            # Predict anomaly (optional threshold logic)
            if threshold is not None:
                predicted = int(bce > threshold)
            else:
                predicted = None  # or leave blank

            stats.append({
                "patch_id": patch_id,
                "true_label": true_label,
                "predicted": predicted,
                "reconstruction_loss": bce,
                "mse_loss": mse,
                "mae_loss": mae,
                "max_diff": np.max(diff),
                "mean_diff": np.mean(diff),
                "std_diff": np.std(diff),
                "channel_count": original.shape[-1],
            })

        df = pd.DataFrame(stats)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(df[df["true_label"] == 1][["patch_id", "reconstruction_loss", "mse_loss", "mae_loss"]])
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        # Save
        save_path = Path(f"kaggle/working/{set_name}_metrics.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ {set_name.capitalize()} set metrics with predictions saved to {save_path}")

        return df

    def prepareLables(self, input_npy, most_common_shape,df,patches):
        valid_items = [item for item in input_npy if item['image'].shape == most_common_shape]
        rx_labels = dict(zip(df["Patch"], df["is_anomaly"]))

        for patch in patches:
            patch["image"] = patch["image"].astype("float32") / 255.0

        # Separate normal and anomaly data
        normal_data = [item for item in patches if "_A" not in item["patch_id"]]
        anomaly_data = [item for item in patches if "_A" in item["patch_id"]]

        # Combine all patch_ids to encode consistently
        all_patch_ids = [item["patch_id"] for item in normal_data + anomaly_data]

        # Fit label encoder on all patch IDs
        patch_id_encoder = LabelEncoder()
        patch_id_encoded = patch_id_encoder.fit_transform(all_patch_ids)

        # Split encoded patch_ids back
        normal_patch_ids_encoded = patch_id_encoded[:len(normal_data)]
        anomaly_patch_ids_encoded = patch_id_encoded[len(normal_data):]

        # Prepare image data and labels
        normal_patches = np.stack([item['image'] for item in normal_data])
        normal_labels = np.zeros(len(normal_patches))  # label = 0

        anomaly_patches = np.stack([item['image'] for item in anomaly_data])
        anomaly_labels = np.ones(len(anomaly_patches))

        return normal_patch_ids_encoded, normal_patches, normal_labels, anomaly_patch_ids_encoded, anomaly_patches, anomaly_labels