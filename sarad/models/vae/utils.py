from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Reshape
import tensorflow.keras.backend as K
import os.path
from sklearn.model_selection import train_test_split
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
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import numpy as np

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

        if valid_items is None:
            raise ValueError("⚠️ Must provide `valid_items` to match patch metadata.")

        print(f"📊 Evaluating {set_name} set reconstructions...")
        reconstructions = model.predict(dataset)

        stats = []

        for idx in range(len(dataset)):
            print(f"enaluating:{idx}, data_set_len:{len(dataset)}")
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

        return df, reconstructions

    def prepareDataModel(self, input_npy, most_common_shape,df,patches):
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



        return normal_patches, normal_labels, anomaly_patches, anomaly_labels

    def mark_anomalous_reconstructions(self,reconstructions, X_val, df_stat, top_k=5, method="cosine"):
        """
        Marks samples whose reconstructions deviate the most from the population of reconstructions.

        Args:
            reconstructions (np.ndarray): Reconstructed images from the model
            X_val (np.ndarray): Original validation images (for display only)
            df_stat (pd.DataFrame): Metrics DataFrame to append anomaly info to
            top_k (int): How many top anomalies to flag
            method (str): Distance method to use ("cosine" or "euclidean")

        Returns:
            df_stat: Updated with 'recon_anomaly_score' and 'recon_anomaly_flag'
        """
        # Flatten reconstructions
        recon_vectors = reconstructions.reshape(reconstructions.shape[0], -1)

        # Compute pairwise distance matrix
        if method == "cosine":
            dist_matrix = cosine_distances(recon_vectors)
        elif method == "euclidean":
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(recon_vectors, metric="euclidean")
        else:
            raise ValueError("Unsupported distance method. Use 'cosine' or 'euclidean'.")

        # Sum of distances from all others = anomaly score
        anomaly_scores = dist_matrix.sum(axis=1)
        df_stat["recon_anomaly_score"] = anomaly_scores

        # Flag top_k anomalies
        threshold_idx = np.argsort(anomaly_scores)[-top_k:]
        df_stat["recon_anomaly_flag"] = 0
        df_stat.loc[threshold_idx, "recon_anomaly_flag"] = 1

        print(f"📌 Marked top {top_k} samples with highest reconstruction deviation as anomalies.")

        # Optional: Plot those
        for idx in threshold_idx:
            original = X_val[idx]
            recon = reconstructions[idx]
            diff = np.abs(original - recon)

            plt.figure(figsize=(8, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(original)
            plt.title("Original")

            plt.subplot(1, 3, 2)
            plt.imshow(recon)
            plt.title("Reconstruction")

            plt.subplot(1, 3, 3)
            plt.imshow(diff, cmap="hot")
            plt.title(f"Error Map\nReconstruction Score: {anomaly_scores[idx]:.4f}")
            plt.suptitle(f"Patch ID: {df_stat.iloc[idx]['patch_id']}", fontsize=10)
            plt.tight_layout()
            plt.show()

        return df_stat

    def visualize_top_anomalies(self,df_stat, dataset, reconstructions, top_n=5, cmap="hot"):
        """
        Visualize original, reconstruction, and error map for top N anomalous patches.

        Args:
            df_stat (pd.DataFrame): DataFrame with reconstruction_loss, patch_id, etc.
            dataset (np.ndarray): Original input images
            reconstructions (np.ndarray): Model reconstructions
            top_n (int): Number of samples to visualize
            cmap (str): Colormap for error map
        """
        # Sort by highest reconstruction loss
        df_top = df_stat.sort_values(by="reconstruction_loss", ascending=False).head(top_n)

        for _, row in df_top.iterrows():
            patch_id = row["patch_id"]
            idx = int(patch_id.split("_")[-1])  # Assumes "val_123" format

            original = dataset[idx]
            recon = reconstructions[idx]
            error_map = np.abs(original - recon)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(original)
            axs[0].set_title("Original")

            axs[1].imshow(recon)
            axs[1].set_title("Reconstruction")

            axs[2].imshow(error_map, cmap=cmap)
            axs[2].set_title(f"Error Map\nLoss: {row['reconstruction_loss']:.4f}")
            fig.suptitle(f"Patch ID: {patch_id} | True Label: {row['true_label']}", fontsize=12)
            plt.tight_layout()
            plt.show()

    def show_patch(self, idx, X_val, reconstructions, recon_errors, y_val, y_pred):
        original = X_val[idx]
        recon = reconstructions[idx]
        diff = np.abs(original - recon)
        error_score = recon_errors[idx]
        true_label = y_val[idx]
        predicted_label = y_pred[idx]

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].imshow(original)
        axs[0].set_title(f"🟢 Original\nLabel: {true_label}")
        axs[0].axis('off')

        axs[1].imshow(recon)
        axs[1].set_title(f"🔁 Reconstructed\nError: {error_score:.4f}")
        axs[1].axis('off')

        axs[2].imshow(diff, cmap='hot')
        axs[2].set_title(f"🔥 Error Map\nPred: {predicted_label}")
        axs[2].axis('off')

        plt.suptitle(f"Patch {idx} — {'Anomaly' if predicted_label else 'Normal'}", fontsize=14)
        plt.tight_layout()
        plt.show()