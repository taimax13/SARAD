import os.path
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Reshape, Conv2DTranspose

from tensorflow.keras.layers import Layer
from tensorflow import keras
import tensorflow as tf

from sarad.evaluators.csv_anomaly_evaluator import get_project_root


@keras.saving.register_keras_serializable()
class Sampling(Layer):
    """Sampling layer using (mean, log_var)"""
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

@keras.saving.register_keras_serializable()
class KLDivergenceLayer(Layer):
    """Identity transform layer that adds KL divergence to the final model loss."""
    def call(self, inputs):
        mean, log_var = inputs
        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch))
        return inputs

@keras.saving.register_keras_serializable()
def dummy_kl_loss(y_true, y_pred):
    """Dummy KL loss to satisfy Keras API for second output."""
    return K.zeros_like(y_pred)

class SARVarAutoencoderTrainer:
    def __init__(self, patch_dir: str, save_path: "models/vae_model.h5"):
        self.patch_dir = Path(patch_dir)
        self.save_path = Path(save_path)
        self.model = None
        self.X_train = None
        self.X_val = None
        self.input_shape = None

    def load_data(self, limit=None):
        print(f"🔍 Looking for .npy files in: {self.patch_dir}")
        files = list(self.patch_dir.glob("*.npy"))
        if limit:
            files = files[:limit]
        if not files:
            raise ValueError(f"No patch files found in {self.patch_dir}")

        # Normalize SAR patches from dB scale to [0, 1]
        def normalize_sar(img, min_db=-30, max_db=0):
            img = np.clip(img, min_db, max_db)
            return (img - min_db) / (max_db - min_db)

        patches = [normalize_sar(np.load(f)) for f in files]
        patches = np.array(patches).astype(np.float32)

        # Auto reshape: (N, H, W) → (N, H, W, 1)
        if patches.ndim == 3:
            patches = np.expand_dims(patches, -1)

        self.input_shape = patches.shape[1:]  # e.g., (128, 128, 2)
        print(f"✅ Final patch shape: {patches.shape} — input shape: {self.input_shape}")

        self.X_train, self.X_val = train_test_split(patches, test_size=0.1, random_state=42)
        print(f"📦 Loaded {len(patches)} patches → Train: {len(self.X_train)}, Val: {len(self.X_val)}")

    def build_model(self, input_shape=None, n_layers=4, latent_dim=64):
        if input_shape is None:
            input_shape = self.input_shape

        inputs = Input(shape=input_shape)
        x = inputs

        # ENCODER
        filters = 64
        for _ in range(n_layers):
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            filters *= 2  # grow deeper layers

        shape_before_flattening = K.int_shape(x)[1:]
        x_flat = Flatten()(x)

        # LATENT SPACE
        mean = Dense(latent_dim, name="z_mean")(x_flat)
        log_var = Dense(latent_dim, name="z_log_var")(x_flat)
        z = Sampling()([mean, log_var])

        # DECODER
        x = Dense(np.prod(shape_before_flattening))(z)
        x = Reshape(target_shape=shape_before_flattening)(x)

        filters //= 2
        for _ in range(n_layers):
            x = Conv2DTranspose(filters, (3, 3), strides=2, activation='relu', padding='same')(x)
            filters //= 2

        outputs = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

        self.model = model

    def train(self, epochs=50, batch_size=16):
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        early_stop = EarlyStopping(patience=10, restore_best_weights=True)

        history = self.model.fit(
            self.X_train, self.X_train,
            validation_data=(self.X_val, self.X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )

        Path("output").mkdir(exist_ok=True)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("output/training_history.csv", index=False)
        print("✅ Training history saved to output/training_history.csv")

        self.plot_history(history)

        # ✅ Evaluate train set after training -> for calibration
        eval_df = self.evaluate_set(dataset=self.X_train, set_name="train", save_path=None)

        # Calculate mean and std of reconstruction loss
        mean_loss = eval_df['reconstruction_loss'].mean()
        std_loss = eval_df['reconstruction_loss'].std()

        print(f"📊 Mean reconstruction loss (BCE): {mean_loss:.5f}, Std: {std_loss:.5f}")

        from scipy.stats import norm
        eval_df['p_value'] = 1 - norm.cdf(eval_df['reconstruction_loss'], loc=mean_loss, scale=std_loss)

        # ✅ Save final eval_df with p-values
        eval_df.to_csv("output/train_metrics_new.csv", index=False)
        print("✅ Train metrics with p-values saved to output/train_metrics.csv")

        return eval_df

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

    def save_model(self):
        if not isinstance(self.save_path, Path):
            self.save_path = Path(self.save_path).expanduser().resolve()

        save_file = self.save_path.with_suffix('.keras')
        save_file.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_file))
        print(f"💾 Model saved to {save_file}")

    def show_reconstruction(self, num_samples=3):
        import random

        def normalize_for_display(img):
            img_min = np.min(img)
            img_max = np.max(img)
            return (img - img_min) / (img_max - img_min + 1e-5)

        if self.X_val is None or len(self.X_val) == 0:
            print("⚠️ No validation data available.")
            return

        actual_samples = min(num_samples, len(self.X_val))
        print(f"🖼️ Showing {actual_samples} reconstruction{'s' if actual_samples > 1 else ''}")
        idxs = random.sample(range(len(self.X_val)), actual_samples)
        recons = self.model.predict(self.X_val[idxs])

        for i in range(actual_samples):
            original = self.X_val[idxs[i]][..., 0]
            reconstructed = recons[i][..., 0]
            diff = np.abs(original - reconstructed)

            original = normalize_for_display(original)
            reconstructed = normalize_for_display(reconstructed)
            diff = normalize_for_display(diff)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(original, cmap='gray')
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(reconstructed, cmap='gray')
            axs[1].set_title("Reconstruction")
            axs[1].axis('off')

            axs[2].imshow(diff, cmap='hot')
            axs[2].set_title("Difference")
            axs[2].axis('off')

            plt.tight_layout()
            plt.show()

    def evaluate_set(self, dataset=None, set_name="validation", save_path=Path("output/validation_metrics.csv")):
        if dataset is None:
            dataset = self.X_val
            set_name = "validation"

        if dataset is None or len(dataset) == 0:
            raise ValueError(f"{set_name.capitalize()} data is not available.")

        print(f"📊 Evaluating {set_name} set reconstructions...")
        reconstructions = self.model.predict(dataset)

        stats = []

        for idx in range(len(dataset)):
            original = dataset[idx]
            recon = reconstructions[idx]

            # Compute the binary crossentropy (same as training loss)
            bce = tf.keras.losses.binary_crossentropy(
                tf.convert_to_tensor(original.flatten()),
                tf.convert_to_tensor(recon.flatten())
            ).numpy().mean()  # Average over all pixels

            mse = np.mean((original - recon) ** 2)
            mae = np.mean(np.abs(original - recon))
            diff = np.abs(original - recon)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)

            stats.append({
                "index": idx,
                "reconstruction_loss": bce,  # BCE, matches training loss
                "mse_loss": mse,
                "mae_loss": mae,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "channel_count": original.shape[-1],
            })

        df = pd.DataFrame(stats)

        if save_path is None:
            save_path = Path(f"output/{set_name}_metrics.csv")
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ {set_name.capitalize()} set metrics saved to {save_path}")

        return df


def main():
    trainer = SARVarAutoencoderTrainer(patch_dir=os.path.join(get_project_root(), "patcher/data/patches/test"), save_path="/home/talexm/models/vae_model.h5")
    trainer.load_data()
    trainer.build_model(n_layers=3)
    trainer.train(epochs=10)
    trainer.save_model()
    trainer.show_reconstruction()
    eval_df = trainer.evaluate_set()
    print(eval_df.head())


if __name__ == "__main__":
    main()
