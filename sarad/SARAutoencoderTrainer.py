import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

# MSE (Mean Squared Error) is simple and pixel-wise — but it treats all pixels equally, even if they’re just small noise or unimportant background.
# and
# SSIM (Structural Similarity Index)


def hybrid_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.5 * mse + 0.5 * ssim


class SARAutoencoderTrainer:
    def __init__(self, patch_dir: str, save_path: str = "sar_autoencoder.h5"):
        self.patch_dir = Path(patch_dir)
        self.save_path = Path(save_path)
        self.model = None
        self.X_train = None
        self.X_val = None

    def load_data(self, limit=None):
        files = list(self.patch_dir.glob("**/*.npy"))
        if limit:
            files = files[:limit]

        valid_patches = []
        for f in files:
            patch = np.load(f)

            if patch.ndim == 2:
                patch = np.expand_dims(patch, axis=-1)  # (128,128,1)

            # Acceptable: (128,128,1) → duplicate channel to make it (128,128,2)
            if patch.shape == (128, 128, 1):
                patch = np.concatenate([patch, patch], axis=-1)

            # Acceptable: already in correct shape
            elif patch.shape == (128, 128, 2):
                pass

            # ❌ All other shapes: skip
            else:
                print(f"⚠️ Skipping malformed patch {f.name} with shape {patch.shape}")
                continue

            valid_patches.append(patch)

        if not valid_patches:
            raise ValueError("❌ No valid patches found in directory!")

        patches = np.array(valid_patches).astype(np.float32) / 255.0

        self.X_train, self.X_val = train_test_split(patches, test_size=0.1, random_state=42)
        print(f"📦 Loaded {len(patches)} patches → Train: {len(self.X_train)}, Val: {len(self.X_val)}")
        return patches

    def build_model(self, input_shape=(256, 256, 1)):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        self.model.compile(optimizer=Adam(1e-3), loss=hybrid_loss)
        print("✅ Autoencoder model compiled.")

    def train(self, epochs=20, batch_size=16):
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        history = self.model.fit(
            self.X_train, self.X_train,
            validation_data=(self.X_val, self.X_val),
            epochs=epochs,
            batch_size=batch_size
        )
        self.plot_history(history)

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
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.save_path)
        print(f"💾 Model saved to {self.save_path}")

def main():
    patch_dir = "path/to/patches"
    model_path = "sarad/models/sar_autoencoder.h5"

    trainer = SARAutoencoderTrainer(patch_dir, model_path)
    trainer.load_data()
    trainer.build_model()
    trainer.train(epochs=20)
    trainer.save_model()


if __name__ == "__main__":
    main()
