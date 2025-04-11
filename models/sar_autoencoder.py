import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class SARAutoencoderTrainer:
    def __init__(self, patch_dir: str, save_path: str = "models/sar_autoencoder.h5"):
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

        patches = [np.load(f) for f in files]
        patches = np.array(patches).astype(np.float32) / 255.0

        # Auto reshape: (N, H, W) → (N, H, W, 1)
        if patches.ndim == 3:
            patches = np.expand_dims(patches, -1)

        self.input_shape = patches.shape[1:]  # e.g., (128, 128, 2)
        print(f"✅ Final patch shape: {patches.shape} — input shape: {self.input_shape}")

        self.X_train, self.X_val = train_test_split(patches, test_size=0.1, random_state=42)
        print(f"📦 Loaded {len(patches)} patches → Train: {len(self.X_train)}, Val: {len(self.X_val)}")

    def build_model(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape  # use detected shape

        print(f"🛠️ Building model with input shape: {input_shape}")
        output_channels = input_shape[-1]

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),

            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(output_channels, (3, 3), activation='sigmoid', padding='same')  # dynamically match output channels
        ])
        self.model.compile(optimizer=Adam(1e-3), loss='mse')
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
        self.model.save(self.save_path.with_suffix('.keras'))
        print(f"💾 Model saved to {self.save_path}")

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
        preds = self.model.predict(self.X_val[idxs])

        for i in range(actual_samples):
            original = self.X_val[idxs[i]][..., 0]
            reconstructed = preds[i][..., 0]
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

def main():
    trainer = SARAutoencoderTrainer(patch_dir="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/patcher/data/patches")
    trainer.load_data()
    trainer.build_model()
    trainer.train(epochs=20)
    trainer.save_model()
    trainer.show_reconstruction()

if __name__ == "__main__":
    main()
