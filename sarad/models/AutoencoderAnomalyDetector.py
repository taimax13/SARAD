import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union

class AutoencoderAnomalyDetector:
    def __init__(self, model_path: Union[str, Path]):
        """Load trained autoencoder model from .h5 or SavedModel format."""
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run image through autoencoder and return reconstructed version."""
        x = image.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))  # Shape: (1, H, W, 1)
        recon = self.model.predict(x, verbose=0)[0, ..., 0]
        recon = (recon * 255).astype(np.uint8)
        return recon

    def compute_error_map(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Compute per-pixel squared reconstruction error."""
        return (original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2

    def compute_score(self, error_map: np.ndarray, method: str = "mean") -> float:
        """Compute scalar anomaly score from error map."""
        if method == "mean":
            return np.mean(error_map)
        elif method == "max":
            return np.max(error_map)
        else:
            raise ValueError("method must be 'mean' or 'max'")

    def save_error_heatmap(self, error_map: np.ndarray, image_name: str, output_dir: Union[str, Path], fmt: str = "png"):
        """Save reconstruction error heatmap."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_name}_ae_heatmap.{fmt}"
        if fmt == "png":
            plt.figure(figsize=(6, 6))
            plt.imshow(error_map, cmap='hot')
            plt.colorbar(label="Reconstruction Error")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        elif fmt == "npz":
            np.savez_compressed(out_path, error_map=error_map)
        else:
            raise ValueError("Unsupported format: use 'png' or 'npz'")

    def process_image(self, image: np.ndarray, image_name: str, output_dir: Union[str, Path], save_fmt: str = "npz"):
        """Process a single image and return score + error map."""
        recon = self.predict(image)
        error_map = self.compute_error_map(image, recon)
        score = self.compute_score(error_map)
        self.save_error_heatmap(error_map, image_name, output_dir, fmt=save_fmt)
        return score, error_map
