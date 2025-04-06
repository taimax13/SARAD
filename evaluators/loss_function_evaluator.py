import numpy as np
import tensorflow as tf
from pathlib import Path

class SiameseAnomalyEvaluator:
    def __init__(self, encoder_model_path: str, reference_embeddings_path: str, threshold: float = 1.0):
        self.encoder = tf.keras.models.load_model(encoder_model_path)
        self.reference_embeddings = np.load(reference_embeddings_path)['embeddings']
        self.threshold = threshold

    def compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute embedding from a single image (normalized to [0,1])"""
        img = image.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # shape: (1, H, W, 1)
        return self.encoder.predict(img, verbose=0)[0]

    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.linalg.norm(emb1 - emb2)

    def evaluate(self, image: np.ndarray) -> float:
        """Compute mean distance to reference clean embeddings"""
        emb = self.compute_embedding(image)
        distances = [self.compute_distance(emb, ref) for ref in self.reference_embeddings]
        return float(np.mean(distances))

    def is_anomalous(self, image: np.ndarray) -> bool:
        """Return True if image is anomalous based on distance threshold"""
        score = self.evaluate(image)
        return score > self.threshold


# # Save to SARAD module
# siamese_path = Path("sarad/siamese_evaluator.py")
# siamese_path.parent.mkdir(parents=True, exist_ok=True)
# siamese_path.write_text(siamese_placeholder_code.strip())
# siamese_path.absolute()