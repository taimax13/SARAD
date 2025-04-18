import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_score
from pathlib import Path
import pandas as pd

class SiameseAnomalyEvaluator:
    def __init__(self, encoder_model_path: str, reference_embeddings_path: str, threshold: float = None):
        self.encoder = tf.keras.models.load_model(encoder_model_path)
        self.reference_embeddings = np.load(reference_embeddings_path)['embeddings']
        self.threshold = threshold or self.compute_95_confidence_threshold()

    def compute_embedding(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1)) if img.ndim == 2 else np.expand_dims(img, axis=0)
        return self.encoder.predict(img, verbose=0)[0]

    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.linalg.norm(emb1 - emb2)

    def evaluate(self, image: np.ndarray) -> dict:
        emb = self.compute_embedding(image)
        distances = [self.compute_distance(emb, ref) for ref in self.reference_embeddings]
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))
        z_score = (mean_distance - self.train_mean) / (self.train_std + 1e-6)

        return {
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "z_score": z_score,
            "is_anomalous": mean_distance > self.threshold
        }

    def compute_95_confidence_threshold(self) -> float:
        all_distances = []
        for ref in self.reference_embeddings:
            dists = [self.compute_distance(ref, other) for other in self.reference_embeddings if not np.array_equal(ref, other)]
            all_distances.extend(dists)

        self.train_mean = np.mean(all_distances)
        self.train_std = np.std(all_distances)
        threshold = self.train_mean + 2 * self.train_std
        print(f"📐 95% threshold set to: {threshold:.4f} (mean={self.train_mean:.4f}, std={self.train_std:.4f})")
        return threshold

    def batch_evaluate_folder(self, folder_path: str) -> pd.DataFrame:
        patch_files = sorted(Path(folder_path).glob("*.npy"))
        results = []

        for patch_file in patch_files:
            img = np.load(patch_file)
            result = self.evaluate(img)
            result["file"] = patch_file.name
            results.append(result)

        df = pd.DataFrame(results)
        df.to_csv("models/output/siamese_eval.csv", index=False)
        print(f"✅ Siamese eval metrics saved to models/output/siamese_eval.csv")
        return df

    def evaluate_clustering(self):
        try:
            score = silhouette_score(self.reference_embeddings, [0]*len(self.reference_embeddings))
            print(f"📊 Silhouette Score (clustering tightness): {score:.4f}")
            return score
        except Exception as e:
            print(f"⚠️ Could not compute clustering score: {e}")
            return None

def main():
    evaluator = SiameseAnomalyEvaluator(
        encoder_model_path="models/encoder_model.keras",
        reference_embeddings_path="models/reference_embeddings.npz"
    )

    # Run evaluation on test patches (already saved as .npy)
    df = evaluator.batch_evaluate_folder("models/output/test_patches")
    print(df.head())

    # Optional: clustering insight
    evaluator.evaluate_clustering()

if __name__ == "__main__":
    main()
