import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SARAD.models.sar_var_autoencoder import Sampling


class VAEAnomalyEvaluator:
    def __init__(self, vae_model_path: str, threshold: float = None):
        self.model = tf.keras.models.load_model(vae_model_path, custom_objects={"Sampling": Sampling}, safe_mode=False)
        self.input_shape = self.model.input_shape[1:]
        self.threshold = threshold
        if threshold is None:
            self.threshold = 0.1  # Default threshold if none computed yet
            print(f"⚠️ No threshold provided. Using default {self.threshold}")

    def create_synthetic_anomaly_set(self, source_folder: str, dest_folder: str, anomaly_fraction: float = 0.2):
        """
        Copies patches from source_folder to dest_folder.
        Randomly corrupts a fraction of the images to simulate anomalies.
        """
        import shutil
        from skimage.util import random_noise

        source_folder = Path(source_folder)
        dest_folder = Path(dest_folder)
        dest_folder.mkdir(parents=True, exist_ok=True)

        patch_files = list(source_folder.glob("*.npy"))
        total = len(patch_files)
        num_anomalies = int(total * anomaly_fraction)

        np.random.seed(42)
        anomaly_indices = np.random.choice(total, size=num_anomalies, replace=False)

        for idx, patch_file in enumerate(patch_files):
            img = np.load(patch_file)

            # Randomly corrupt some images
            if idx in anomaly_indices:
                patch_file_name = patch_file.stem + "_A.npy"
                # Example corruption: add random noise
                corrupted = random_noise(img, mode='s&p', amount=0.05)  # salt & pepper noise
                corrupted = np.clip(corrupted, 0, 1)
                np.save(dest_folder / patch_file_name, corrupted)
                print(f"🔴 Corrupted and saved: {patch_file_name}")
            else:
                np.save(dest_folder / patch_file.name, img)

        print(f"✅ Created synthetic anomaly set in {dest_folder} with {num_anomalies} corrupted images out of {total}")

    def compute_reconstruction_loss(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        original = original.astype(np.float32) / 255.0
        reconstructed = reconstructed.astype(np.float32) / 255.0

        # Binary crossentropy per pixel
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        loss = loss_fn(original.flatten(), reconstructed.flatten()).numpy()
        return loss

    def normalize_sar(self, img, min_db=-30, max_db=0):
        img = np.clip(img, min_db, max_db)
        return (img - min_db) / (max_db - min_db)


    def evaluate(self, image: np.ndarray) -> dict:
        img = self.normalize_sar(image)
        img = np.expand_dims(img, axis=(0, -1)) if img.ndim == 2 else np.expand_dims(img, axis=0)

        reconstructed = self.model.predict(img, verbose=0)[0]
        recon_loss = self.compute_reconstruction_loss(img[0], reconstructed)
        is_anomalous = recon_loss > self.threshold

        return {
            "reconstruction_loss": recon_loss,
            "is_anomalous": is_anomalous
        }

    def batch_evaluate_folder(self, folder_path: str) -> pd.DataFrame:
        patch_files = sorted(Path(folder_path).glob("*.npy"))
        results = []

        for patch_file in patch_files:
            img = np.load(patch_file)
            result = self.evaluate(img)
            result["file"] = patch_file.name
            results.append(result)

        df = pd.DataFrame(results)
        output_path = Path("models/output/vae_eval.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ VAE eval metrics saved to {output_path}")
        return df

    def compute_threshold_from_training(self, training_folder: str, std_multiplier: float = 3.0):
        patch_files = sorted(Path(training_folder).glob("*.npy"))
        losses = []

        for patch_file in patch_files:
            img = np.load(patch_file)
            loss = self.evaluate(img)["reconstruction_loss"]
            losses.append(loss)

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        self.threshold = mean_loss + std_multiplier * std_loss
        print(f"📐 Threshold recomputed: {self.threshold:.6f} (mean={mean_loss:.6f}, std={std_loss:.6f})")
        return self.threshold

    def score_new_image(self, image, mean_loss, std_loss):
        """
        Scores a new unseen image:
        - Computes the reconstruction loss (binary crossentropy).
        - Computes the p-value based on the training distribution.
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        if image.shape != self.input_shape:
            raise ValueError(f"Image shape {image.shape} does not match model input shape {self.input_shape}.")

        recon = self.model.predict(np.expand_dims(image, axis=0))[0]

        bce = tf.keras.losses.binary_crossentropy(
            tf.convert_to_tensor(image.flatten()),
            tf.convert_to_tensor(recon.flatten())
        ).numpy().mean()

        from scipy.stats import norm
        p_value = 1 - norm.cdf(bce, loc=mean_loss, scale=std_loss)

        print(f"🔎 Reconstruction BCE Loss: {bce:.5f}")
        print(f"🔎 P-value: {p_value:.5f} (Lower p → More anomalous)")

        return bce, p_value

    def plot_anomaly(self, image, mean_loss, std_loss):
        """
        Plot the original, reconstruction, difference, and p-value.
        """
        import matplotlib.pyplot as plt

        recon = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]

        # Compute p-value
        bce = tf.keras.losses.binary_crossentropy(
            tf.convert_to_tensor(image.flatten()),
            tf.convert_to_tensor(recon.flatten())
        ).numpy().mean()

        from scipy.stats import norm
        p_value = 1 - norm.cdf(bce, loc=mean_loss, scale=std_loss)

        diff = np.abs(image - recon)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(image[..., 0], cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(recon[..., 0], cmap='gray')
        axs[1].set_title("Reconstruction")
        axs[1].axis('off')

        axs[2].imshow(diff[..., 0], cmap='hot')
        axs[2].set_title(f"Difference\np-value: {p_value:.5f}")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"🔎 BCE Loss: {bce:.5f}, p-value: {p_value:.5f}")


def main():
    threshold = 0.005
    evaluator = VAEAnomalyEvaluator(
        vae_model_path="/Users/talexm/models/vae_model.keras", threshold=threshold
    )

    # 1️⃣ Create synthetic anomaly set (test2)
    evaluator.create_synthetic_anomaly_set(
        source_folder="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/patcher/data/patches/test",
        dest_folder="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/patcher/data/patches/test2"
    )

    # 2️⃣ Load train_metrics to get mean_loss and std_loss
    train_metrics = pd.read_csv("/Users/talexm/PyProcessing/AnomalyDetector /SARAD/models/output/train_metrics_new.csv")
    mean_loss = train_metrics["reconstruction_loss"].mean()
    std_loss = train_metrics["reconstruction_loss"].std()

    # 3️⃣ Evaluate the whole test2 set
    test_patches = []
    for patch_file in sorted(Path("/Users/talexm/PyProcessing/AnomalyDetector /SARAD/patcher/data/patches/test2").glob("*.npy")):
        img = np.load(patch_file)
        bce, p_value = evaluator.score_new_image(img, mean_loss, std_loss)
        test_patches.append({
            "file": patch_file.name,
            "reconstruction_loss": bce,
            "p_value": p_value,
            "is_anomalous": p_value < threshold
        })

    df = pd.DataFrame(test_patches)
    df.to_csv("models/output/test2_eval.csv", index=False)
    df["true_anomaly"] = df["file"].str.contains("_A")
    df["predicted_anomaly"] = df["p_value"] < threshold

    # ✅ Show anomalies found
    anomalies = df[df["p_value"] < threshold]

    print("🔎 Anomalies detected:")
    print(anomalies[["file", "reconstruction_loss", "p_value"]])

    from sklearn.metrics import classification_report

    print(classification_report(df["true_anomaly"], df["predicted_anomaly"]))

    # Plot the first 3 anomalies
    for file in anomalies["file"][:3]:
        img = np.load("/Users/talexm/PyProcessing/AnomalyDetector /SARAD/patcher/data/patches/test2/" + file)
        evaluator.plot_anomaly(img, mean_loss, std_loss)

    # Load the CSVs
    train_df = pd.read_csv("/Users/talexm/PyProcessing/AnomalyDetector /SARAD/models/output/train_metrics_new.csv")
    test_df = pd.read_csv("/Users/talexm/PyProcessing/AnomalyDetector /SARAD/evaluators/models/output/test2_eval.csv")

    # Add dataset label
    train_df["dataset"] = "train"
    test_df["dataset"] = "test"

    # Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df["p_value"], color="green", label="Train", kde=True, stat="density", bins=50)
    sns.histplot(test_df["p_value"], color="purple", label="Test", kde=True, stat="density", bins=50)

    plt.title("P-value Distribution — Train vs Test")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
