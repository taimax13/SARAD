import numpy as np
from pathlib import Path
from PIL import Image

from SARAD.evaluators.csv_anomaly_evaluator import CSVAnomalyEvaluator
from SARAD.evaluators.sar_model_evaluator import SARModelEvaluator
from SARAD.patcher.sar_patcher import SARPatcher


class MiniCaseCollector:
    def __init__(self, input_folder="/Users/talexm/Desktop/val_data", output_file="data/collected_sar_array.npy", resize_shape=(128, 128)):
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.resize_shape = resize_shape

    def load_images(self):
        image_paths = sorted(self.input_folder.glob("*.png"))
        if not image_paths:
            raise ValueError(f"❌ No .png images found in {self.input_folder}")
        print(f"📸 Found {len(image_paths)} images in {self.input_folder}")

        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("L")  # Grayscale
            img = img.resize(self.resize_shape)
            arr = np.array(img).astype(np.float32)
            images.append(arr)

        return np.stack(images, axis=0)  # (N, H, W)

    def save_to_npy(self):
        images = self.load_images()
        images = np.expand_dims(images, axis=-1)  # (N, H, W, 1)

        # 💡 Duplicate the single channel → (N, H, W, 2)
        images = np.concatenate([images, images], axis=-1)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_file, images)
        print(f"💾 Saved collected SAR array to {self.output_file} with shape {images.shape}")

        return self.output_file


def main():
    collector = MiniCaseCollector(
        input_folder="/Users/talexm/Desktop/val_data",
        output_file="data/collected_sar_array.npy"
    )
    collector.save_to_npy()
    patcher = SARPatcher(
        input_path="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/mini_cases/data/collected_sar_array.npy",
        output_dir="data/patches/test",
        patch_size=128,
        stride=128
    )
    patch_count = patcher.extract_patches()
    print(f"✅ Done. Total patches extracted: {patch_count}")
    evaluator = SARModelEvaluator(
        model_path="/Users/talexm/models/sar_autoencoder.keras",
        patch_dir="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/mini_cases/data/patches/test"
    )

    df = evaluator.evaluate_set(set_name="mini_val", save_path="models/output/mini_val_metrics.csv")
    print(df.head())
    evaluator = CSVAnomalyEvaluator(
        calibration_csv="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/models/output/train_metrics.csv",
        metric_col="mse_loss",
        z_threshold=3
    )

    df = evaluator.evaluate_csv(
        test_csv="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/mini_cases/models/output/mini_val_metrics.csv",
        output_path="output/test_anomaly_results.csv"
    )

    print(df[["index", "mse_loss", "z_score", "is_anomalous_z", "is_anomalous_95"]].head())



if __name__ == "__main__":
    main()
