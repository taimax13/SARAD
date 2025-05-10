import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
from typing import Union
import matplotlib.pyplot as plt


class RXDetector:
    def __init__(self, input_data: Union[str, Path], output_dir: Union[str, Path]):
        self.input_data = Path(input_data)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_2d_safe(self, arr: np.ndarray, name: str) -> np.ndarray:
        print(f"👀 Pre-check {name}: shape={arr.shape}, dtype={arr.dtype}")

        if arr.ndim == 0:
            raise ValueError("0D scalar patch.")
        elif arr.ndim == 1:
            return np.expand_dims(arr, axis=0)
        elif arr.ndim == 2:
            return arr
        elif arr.ndim == 3 and arr.shape[-1] in [1, 2]:
            return arr
        elif arr.ndim > 3:
            arr = arr.squeeze()
            return self.to_2d_safe(arr, name)
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    def compute_rx_map(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            reshaped = image.reshape(-1, 1).astype(np.float64)
        elif image.ndim == 3 and image.shape[-1] == 2:
            reshaped = image.reshape(-1, 2).astype(np.float64)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if np.isnan(reshaped).all() or np.std(reshaped, axis=0).min() == 0:
            raise ValueError("Flat or NaN-only patch — RX cannot be computed.")

        mean = np.mean(reshaped, axis=0)
        cov = np.cov(reshaped, rowvar=False)

        if cov.ndim < 2:
            raise ValueError("Covariance matrix is degenerate or 0D.")

        inv_cov = np.linalg.pinv(cov)
        scores = [distance.mahalanobis(p, mean, inv_cov) for p in reshaped]
        return np.array(scores).reshape(image.shape[:2])

    def compute_max_score(self, rx_map: np.ndarray) -> float:
        return float(np.max(rx_map))

    def save_rx_heatmap(self, rx_map: np.ndarray, image_name: str, fmt: str = "npz"):
        out_path = self.output_dir / f"{image_name}_rx.{fmt}"
        if fmt == "npz":
            np.savez_compressed(out_path, rx_map=rx_map)
        elif fmt == "npy":
            np.save(out_path, rx_map)
        elif fmt == "png":
            plt.figure(figsize=(6, 6))
            plt.imshow(rx_map, cmap='hot')
            plt.colorbar(label='RX Score')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        else:
            raise ValueError(f"Unsupported format '{fmt}'.")


def main2():
    input_npy = "/Users/talexm/PyProcessing/AnomalyDetector /SARAD/data_collector/data/collected_sar_array.npy"
    output_dir = "./output/rx_heatmaps"
    save_format = "npz"
    top_n = 5

    patches = np.load(input_npy)
    print(f"🚀 Starting RX anomaly detection on stacked array of shape: {patches.shape}")

    rx = RXDetector(input_data=input_npy, output_dir=output_dir)
    results = {}

    for i, patch in enumerate(patches):
        try:
            patch_2d = rx.to_2d_safe(patch, f"patch_{i}")
            rx_map = rx.compute_rx_map(patch_2d)
            max_score = rx.compute_max_score(rx_map)
            rx.save_rx_heatmap(rx_map, f"patch_{i}", fmt=save_format)
            results[f"patch_{i}"] = max_score
        except Exception as e:
            print(f"⚠️ Skipping patch_{i}: {type(e).__name__}: {e}")

    df = pd.DataFrame(list(results.items()), columns=["Patch", "Max_RX_Score"])
    csv_path = Path(output_dir) / "rx_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"📁 RX scores saved to CSV: {csv_path}")

    #top_anomalies = df.sort_values(by="Max_RX_Score", ascending=False).head(top_n)

    mean_score = df["Max_RX_Score"].mean()
    std_score = df["Max_RX_Score"].std()
    threshold = df["Max_RX_Score"].quantile(0.95)

    print(f"\n📊 RX Anomaly Threshold: mean={mean_score:.4f}, std={std_score:.4f}, threshold={threshold:.4f}")

    df["is_anomaly"] = df["Max_RX_Score"] > threshold
    top_anomalies = df[df["is_anomaly"]].copy()

    print(f"🚨 Found {len(top_anomalies)} statistically significant anomalies (score > mean + 2*std)")

    # Or optionally: score > 95th percentile
    # threshold = df["Max_RX_Score"].quantile(0.95)

    print(f"\n👀 Top {top_n} anomalies:")
    for _, row in top_anomalies.iterrows():
        print(f"{row['Patch']}: {row['Max_RX_Score']:.4f}")
        rx_path = Path(output_dir) / f"{row['Patch']}_rx.{save_format}"
        if rx_path.exists() and save_format == "png":
            img = plt.imread(rx_path)
            plt.imshow(img, cmap='hot')
            plt.title(row['Patch'])
            plt.colorbar()
            plt.axis('off')
            plt.show()

    # Show the top N anomaly maps
    print(f"\n📸 Displaying top {top_n} anomaly heatmaps with original SAR patches:")
    for _, row in top_anomalies.iterrows():
        patch_id = row['Patch']
        rx_path = Path(output_dir) / f"{patch_id}_rx.{save_format}"
        patch_idx = int(patch_id.split("_")[1])  # Extract number from "patch_XX"

        # Load RX map
        if not rx_path.exists():
            print(f"⚠️ Missing RX file for {patch_id}")
            continue

        if save_format == "npz":
            rx_map = np.load(rx_path)["rx_map"]
        elif save_format == "npy":
            rx_map = np.load(rx_path)
        else:
            print(f"⚠️ Unsupported format: {save_format}")
            continue

        # Load original SAR patch (VV + VH)
        original_patch = patches[patch_idx]  # shape: (H, W, 2)

        # Display side by side
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))

        axs[0].imshow(original_patch[..., 0], cmap='gray')
        axs[0].set_title(f"{patch_id} - VV Band")
        axs[0].axis('off')

        axs[1].imshow(original_patch[..., 1], cmap='gray')
        axs[1].set_title(f"{patch_id} - VH Band")
        axs[1].axis('off')

        axs[2].imshow(rx_map, cmap='hot')
        axs[2].set_title(f"RX Score: {row['Max_RX_Score']:.4f}")
        axs[2].axis('off')

        plt.suptitle(f"Patch: {patch_id}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # 🎯 Find a "normal" patch (score close to median)
    median_score = df["Max_RX_Score"].median()
    df["score_diff_from_median"] = (df["Max_RX_Score"] - median_score).abs()
    normal_patch_row = df.sort_values("score_diff_from_median").iloc[0]
    normal_patch_id = normal_patch_row["Patch"]
    normal_patch_idx = int(normal_patch_id.split("_")[1])

    print(f"\n🔍 Comparing anomaly vs normal:")
    print(f"🔥 Anomaly: {top_anomalies.iloc[0]['Patch']} - Score: {top_anomalies.iloc[0]['Max_RX_Score']:.4f}")
    print(f"✅ Normal:  {normal_patch_id} - Score: {normal_patch_row['Max_RX_Score']:.4f}")

    # Load anomaly
    anomaly_patch_id = top_anomalies.iloc[0]["Patch"]
    anomaly_idx = int(anomaly_patch_id.split("_")[1])
    anomaly_patch = patches[anomaly_idx]
    anomaly_rx_path = Path(output_dir) / f"{anomaly_patch_id}_rx.{save_format}"
    anomaly_rx = np.load(anomaly_rx_path)["rx_map"] if save_format == "npz" else np.load(anomaly_rx_path)

    # Load normal
    normal_patch = patches[normal_patch_idx]
    normal_rx_path = Path(output_dir) / f"{normal_patch_id}_rx.{save_format}"
    normal_rx = np.load(normal_rx_path)["rx_map"] if save_format == "npz" else np.load(normal_rx_path)

    # Show side-by-side comparison
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    axs[0, 0].imshow(anomaly_patch[..., 0], cmap="gray")
    axs[0, 0].set_title(f"{anomaly_patch_id} - VV (Anomaly)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(anomaly_patch[..., 1], cmap="gray")
    axs[0, 1].set_title(f"{anomaly_patch_id} - VH")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(anomaly_rx, cmap="hot")
    axs[0, 2].set_title(f"RX Score: {top_anomalies.iloc[0]['Max_RX_Score']:.4f}")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(normal_patch[..., 0], cmap="gray")
    axs[1, 0].set_title(f"{normal_patch_id} - VV (Normal)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(normal_patch[..., 1], cmap="gray")
    axs[1, 1].set_title(f"{normal_patch_id} - VH")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(normal_rx, cmap="hot")
    axs[1, 2].set_title(f"RX Score: {normal_patch_row['Max_RX_Score']:.4f}")
    axs[1, 2].axis("off")

    plt.suptitle("Anomaly vs Normal Patch Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main2()
