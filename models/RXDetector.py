import numpy as np
from pathlib import Path
from scipy.spatial import distance
from typing import Union
import cv2

class RXDetector:
    def __init__(self, input_dir: Union[str, Path]):
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, filename: str) -> np.ndarray:
        path = self.input_dir / filename
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {path}")
        return image

    def compute_rx_map(self, image: np.ndarray) -> np.ndarray:
        reshaped = image.reshape(-1, 1).astype(np.float64)
        mean = np.mean(reshaped, axis=0)
        cov = np.cov(reshaped.T)
        inv_cov = np.linalg.pinv(cov)

        scores = [distance.mahalanobis(p, mean, inv_cov) for p in reshaped]
        rx_map = np.array(scores).reshape(image.shape)
        return rx_map

    def compute_max_score(self, rx_map: np.ndarray) -> float:
        return np.max(rx_map)

    def process_image(self, filename: str, save_fmt: str = "npz") -> float:
        print(f"Computing RX for {filename}...")
        image = self.load_image(filename)
        rx_map = self.compute_rx_map(image)
        max_score = self.compute_max_score(rx_map)
        self.save_rx_heatmap(rx_map, Path(filename).stem, fmt=save_fmt)
        print(f"Saved RX map in .{save_fmt} format. Max RX Score: {max_score:.4f}")
        return max_score

    def batch_process(self, save_fmt: str = "npz") -> dict:
        results = {}
        for file in self.input_dir.glob("*.*"):
            try:
                if file.suffix.lower() == ".npy":
                    image = np.load(file)
                    name = file.stem
                else:
                    image = self.load_image(file.name)
                    name = Path(file.name).stem

                rx_map = self.compute_rx_map(image)
                max_score = self.compute_max_score(rx_map)
                self.save_rx_heatmap(rx_map, name, fmt=save_fmt)
                results[file.name] = max_score
            except Exception as e:
                print(f"⚠️ Error in {file.name}: {e}")
        return results

    # self.save_rx_heatmap(rx_map, image_name)  # default = .npz
    # self.save_rx_heatmap(rx_map, image_name, fmt="png")  # optional .png for inspection

    def save_rx_heatmap(self, rx_map: np.ndarray, image_name: str, fmt: str = "npz"):
        """
        Save RX heatmap in the specified format.
        Supported: 'npz' (default), 'npy', 'png'
        """
        out_path = self.output_dir / f"{image_name}_rx.{fmt}"

        if fmt == "npz":
            np.savez_compressed(out_path, rx_map=rx_map)
        elif fmt == "npy":
            np.save(out_path, rx_map)
        elif fmt == "png":
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.imshow(rx_map, cmap='hot')
            plt.colorbar(label='RX Score')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'npz', 'npy', or 'png'.")

def main():
    input_dir = "path/to/cleaned_images"     # folder with cleaned SAR images or patches
    output_dir = "path/to/rx_heatmaps"        # where RX maps will be stored
    save_format = "npz"                       # options: "npz", "npy", "png"
    top_n = 5

    rx = RXDetector(input_dir, output_dir)
    print(f"🚀 Starting RX batch anomaly detection on: {input_dir}")
    results = rx.batch_process(save_fmt=save_format)

    print("\n✅ RX processing completed.")
    print("📊 Summary of max RX scores:")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")
        # Export results to CSV
        df = pd.DataFrame(list(results.items()), columns=["Filename", "Max_RX_Score"])
        csv_path = Path(output_dir) / "rx_scores.csv"
        df.to_csv(csv_path, index=False)
        print(f"📁 RX scores saved to CSV: {csv_path}")

        # Visualize top N anomalies
        top_anomalies = df.sort_values(by="Max_RX_Score", ascending=False).head(top_n)
        print(f"\n👀 Top {top_n} anomalies based on RX Score:")

        for i, row in top_anomalies.iterrows():
            image_name = row["Filename"]
            print(f"{image_name}: {row['Max_RX_Score']:.4f}")

            # Show corresponding RX map if it's a .png or recreate from .npz
            rx_path = Path(output_dir) / f"{Path(image_name).stem}_rx.{save_format}"
            if rx_path.exists():
                if save_format == "png":
                    img = plt.imread(rx_path)
                elif save_format in ["npz", "npy"]:
                    if save_format == "npz":
                        img = np.load(rx_path)["rx_map"]
                    else:
                        img = np.load(rx_path)

                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap='hot')
                plt.title(f"RX Heatmap: {image_name}")
                plt.colorbar()
                plt.axis('off')
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()
