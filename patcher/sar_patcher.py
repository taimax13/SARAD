import numpy as np
import pandas as pd
from pathlib import Path

class SARPatcher:
    def __init__(self, input_path, output_dir="data/patches", patch_size=128, stride=128):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_array(self):
        if self.input_path.suffix == ".npy":
            print(f"📥 Loading .npy image from {self.input_path}")
            return np.load(self.input_path)
        elif self.input_path.suffix == ".parquet":
            print(f"📥 Loading from .parquet and reconstructing dummy array...")
            df = pd.read_parquet(self.input_path)
            dummy_shape = (256, 256, 2)
            return np.random.rand(*dummy_shape).astype(np.float32)  # Replace this with real data logic
        else:
            raise ValueError("Unsupported file format. Use .npy or .parquet")

    def extract_patches(self):
        image = self.load_array()
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        h, w, c = image.shape
        patches = []
        count = 0

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                if np.isnan(patch).any():
                    continue
                patch_path = self.output_dir / f"patch_{count:04d}.npy"
                np.save(patch_path, patch)
                count += 1

        print(f"🧩 Extracted {count} patches into {self.output_dir}")
        return count

def main():
    patcher = SARPatcher(
        input_path="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/data_collector/data/collected_sar_array.npy",
        output_dir="data/patches",
        patch_size=128,
        stride=128
    )
    patch_count = patcher.extract_patches()
    print(f"✅ Done. Total patches extracted: {patch_count}")

if __name__ == "__main__":
    main()
