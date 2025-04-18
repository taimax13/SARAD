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

        count = 0

        def save_patch(patch, i, count):
            # Ensure proper shape
            if patch.ndim == 2:
                patch = np.expand_dims(patch, axis=-1)
            if patch.ndim != 3:
                print(f"⚠️ Skipping patch {count} from image {i} with invalid shape: {patch.shape}")
                return 0
            patch_path = self.output_dir / f"{i}_patch_{count:04d}.npy"
            np.save(patch_path, patch)
            return 1

        # Batch of images
        if image.ndim == 4:  # (N, H, W, C)
            for i, single_image in enumerate(image):
                h, w, c = single_image.shape
                if h < self.patch_size or w < self.patch_size:
                    print(f"⚠️ Skipping image {i}, too small: {single_image.shape}")
                    continue

                extracted = False
                for y in range(0, h - self.patch_size + 1, self.stride):
                    for x in range(0, w - self.patch_size + 1, self.stride):
                        patch = single_image[y:y + self.patch_size, x:x + self.patch_size]
                        count += save_patch(patch, i, count)
                        extracted = True

                # If image is exactly patch size → still save 1 patch
                if not extracted and (h, w) == (self.patch_size, self.patch_size):
                    count += save_patch(single_image, i, count)

        # Single image
        elif image.ndim == 3:  # (H, W, C)
            h, w, c = image.shape
            if h < self.patch_size or w < self.patch_size:
                print(f"⚠️ Image too small to patch: {image.shape}")
            else:
                for y in range(0, h - self.patch_size + 1, self.stride):
                    for x in range(0, w - self.patch_size + 1, self.stride):
                        patch = image[y:y + self.patch_size, x:x + self.patch_size]
                        count += save_patch(patch, 0, count)

                if (h, w) == (self.patch_size, self.patch_size) and count == 0:
                    count += save_patch(image, 0, count)

        else:
            raise ValueError(f"❌ Unexpected image shape: {image.shape}")

        print(f"🧩 Extracted {count} patches into {self.output_dir}")
        return count

    def _extract_from_single_image(self, image, img_index):
        h, w, c = image.shape
        count = 0
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                if np.isnan(patch).any():
                    continue
                patch_path = self.output_dir / f"{img_index}_patch_{count:04d}.npy"
                # Ensure patch shape = (H, W, 1)
                if patch.ndim == 2:
                    patch = np.expand_dims(patch, axis=-1)
                elif patch.shape[-1] != 1:
                    raise ValueError(f"❌ Invalid patch shape: {patch.shape}")

                # Save the patch
                np.save(patch_path, patch)

                np.save(patch_path, patch)
                count += 1
        print(f"🧩 Extracted {count} patches from image {img_index}")


def main():
    patcher = SARPatcher(
        input_path="/Users/talexm/PyProcessing/AnomalyDetector /SARAD/data_collector/data/collected_sar_array.npy",
        output_dir="data/patches/test",
        patch_size=128,
        stride=128
    )
    patch_count = patcher.extract_patches()
    print(f"✅ Done. Total patches extracted: {patch_count}")

if __name__ == "__main__":
    main()
