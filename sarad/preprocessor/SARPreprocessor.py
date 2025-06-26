import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List


class SARPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, patch_size: Tuple[int, int] = (256, 256)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, filename: str) -> np.ndarray:
        path = self.input_dir / filename
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {path}")
        return image

    def remove_stripes_fft(self, image: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        fshift[crow - 5:crow + 5, ccol - 60:ccol + 60] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        return img_back.astype(np.uint8)

    def split_into_patches(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        h, w = image.shape
        ph, pw = self.patch_size
        patches = []
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patch = image[i:i+ph, j:j+pw]
                if patch.shape == (ph, pw):
                    patches.append(((i, j), patch))
        return patches

    def save_patches(self, patches: List[Tuple[Tuple[int, int], np.ndarray]], base_name: str):
        patch_dir = self.output_dir / base_name
        patch_dir.mkdir(parents=True, exist_ok=True)
        for (i, j), patch in patches:
            patch_path = patch_dir / f"patch_{i}_{j}.npy"
            np.save(patch_path, patch)

    def process_and_save(self, filename: str):
        print(f"Processing {filename}...")
        image = self.load_image(filename)
        cleaned = self.remove_stripes_fft(image)
        patches = self.split_into_patches(cleaned)
        base_name = Path(filename).stem
        self.save_patches(patches, base_name)
        print(f"Saved {len(patches)} patches for {filename}.")

    def process_all(self):
        sar_files = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        ])
        if not sar_files:
            print("No SAR images found.")
            return
        for file in sar_files:
            try:
                self.process_and_save(file)
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")
