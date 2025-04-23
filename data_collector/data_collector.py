from collections import Counter

import pandas as pd
import numpy as np
from pathlib import Path
from sentinelhub import SHConfig
import ee
import geemap
import pyarrow
import boto3
import rasterio
from skimage.transform import resize
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class DataCollector:
    def __init__(self, area_of_interest=None, data_source='GEE', gee_project='asterra-454018', s3_bucket=None,
                 s3_prefix=None, local_dir=None):
        self.area_of_interest = area_of_interest
        self.data_source = data_source
        self.config = SHConfig()
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.local_dir = Path(local_dir) if local_dir else None

        if self.data_source == 'GEE':
            print("🔑 Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize(project=gee_project)
            self.roi = ee.Geometry.Rectangle(area_of_interest)

    def collect_metrics(self):
        if self.data_source == 'GEE':
            sar_image = self.download_sentinel_data_gee()
            features = self.extract_features_gee(sar_image)
        elif self.data_source == 'S3':
            images = self.download_sentinel_data_s3()
            features = self.extract_features(images)
        elif self.data_source == 'LOCAL':
            images = self.load_images_from_local()
            features = self.extract_features(images)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

        self.save_to_parquet(features)

    def download_sentinel_data_gee(self):
        print("🌍 Fetching Sentinel-1 GRD data from GEE...")
        images = self.get_image_list_from_gee(max_images=50)

        print(f"📸 Number of images to process: {len(images)}")
        arrays, shapes = [], []

        for i, image in enumerate(images):
            try:
                print(f"📥 Processing image {i + 1}/{len(images)}...")
                arr = self.fetch_numpy_array(image)
                if arr is None or np.isnan(arr).all():
                    print(f"⚠️ Image {i} returned empty or NaNs. Skipping.")
                    continue
                shapes.append(arr.shape)
                arrays.append((i, arr))
            except Exception as e:
                print(f"❌ Failed image {i}: {e}")

        if not arrays:
            raise RuntimeError("❌ No valid SAR images collected.")

        target_shape = self.get_most_common_shape(shapes)
        print(f"📏 Most common shape: {target_shape}")

        normalized = []
        for i, arr in arrays:
            if arr.shape != target_shape:
                print(f"🔄 Resizing image {i} from {arr.shape} to {target_shape}")
                arr = self.resize_to_target(arr, target_shape)
            normalized.append(arr)

        final_array = np.stack(normalized)
        print(f"📐 Final stacked shape: {final_array.shape}")

        output_path = Path("data/collected_sar_array.npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, final_array)
        print(f"💾 Saved SAR array to {output_path}")

        return pd.DataFrame([{
            "image_count": len(normalized),
            "array_shape": str(final_array.shape),
            "saved_path": str(output_path)
        }])

    def get_image_list_from_gee(self, max_images=50):
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(self.roi)
            .filterDate("2024-07-01", "2025-03-31")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )
        total = collection.size().getInfo()
        print(f"📸 Total available: {total}")
        images = collection.toList(min(total, max_images))
        return [ee.Image(images.get(i)).clip(self.roi) for i in range(min(total, max_images))]

    def fetch_numpy_array(self, image):
        return geemap.ee_to_numpy(image, region=self.roi, bands=["VV", "VH"], scale=10)

    def get_most_common_shape(self, shapes):
        return Counter(shapes).most_common(1)[0][0]

    def resize_to_target(self, arr, target_shape):
        return resize(arr, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    # --- Sub-methods ---

    def _process_image_collection(self, dataset, max_images=50):
        images = dataset.toList(max_images)
        raw_arrays = []
        shape_counter = Counter()

        for i in range(max_images):
            image = ee.Image(images.get(i)).clip(self.roi)
            try:
                arr = self._download_image_array(image, i, max_images)
                if arr is None:
                    continue

                arr = self._normalize_image_shape(arr)
                shape_counter[arr.shape] += 1
                raw_arrays.append((i, arr))

            except Exception as e:
                print(f"❌ Failed to process image {i}: {e}")

        if not raw_arrays:
            raise RuntimeError("❌ No valid images collected from GEE.")

        # ✅ Determine most common shape
        most_common_shape = shape_counter.most_common(1)[0][0]
        print(f"📏 Most common shape: {most_common_shape}")

        # ✅ Normalize all arrays to most common shape
        collected_arrays = []
        for i, arr in raw_arrays:
            if arr.shape != most_common_shape:
                print(f"🔄 Resizing image {i} from {arr.shape} to {most_common_shape}")
                arr = resize(arr, most_common_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
            collected_arrays.append(arr)

        return self._save_and_return_stack(collected_arrays)

    def _download_image_array(self, image, index, total):
        print(f"📥 Processing image {index + 1}/{total}...")
        arr = geemap.ee_to_numpy(image, region=self.roi, bands=["VV", "VH"], scale=10)

        if arr is None or np.isnan(arr).all():
            print(f"⚠️ Image {index} returned empty or all-NaN array. Skipping.")
            return None

        return arr

    def _normalize_image_shape(self, arr):
        if arr.ndim == 2:
            return np.expand_dims(arr, -1)
        return arr

    def _ensure_shape_consistency(self, arr, target_shape, index):
        if arr.shape != target_shape:
            print(f"🔄 Resizing image {index} from {arr.shape} to {target_shape}")
            arr = resize(arr, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
        return arr

    def _save_and_return_stack(self, collected_arrays):
        final_array = np.stack(collected_arrays)
        print(f"📐 Final stacked SAR array shape: {final_array.shape}")

        output_array_path = Path("data/collected_sar_array.npy")
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_array_path, final_array)
        print(f"💾 SAR array saved to {output_array_path}")

        return pd.DataFrame([{
            "image_count": len(collected_arrays),
            "array_shape": str(final_array.shape),
            "saved_path": str(output_array_path)
        }])

    def download_sentinel_data_s3(self):
        print(f"📡 Downloading SAR data from S3 bucket: {self.s3_bucket}/{self.s3_prefix}")
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.s3_prefix)
        print(response)

        image_paths = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.tif'):
                local_path = Path("data/s3_downloads") / Path(key).name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(self.s3_bucket, key, str(local_path))
                image_paths.append(local_path)

        print(f"✅ Downloaded {len(image_paths)} images from S3")
        return image_paths

    def load_images_from_local(self):
        print(f"📂 Loading images from local directory: {self.local_dir}")
        arr_img = list(self.local_dir.glob("*.*"))
        print(f"Extracted:{len(arr_img)}")
        return arr_img

    def extract_features_gee(self, sar_image):
        print("🧪 Extracting full SAR array from GEE...")
        sar_numpy = geemap.ee_to_numpy(sar_image, region=self.roi, bands=["VV", "VH"], scale=10)

        print("📐 Extracted SAR array shape:", sar_numpy.shape)
        print("📊 VV min/max:", np.nanmin(sar_numpy[..., 0]), np.nanmax(sar_numpy[..., 0]))
        print("📊 VH min/max:", np.nanmin(sar_numpy[..., 1]), np.nanmax(sar_numpy[..., 1]))

        nan_count = np.isnan(sar_numpy).sum()
        print(f"🧼 NaN values in SAR array: {nan_count}")

        output_array_path = Path("data/collected_sar_array.npy")
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_array_path, sar_numpy)
        print(f"💾 SAR array saved to {output_array_path}")

        return pd.DataFrame([{
            "image_shape": str(sar_numpy.shape),
            "saved_path": str(output_array_path)
        }])

    def extract_features(self, image_paths):
        print("🧪 Extracting features from SAR images...")
        features = []
        for path in image_paths:
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
                stats = {
                    "image_id": path.name,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr))
                }
                features.append(stats)
        return pd.DataFrame(features)

    def save_to_parquet(self, features):
        output_path = Path("data/collected_sar_features.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False)
        print(f"💾 Features saved to {output_path}")


def main():
    from sentinelhub import BBox, CRS

    # Rectangle: [West, South, East, North] (California example)
    small_roi = [-122.5, 37.5, -122.4, 37.6]

    collector = DataCollector(
        area_of_interest=small_roi,
        data_source='GEE',
        gee_project='asterra-454018'
    )
    collector.collect_metrics()

def main2():
    # Local test case
    local_data_dir = "/Users/talexm/Desktop/val_data"

    collector = DataCollector(
        data_source='LOCAL',
        local_dir=local_data_dir
    )
    collector.collect_metrics()


if __name__ == "__main__":
    main()
