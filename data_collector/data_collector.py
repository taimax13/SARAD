import pandas as pd
import numpy as np
from pathlib import Path
from sentinelhub import SHConfig
import ee
import geemap
import pyarrow
import boto3
import rasterio
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
        dataset = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(self.roi)
            .filterDate("2025-01-01", "2025-03-31")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )
        count = dataset.size().getInfo()
        print(f"📸 Number of Sentinel-1 images found: {count}")

        sar_image = dataset.median().clip(self.roi)
        print("🛰️ Image metadata:", sar_image.getInfo())
        return sar_image

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
    main2()
