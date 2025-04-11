import pandas as pd
import numpy as np
from pathlib import Path
from sentinelhub import SHConfig
import ee
import geemap
import pyarrow


class DataCollector:
    def __init__(self, area_of_interest, data_source='GEE', gee_project='asterra-454018'):
        self.area_of_interest = area_of_interest
        self.data_source = data_source
        self.config = SHConfig()

        if self.data_source == 'GEE':
            print("🔑 Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize(project=gee_project)
            self.roi = ee.Geometry.Rectangle(area_of_interest)

    def collect_metrics(self):
        if self.data_source == 'GEE':
            sar_image = self.download_sentinel_data_gee()
            features = self.extract_features_gee(sar_image)
        else:
            images = self.download_sentinel_data(self.area_of_interest)
            features = self.extract_features(images)

        self.save_to_parquet(features)

    def download_sentinel_data(self, aoi):
        print(f"📡 Downloading SAR data from S3 for AOI: {aoi}")
        return []

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

    def extract_features_gee(self, sar_image):
        print("🧪 Extracting full SAR array from GEE...")

        sar_numpy = geemap.ee_to_numpy(sar_image, region=self.roi, bands=["VV", "VH"], scale=10)

        print("📐 Extracted SAR array shape:", sar_numpy.shape)
        print("📊 VV min/max:", np.nanmin(sar_numpy[..., 0]), np.nanmax(sar_numpy[..., 0]))
        print("📊 VH min/max:", np.nanmin(sar_numpy[..., 1]), np.nanmax(sar_numpy[..., 1]))

        nan_count = np.isnan(sar_numpy).sum()
        print(f"🧼 NaN values in SAR array: {nan_count}")

        # Optionally show a band
        import matplotlib.pyplot as plt
        plt.imshow(sar_numpy[..., 0], cmap='gray')
        plt.title("VV Band Snapshot")
        plt.colorbar()
        plt.show()

        # Save array as before
        output_array_path = Path("data/collected_sar_array.npy")
        output_array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_array_path, sar_numpy)
        print(f"💾 SAR array saved to {output_array_path}")

        return pd.DataFrame([{
            "image_shape": str(sar_numpy.shape),
            "saved_path": str(output_array_path)
        }])

    def extract_features(self, images):
        print("🧪 Extracting features from SAR images...")
        return pd.DataFrame([
            {"image_id": "placeholder", "mean_intensity": 0.0, "texture": 0.0, "entropy": 0.0}
            for _ in images
        ])

    def save_to_parquet(self, features):
        output_path = Path("data/collected_sar_features.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False)
        print(f"💾 Features saved to {output_path}")


def main():
    from sentinelhub import BBox, CRS

    # Rectangle: [West, South, East, North] (California example)
    small_roi = [-122.5, 37.5, -122.4, 37.6]
    california_bounds = small_roi #[-124.4, 32.5, -114.3, 42]
    collector = DataCollector(area_of_interest=california_bounds, data_source='GEE')
    collector.collect_metrics()


if __name__ == "__main__":
    main()
