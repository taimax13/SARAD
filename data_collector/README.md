# 🌍 SAR Data Collector

This Python module enables the collection and processing of Sentinel-1 SAR (Synthetic Aperture Radar) images from multiple sources including **Google Earth Engine (GEE)**, **Amazon S3**, and **local directories**. It provides robust utilities for image normalization, resizing, and feature extraction into Parquet and NumPy formats.

---

## 📦 Features

- ✅ Download Sentinel-1 GRD data via Google Earth Engine (GEE)
- ✅ Fetch SAR images from AWS S3 or local directory
- ✅ Normalize and resize images to a consistent shape
- ✅ Extract basic statistical features (mean, std, min, max)
- ✅ Save data as `.npy` and `.parquet` files for downstream use

---

## 🛠 Installation

Make sure you have Python 3.8+ and install the required dependencies:

```bash
pip install -r requirements.txt
Also make sure the following are installed and configured:

earthengine-api (GEE authentication required)

gcloud CLI (if using GEE)

AWS credentials (~/.aws/credentials) if using S3

🚀 Usage
▶️ Run with Google Earth Engine
```bash

python data_collector.py
````
This will:

Authenticate with GEE
Download up to 50 Sentinel-1 images - you can play with number here 
Resize them to a consistent shape
Save the result as data/collected_sar_array.npy and metadata as .parquet

▶️ Run with Local Folder
Uncomment and use main2() in the script:

```python
def main2():
    local_data_dir = "/path/to/sar_images"

    collector = DataCollector(
        data_source='LOCAL',
        local_dir=local_data_dir
    )
    collector.collect_metrics()
 ```   
    
🧪 Output
After execution, you'll find:

data/collected_sar_array.npy – normalized stacked image array

data/collected_sar_features.parquet – basic statistics per image

📁 Folder Structure
```kotlin

.
├── data_collector.py
├── requirements.txt
├── data/
│   ├── collected_sar_array.npy
│   └── collected_sar_features.parquet
```
🔧 Configuration
You can change the input source by providing the data_source argument:

Source	Value	Additional Params
GEE	'GEE'	area_of_interest, gee_project
S3	'S3'	s3_bucket, s3_prefix
Local	'LOCAL'	local_dir

🌐 Requirements
geemap

earthengine-api

rasterio

boto3

pyarrow

scikit-image

numpy

pandas

🧠 Example Area of Interest
```
python
# Bounding Box [West, South, East, North]
small_roi = [-122.5, 37.5, -122.4, 37.6]
```

📜 License
MIT License. Use responsibly and give credit when due.

🤖 Author
Created by Talex Maxim for automated satellite data collection and processing.

