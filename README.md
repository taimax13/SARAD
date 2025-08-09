from pathlib import Path

readme_content = """
# 🌌 SARAD: Smart Anomaly Detection for SAR Images

**SARAD** is a modular, intelligent pipeline designed to detect and filter anomalous satellite images — especially noisy, artifact-heavy **Synthetic Aperture Radar (SAR)** data — using a fusion of classic statistical methods and deep learning.

---

## 🧠 What It Does

SARAD runs through the following steps:

1. **Preprocessing**: Cleans noisy raw images (e.g., stripes, edge glitches) with FFT filters and splits them into patches.
2. **RX Detection**: Uses the Reed-Xiaoli (RX) algorithm to detect statistical outliers based on Mahalanobis distance.
3. **Autoencoder Detection**: Reconstructs patches using a deep convolutional autoencoder and measures reconstruction error.
4. **Fusion & Clustering**: Combines RX and AE scores, then clusters them using DBSCAN to determine anomaly groupings.
5. **Export**: Saves scores, heatmaps, clustering results, and visualizations.

---

## 🧩 Components

- `SARPreprocessor`: Cleans and splits raw SAR images
- `RXDetector`: Computes RX anomaly scores and heatmaps
- `AutoencoderAnomalyDetector`: Evaluates AE-based reconstruction error
- `SARAutoencoderTrainer`: Trains a convolutional autoencoder on clean SAR patches
- `JointAnomalyClassifier`: Combines RX + AE scores and clusters using DBSCAN
- `SARADPipelineManager`: Orchestrates the full pipeline end-to-end

---

## ⚡ Quickstart

### 1. Preprocess Raw SAR Images

```python
from sarad.pipeline import SARPreprocessor

pre = SARPreprocessor("data/raw", "data/cleaned")
pre.process_all()
```
[![Notebook Log Preview](images/log_preview.png)](https://www.kaggle.com/code/talexmaxim/ansamble-of-anomaly-ima-3bf0fe/log?scriptVersionId=255094833)
