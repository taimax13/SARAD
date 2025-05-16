# --- SOTA DINOv2 + Mahalanobis Anomaly Detection Script ---

import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt

# 1. GLOBAL SETTINGS
BATCH_SIZE = 8
PADIM_DIM = 96
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. LOAD PATCH FILES
raw = Path('data/raw')
files = sorted(raw.glob('*.png'))
assert len(files) >= 2, "Not enough .png patches found in data/raw"
split = int(0.7 * len(files))
train_files, test_files = files[:split], files[split:]
print(f"Train patches: {len(train_files)}, Test patches: {len(test_files)}")

# 3. PREPROCESSING
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
cfg = processor.size
if isinstance(cfg, dict):
    H = cfg.get("height", cfg.get("shortest_edge"))
    W = cfg.get("width",  cfg.get("shortest_edge"))
elif isinstance(cfg, (list, tuple)):
    H, W = cfg
else:
    H = W = cfg
mean, std = processor.image_mean, processor.image_std
transform = T.Compose([
    T.Resize((H, W)),
    T.ToTensor(),
    T.Normalize(mean, std)
])

# 4. DATASET
class PatchDataset(Dataset):
    def __init__(self, file_list): self.file_list = file_list
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        im = Image.open(self.file_list[idx]).convert("RGB")
        im_t = transform(im)
        pixel = processor(images=T.ToPILImage()(im_t), return_tensors="pt").pixel_values[0]
        return pixel, str(self.file_list[idx])

train_dl = DataLoader(PatchDataset(train_files), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(PatchDataset(test_files),  batch_size=1, shuffle=False)

# 5. DINOv2 MODEL LOADING
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
print("Loaded DINOv2-base model.")

# 6. FEATURE EXTRACTION ON NORMAL PATCHES
features = []
with torch.no_grad():
    for pixels, _ in train_dl:
        f = model(pixel_values=pixels.to(device)).last_hidden_state[:,0,:].cpu().numpy()
        features.append(f)
features = np.vstack(features)
print(f"Extracted features from {features.shape[0]} train patches.")

# 7. FIT MAHALANOBIS (PaDiM) ON FEATURE SUBSET
sel = np.random.choice(features.shape[1], PADIM_DIM, replace=False)
mu = features[:,sel].mean(axis=0)
cov = np.cov(features[:,sel], rowvar=False) + np.eye(PADIM_DIM)*1e-6
icov = np.linalg.inv(cov)
def mahalanobis_score(f_vec):
    diff = f_vec[sel] - mu
    return float(diff @ icov @ diff)

# 8. SCORE ALL TEST PATCHES
patch_scores, file_names = [], []
with torch.no_grad():
    for pixels, fname in test_dl:
        feat = model(pixel_values=pixels.to(device)).last_hidden_state[:,0,:].cpu().numpy()[0]
        score = mahalanobis_score(feat)
        patch_scores.append(score)
        file_names.append(fname[0])
patch_scores = np.array(patch_scores)

# 9. VISUALIZATION: HISTOGRAM AND TOP ANOMALIES
plt.figure(figsize=(10,5))
plt.hist(patch_scores, bins=40, alpha=0.7, color='navy')
plt.xlabel("Mahalanobis Anomaly Score")
plt.ylabel("Number of Patches")
plt.title("Distribution of Test Patch Anomaly Scores")
plt.show()

topk = np.argsort(patch_scores)[-12:][::-1]
fig, axs = plt.subplots(3, 4, figsize=(12,8))
for ax, idx in zip(axs.flat, topk):
    im = Image.open(file_names[idx])
    ax.imshow(im)
    ax.set_title(f"Score={patch_scores[idx]:.2f}")
    ax.axis("off")
plt.suptitle("Top 12 Most Anomalous Test Patches")
plt.tight_layout()
plt.show()

print("✔️ End-to-end DINOv2 + Mahalanobis anomaly detection complete!")
