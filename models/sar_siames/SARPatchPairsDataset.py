import os
import random
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path


class SARPatchPairsDataset:
    """
    TensorFlow Dataset for loading paired SAR image patches for contrastive learning.
    
    This dataset creates pairs of SAR image patches (stored as .npy files) where:
    - Positive pairs (label=0) consist of two normal patches from the same original image.
    - Negative pairs (label=1) consist of one normal patch and one anomaly patch, 
      preferably from the same original image if available.
      
    File naming convention:
    - Normal patch: <imgidx>_patch_<patchidx>.npy
    - Anomaly patch: <imgidx>_patch_<patchidx>_A.npy
    
    Args:
        dataset_folder (str): Path to the folder containing .npy patch files.
        transform (callable, optional): Optional transform function to apply to each patch.
        num_pairs (int): Number of patch pairs to generate.
        batch_size (int): Batch size for the dataset.
        
    Returns:
        A tf.data.Dataset that yields tuples containing:
            patch_1 (Tensor): First patch in the pair (H, W, C).
            patch_2 (Tensor): Second patch in the pair (H, W, C).
            label (float Tensor): 0 for positive pair, 1 for negative pair.
            p1_name (string): Filename of the first patch.
            p2_name (string): Filename of the second patch.
    """
    
    def __init__(self, dataset_folder, transform=None, num_pairs=100, batch_size=32, pair_weight=0.5):
        self.dataset_folder = Path(dataset_folder)
        self.transform = transform
        self.pairs = []
        self.labels = []
        self.batch_size = batch_size
        
        # Gather normal and anomaly patches
        self.normal_patches = [f for f in os.listdir(dataset_folder) if f.endswith('.npy') and '_A.npy' not in f]
        self.anomaly_patches = [f for f in os.listdir(dataset_folder) if f.endswith('_A.npy')]
        
        print(f"📦 Found {len(self.normal_patches)} normal patches and {len(self.anomaly_patches)} anomaly patches")
        
        # Group normal patches by image index
        self.normal_by_imgidx = {}
        for f in self.normal_patches:
            img_idx = f.split('_')[0]
            self.normal_by_imgidx.setdefault(img_idx, []).append(f)
        
        # Build pairs
        self._build_pairs(num_pairs, pair_weight)
        
        # Create TensorFlow dataset
        self.dataset = self._create_tf_dataset()
    
    def _build_pairs(self, num_pairs, pair_weight=0.5):
        """Build the pairs of patches for contrastive learning."""
        print(f"🔄 Building {num_pairs} patch pairs for contrastive learning...")
        
        for _ in range(num_pairs):
            if random.random() < pair_weight:  # Positive pair (normal-normal, same image)
                img_idx = random.choice(list(self.normal_by_imgidx.keys()))
                patches = self.normal_by_imgidx[img_idx]
                if len(patches) < 2:
                    continue  # skip if not enough to sample
                p1, p2 = random.sample(patches, 2)
                label = 0
            else:  # Negative pair (normal-anomaly)
                p1 = random.choice(self.normal_patches)
                img_idx = p1.split('_')[0]
                matching_anomalies = [f for f in self.anomaly_patches if f.startswith(img_idx + '_')]
                p2 = random.choice(matching_anomalies if matching_anomalies else self.anomaly_patches)
                label = 1
                
            self.pairs.append((p1, p2))
            self.labels.append(label)
            
        print(f"✅ Created {len(self.pairs)} pairs: {self.labels.count(0)} positive, {self.labels.count(1)} negative")
    
    def _load_patch(self, patch_name):
        # Convert tf.Tensor to string
        if isinstance(patch_name, tf.Tensor):
            patch_name = patch_name.numpy().decode("utf-8")

        patch_path = str(self.dataset_folder / patch_name)
        patch = np.load(patch_path)
        return patch
    
    def _parse_pair(self, p1_name, p2_name, label):
        """Load and process a pair of patches."""
        patch1 = self._load_patch(p1_name)
        patch2 = self._load_patch(p2_name)
        
        label = np.float32(label)

        return patch1, patch2, label, p1_name, p2_name
    
    def _create_tf_dataset(self):
        """Create a TensorFlow Dataset from the pairs."""
        # Create datasets for each component
        p1_names = tf.data.Dataset.from_tensor_slices([p[0] for p in self.pairs])
        p2_names = tf.data.Dataset.from_tensor_slices([p[1] for p in self.pairs])
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        
        # Zip them together
        dataset = tf.data.Dataset.zip((p1_names, p2_names, labels))
        
        # Map the loading function to each element
        dataset = dataset.map(
            lambda p1, p2, label: tf.py_function(
                self._parse_pair,
                [p1, p2, label],
                [tf.float32, tf.float32, tf.float32, tf.string, tf.string]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set output shapes and types
        dataset = dataset.map(
            lambda p1, p2, label, p1_name, p2_name: (
                p1, p2, label, p1_name, p2_name
            )
        )
        
        return dataset
    
    def get_dataset(self, shuffle=True, cache=True):
        """Get the TensorFlow dataset ready for training."""
        ds = self.dataset
        
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.pairs))
        
        ds = ds.batch(self.batch_size)
        
        if cache:
            ds = ds.cache()
            
        return ds.prefetch(tf.data.AUTOTUNE)


# perform transform=normalize SAR patches? verify if preprocess patch does that 
def normalize_sar(img, min_db=-30, max_db=0):
    """Normalize SAR patches from dB scale to [0, 1]"""
    img = np.clip(img, min_db, max_db)
    return (img - min_db) / (max_db - min_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pair_weight", type=float, default=0.5)
    args = parser.parse_args()

    # Load train dataset
    dataset = SARPatchPairsDataset(
        dataset_folder=args.dataset_folder,
        transform=None,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        pair_weight=args.pair_weight
    )
    print(f"Total dataset size: {len(dataset.pairs)} pairs, ~{len(dataset.pairs) // args.batch_size} batches")

    full_dataset = dataset.get_dataset(shuffle=True, cache=False)
    num_batches = len(dataset.pairs) // args.batch_size
    val_dataset = full_dataset.take(num_batches // 5)  # 20% validation
    train_dataset = full_dataset.skip(num_batches // 5)

    # iterate over a batch
    for p1, p2, labels, p1_names, p2_names in train_dataset.take(1):
        print(f"Train batch - Patch 1 shape: {p1.shape}, Patch 2 shape: {p2.shape}")
        print(f"Labels: {labels.numpy()}")
