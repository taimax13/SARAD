#!/usr/bin/env python3
"""
SAR Defect Detection: DINOv3 features + Gated MIL + group safe split + cache + calibration + TTA + XAI HTML

Highlights
1. Group safe scene split with stratification to avoid leakage
2. Sampling of suspected defect scenes and at least five thousand normal scenes from S3
3. DINOv3 feature caching on disk for fast iteration
4. Lightweight Gated Attention MIL head with EMA, warmup cosine, class weighting
5. Optional isotonic calibration or temperature scaling
6. Test time augmentation on inference
7. Rich HTML report with interactive Plotly figures and attention overlays

Notes
• Set CFG.S3_BUCKET, CFG.S3_PREFIXES, CFG.CSV_PATH for your environment
• To download facebook dinov3 you need a Hugging Face token in env HUGGINGFACE_HUB_TOKEN or pass --hf-token
• All artifacts are written under CFG.OUT_DIR/exports/<timestamp>
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from PIL import Image, ImageFilter, ImageFile, ImageOps

import boto3
from botocore.config import Config as BotoConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from huggingface_hub import login as hf_login
from transformers import AutoConfig, AutoModel

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger("sar_mil")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class Config:
    # IO
    OUT_DIR: str = "./runs/sar_dinov3_mil"
    CSV_PATH: str = "./under20pleaks.csv"  # CSV with column scene_id and either leak_suspected_count or label
    S3_BUCKET: str = "imagery-storage"
    S3_REGION: str = "us-west-2"
    S3_PREFIXES: List[str] = None
    FILE_EXTS: List[str] = None

    # Data
    MIN_NORMALS: int = 5200
    IMG_SIZE: int = 256
    DESPECKLE: int = 3  # median filter size, 1 disables
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Backbone
    DINOV3_ID: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    HIDDEN_LAYERS: List[int] = None
    L2_NORMALIZE_FEATS: bool = True

    # MIL head
    ATT_DIM: int = 256
    N_HEADS: int = 2
    ATT_DROPOUT: float = 0.10

    # Train
    BATCH_SIZE: int = 8
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 50
    PATIENCE: int = 10
    WARMUP_EPOCHS: int = 3
    ACCUM_STEPS: int = 1
    MAX_GRAD_NORM: float = 5.0

    # Operating point
    POLICY: str = "f1"  # f1 or recall_at or fpr_at
    POLICY_VALUE: float = 0.80

    # Calibration and TTA
    CALIBRATE: bool = True
    CALIB_KIND: str = "isotonic"  # isotonic or temperature
    TTA_TEST: bool = False
    TTA_KINDS: List[str] = None  # from {"orig","hflip","vflip"}

    # CI
    N_BOOT: int = 1000
    SEED: int = 42

    # HF token
    HF_TOKEN: str = ""

    def __post_init__(self) -> None:
        if self.S3_PREFIXES is None:
            self.S3_PREFIXES = ["ALOS/"]
        if self.FILE_EXTS is None:
            self.FILE_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        if self.HIDDEN_LAYERS is None:
            self.HIDDEN_LAYERS = [-1, -3]
        if self.TTA_KINDS is None:
            self.TTA_KINDS = ["orig", "hflip", "vflip"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# Default CFG with env token pickup
CFG = Config(
    HF_TOKEN=(
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
)
Path(CFG.OUT_DIR).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hfhome")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hfhome/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hfhome/transformers")
set_seed(CFG.SEED)

# ---------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------


def s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        region_name=CFG.S3_REGION,
        config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 10, "mode": "standard"}),
    )


def s3_list_images(bucket: str, prefixes: Sequence[str], exts: Sequence[str]) -> List[str]:
    cli = s3_client()
    keys: List[str] = []
    exts = tuple(e.lower() for e in exts)
    for prefix in prefixes:
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = cli.list_objects_v2(**kw)
            for it in resp.get("Contents", []):
                k = it["Key"]
                if k.lower().endswith(exts):
                    keys.append(k)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
    return keys


def s3_read_bytes(bucket: str, key: str) -> bytes:
    return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


# ---------------------------------------------------------------------
# Robust image IO
# ---------------------------------------------------------------------


def _arr_to_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        p1 = np.nanpercentile(arr, 1)
        p99 = np.nanpercentile(arr, 99)
        g = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
        rgb = np.stack([g, g, g], -1)
    else:
        arr = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., None], 3, axis=-1)
        p1 = np.nanpercentile(arr, 1, axis=(0, 1), keepdims=True)
        p99 = np.nanpercentile(arr, 99, axis=(0, 1), keepdims=True)
        rgb = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _read_tiff_safe(b: bytes) -> Optional[Image.Image]:
    try:
        bio = io.BytesIO(b)
        with tiff.TiffFile(bio) as tf:
            series = tf.series[0]
            if getattr(series, "levels", None) and len(series.levels) > 1:
                arr = series.levels[-1].asarray()
            else:
                arr = series.asarray()
        return _arr_to_rgb(arr)
    except Exception:
        try:
            bio = io.BytesIO(b)
            arr = tiff.imread(bio)
            return _arr_to_rgb(arr)
        except Exception:
            return None


def read_image_any(bucket: str, key: str) -> Optional[Image.Image]:
    b = s3_read_bytes(bucket, key)
    lower = key.lower()
    if lower.endswith((".tif", ".tiff")):
        img = _read_tiff_safe(b)
        if img is not None:
            return img
    bio = io.BytesIO(b)
    try:
        img = Image.open(bio)
        img = ImageOps.exif_transpose(img)
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")
        return img
    except Exception:
        return _read_tiff_safe(b)


# ---------------------------------------------------------------------
# Scene token
# ---------------------------------------------------------------------

ALOS_RGX = re.compile(r"(ALOS[0-9A-Z\-_]+(?:_L1\.\d)?)")


def canon_token(s: str) -> str:
    s = str(s)
    m = ALOS_RGX.search(s)
    if m:
        return m.group(1)
    b = os.path.basename(s)
    b = os.path.splitext(b)[0]
    parts = re.split(r"[_\-\.]", b)
    return "_".join(parts[:2]) if len(parts) >= 2 else b[:24]


# ---------------------------------------------------------------------
# Dataset builder and split
# ---------------------------------------------------------------------


def build_dataset_from_csv() -> Tuple[List[str], List[int]]:
    """
    Returns
    dataset_keys: list of S3 keys for positives and sampled normals
    labels: list of int labels aligned to dataset_keys
    """
    df = pd.read_csv(CFG.CSV_PATH)
    assert "scene_id" in df.columns, "CSV must include column scene_id"

    label_map: Dict[str, int] = {}
    if "leak_suspected_count" in df.columns:
        for _, r in df.iterrows():
            label_map[str(r["scene_id"])] = 1 if int(r["leak_suspected_count"]) > 0 else 0
    elif "label" in df.columns:
        for _, r in df.iterrows():
            label_map[str(r["scene_id"])] = int(r["label"])
    else:
        for _, r in df.iterrows():
            label_map[str(r["scene_id"])] = 1

    all_keys = s3_list_images(CFG.S3_BUCKET, CFG.S3_PREFIXES, CFG.FILE_EXTS)
    assert len(all_keys) > 0, "No images found in the specified bucket and prefixes"

    leak_ids = set(label_map.keys())
    leak_keys: List[str] = []
    found: set = set()
    for k in all_keys:
        tok = canon_token(k)
        for sid in leak_ids:
            if canon_token(sid) == tok or sid in k:
                leak_keys.append(k)
                found.add(sid)
                break

    if len(found) == 0:
        LOGGER.warning("No positive scenes found from CSV")

    normals = [k for k in all_keys if k not in leak_keys]
    if len(normals) < CFG.MIN_NORMALS:
        LOGGER.warning("Only %d normals available while %d requested. Using all normals.", len(normals), CFG.MIN_NORMALS)
        pick_normals = normals
    else:
        rng = random.Random(CFG.SEED)
        rng.shuffle(normals)
        pick_normals = normals[: CFG.MIN_NORMALS]

    dataset = leak_keys + pick_normals

    labels: List[int] = []
    for k in dataset:
        y = 0
        t = canon_token(k)
        for sid, v in label_map.items():
            if canon_token(sid) == t or sid in k:
                y = int(v)
                break
        labels.append(y)

    LOGGER.info("Dataset built. Positives %d   Normals %d   Total %d", sum(labels), len(dataset) - sum(labels), len(dataset))
    return dataset, labels


def group_stratified_split(
    keys: Sequence[str],
    labels: Sequence[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Group safe split by scene token with stratification at scene level.
    Ensures both classes in each split and no scene leakage across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    groups = [canon_token(k) for k in keys]

    scene_to_y: Dict[str, int] = {}
    for k, y, g in zip(keys, labels, groups):
        scene_to_y[g] = max(scene_to_y.get(g, 0), int(y))
    scene_tokens = list(scene_to_y.keys())
    scene_labels = [scene_to_y[t] for t in scene_tokens]

    for attempt in range(200):
        rs = seed + attempt
        trval_tokens, te_tokens = train_test_split(scene_tokens, test_size=test_ratio, stratify=scene_labels, random_state=rs)
        trval_labels = [scene_to_y[t] for t in trval_tokens]
        val_size = val_ratio / (train_ratio + val_ratio)
        tr_tokens, va_tokens = train_test_split(trval_tokens, test_size=val_size, stratify=trval_labels, random_state=rs)

        def slice_tokens(tokens: Iterable[str]) -> Tuple[List[str], List[int]]:
            token_set = set(tokens)
            ks = [k for k, g in zip(keys, groups) if g in token_set]
            ys = [labels[i] for i, k in enumerate(keys) if canon_token(k) in token_set]
            return ks, ys

        tr_keys, tr_labels = slice_tokens(tr_tokens)
        va_keys, va_labels = slice_tokens(va_tokens)
        te_keys, te_labels = slice_tokens(te_tokens)

        def pos(ys) -> int: return int(np.sum(np.array(ys) == 1))
        def neg(ys) -> int: return int(np.sum(np.array(ys) == 0))

        ok = (
            pos(tr_labels) > 0 and pos(va_labels) > 0 and pos(te_labels) > 0
            and neg(tr_labels) > 0 and neg(va_labels) > 0 and neg(te_labels) > 0
        )
        if not ok:
            continue

        Gtr = set(canon_token(k) for k in tr_keys)
        Gva = set(canon_token(k) for k in va_keys)
        Gte = set(canon_token(k) for k in te_keys)
        assert len(Gtr & Gva) == 0 and len(Gtr & Gte) == 0 and len(Gva & Gte) == 0, "Group leakage detected"

        return (tr_keys, tr_labels), (va_keys, va_labels), (te_keys, te_labels)

    raise RuntimeError("Could not produce stratified splits with both classes in each split")


# ---------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------


def preprocess_pil(img: Image.Image, train: bool, IMG_SIZE: int, DESPECKLE: int) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if DESPECKLE > 1:
        img = img.filter(ImageFilter.MedianFilter(size=DESPECKLE))
    if train:
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.5:
            img = ImageOps.flip(img)
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=False)
        scale = random.uniform(0.9, 1.1)
        w, h = img.size
        cw, ch = int(w * scale), int(h * scale)
        if cw < w or ch < h:
            x0 = random.randint(0, max(0, w - cw))
            y0 = random.randint(0, max(0, h - ch))
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.asarray(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x


# ---------------------------------------------------------------------
# HF token and DINOv3
# ---------------------------------------------------------------------


def ensure_hf_token(explicit: Optional[str] = None) -> str:
    tok = explicit or CFG.HF_TOKEN
    if not tok:
        for path in ["./hf_token.txt", os.path.expanduser("~/.hf_token")]:
            if Path(path).exists():
                try:
                    tok = Path(path).read_text(encoding="utf-8").strip()
                    break
                except Exception:
                    pass
    if not tok:
        raise RuntimeError("Hugging Face token is required. Set env HUGGINGFACE_HUB_TOKEN or provide --hf-token")
    try:
        hf_login(tok, add_to_git_credential=False, write_permission=False)
    except Exception as e:
        LOGGER.warning("Hugging Face login warning: %s", str(e))
    CFG.HF_TOKEN = tok
    return tok


def load_dinov3(model_id: str) -> Tuple[AutoModel, int, int, int]:
    ensure_hf_token(CFG.HF_TOKEN)
    cfg = AutoConfig.from_pretrained(model_id, token=CFG.HF_TOKEN)
    cfg.output_hidden_states = True
    model = AutoModel.from_pretrained(model_id, config=cfg, trust_remote_code=True, token=CFG.HF_TOKEN)
    model.eval().to(torch.device("cpu"))  # feature extraction on CPU to save GPU memory
    hidden = int(getattr(cfg, "hidden_size", 768))
    patch = int(getattr(cfg, "patch_size", 16))
    regs = int(getattr(cfg, "num_register_tokens", 0))
    return model, hidden, patch, regs


@torch.no_grad()
def extract_features(backbone, xb: torch.Tensor, skip_tokens: int, hidden_layers: Sequence[int]) -> torch.Tensor:
    out = backbone(pixel_values=xb, output_hidden_states=True)
    hs = out.hidden_states
    feats = [hs[i][:, skip_tokens:, :] for i in hidden_layers]
    cat = torch.cat(feats, dim=-1)
    if CFG.L2_NORMALIZE_FEATS:
        cat = F.normalize(cat, dim=-1)
    return cat


# ---------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------


def precompute_split_features(
    backbone,
    keys: Sequence[str],
    labels: Sequence[int],
    split: str,
    IMG_SIZE: int,
    PATCH: int,
    REGS: int,
) -> str:
    cache_dir = Path(CFG.OUT_DIR) / "feat_cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    t0 = time.time()
    for i, (k, y) in enumerate(zip(keys, labels), 1):
        out_pt = cache_dir / f"{i:06d}.pt"
        if out_pt.exists():
            rows.append({"path": str(out_pt), "key": k, "label": int(y)})
            continue
        img = read_image_any(CFG.S3_BUCKET, k)
        if img is None:
            LOGGER.warning("[skip] cannot read %s", k)
            continue
        x = preprocess_pil(img, train=False, IMG_SIZE=IMG_SIZE, DESPECKLE=CFG.DESPECKLE).unsqueeze(0)
        feats = extract_features(backbone, x, skip_tokens=1 + REGS, hidden_layers=CFG.HIDDEN_LAYERS)[0].to(torch.float16)
        torch.save({"feat": feats, "key": k, "label": int(y), "img_size": IMG_SIZE, "patch": PATCH}, out_pt)
        rows.append({"path": str(out_pt), "key": k, "label": int(y)})
        if i % 50 == 0:
            LOGGER.info("[cache][%s] %d/%d in %.1f min", split, i, len(keys), (time.time() - t0) / 60.0)
    pd.DataFrame(rows).to_csv(cache_dir / "index.csv", index=False)
    return str(cache_dir)


class FeatDataset(torch.utils.data.Dataset):
    def __init__(self, index_csv: str):
        df = pd.read_csv(index_csv)
        self.rows = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        d = torch.load(r["path"], map_location="cpu")
        feat = d["feat"].to(torch.float32)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        k = d["key"]
        return feat, y, k


def collate_feat(batch):
    if len(batch) == 0:
        return torch.empty(0, 1), torch.empty(0), []
    feats = [b[0] for b in batch]
    maxN = max(f.shape[0] for f in feats)
    feats_pad = []
    for f in feats:
        if f.shape[0] < maxN:
            pad = torch.zeros(maxN - f.shape[0], f.shape[1], dtype=f.dtype)
            feats_pad.append(torch.cat([f, pad], dim=0))
        else:
            feats_pad.append(f)
    x = torch.stack(feats_pad, 0)
    y = torch.stack([b[1] for b in batch], 0)
    keys = [b[2] for b in batch]
    return x.to(DEVICE), y.to(DEVICE), keys


# ---------------------------------------------------------------------
# Gated Attention MIL head
# ---------------------------------------------------------------------


class GatedAttnHead(nn.Module):
    def __init__(self, D: int, D_att: int = 256, n_heads: int = 2, dropout: float = 0.10):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(D, D_att), nn.Tanh())
        self.key = nn.Sequential(nn.Linear(D, D_att), nn.Sigmoid())
        self.scorer = nn.Linear(D_att, n_heads, bias=False)
        self.proj = nn.Linear(D, D)
        self.cls = nn.Linear(D, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.drop(feats)                                 # b n d
        a = self.gate(z) * self.key(z)                       # b n d_att
        scores = self.scorer(a).transpose(1, 2)              # b h n
        weights = torch.softmax(scores, dim=-1)              # b h n
        pooled = torch.einsum("bhn,bnd->bhd", weights, self.proj(z))
        agg = pooled.mean(dim=1)                             # b d
        logits = self.cls(agg).squeeze(-1)                   # b
        weights_mean = weights.mean(dim=1)                   # b n
        return logits, weights_mean


# ---------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------


def make_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)


def make_scheduler(optimizer, epochs: int, warmup: int):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_on_feats(head: nn.Module, loader, scaler, optimizer, loss_fn) -> float:
    head.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (xb, yb, _) in enumerate(loader):
        if xb.numel() == 0:
            continue
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, _ = head(xb)
            loss = loss_fn(logits, yb)
        scaler.scale(loss / CFG.ACCUM_STEPS).backward()
        if (step + 1) % CFG.ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), CFG.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate_on_feats(head: nn.Module, loader):
    head.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    probs: List[float] = []
    logits_all: List[float] = []
    labels: List[float] = []
    keys: List[str] = []
    total = 0.0
    n = 0
    for xb, yb, ks in loader:
        if xb.numel() == 0:
            continue
        logits, _ = head(xb)
        loss = loss_fn(logits, yb)
        p = torch.sigmoid(logits)
        total += loss.item() * yb.size(0)
        n += yb.size(0)
        probs.extend(p.detach().cpu().numpy().tolist())
        logits_all.extend(logits.detach().cpu().numpy().tolist())
        labels.extend(yb.cpu().numpy().tolist())
        keys.extend(ks)
    return total / max(1, n), np.array(probs), np.array(labels), keys, np.array(logits_all)


def compute_metrics(y_true, y_prob, thr=None, policy="f1", policy_value=0.8):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob)

    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    if thr is None:
        if policy == "f1":
            if thr_pr.size > 0 and prec.size > 1 and rec.size > 1:
                f1 = (2 * prec[1:] * rec[1:]) / (prec[1:] + rec[1:] + 1e-9)
                thr = float(thr_pr[np.nanargmax(f1)]) if f1.size else 0.5
            else:
                thr = 0.5
        elif policy == "recall_at":
            target = float(policy_value)
            ok = np.where(rec[1:] >= target)[0]
            if ok.size > 0:
                thr = float(thr_pr[ok[0]])
            else:
                thr = float(thr_pr[-1]) if thr_pr.size > 0 else 0.5
        elif policy == "fpr_at":
            max_fpr = float(policy_value)
            fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
            ok = np.where(fpr <= max_fpr)[0]
            if ok.size == 0:
                thr = float(thr_roc[np.argmin(fpr)])
            else:
                j = ok[np.argmax(tpr[ok])]
                thr = float(thr_roc[j])
        else:
            thr = 0.5

    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "anomaly"], digits=4, zero_division=0)
    acc = float((y_pred == y_true).mean())
    return dict(auroc=float(auroc), auprc=float(auprc), thr=float(thr), acc=float(acc), cm=cm, report=rep)


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aurocs: List[float] = []
    auprcs: List[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))

    def ci(a):
        if len(a) == 0:
            return (float("nan"), float("nan"), float("nan"))
        a = np.sort(a)
        lo = a[int(0.025 * len(a))]
        hi = a[int(0.975 * len(a)) - 1]
        return (float(np.mean(a)), float(lo), float(hi))

    mA, loA, hiA = ci(aurocs)
    mP, loP, hiP = ci(auprcs)
    return dict(auroc=(mA, loA, hiA), auprc=(mP, loP, hiP))


def expected_calibration_error(y_true, y_prob, bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(y_prob, edges) - 1
    ece = 0.0
    mce = 0.0
    for i in range(bins):
        m = idx == i
        if not m.any():
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        gap = abs(acc - conf)
        ece += gap * (m.sum() / len(y_true))
        mce = max(mce, gap)
    return ece, mce


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------


def save_curves(y_true, y_prob, out_png: Path):
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"AUROC {roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.plot(r, p, label=f"AUPRC {ap:.3f}")
    base = float(np.mean(y_true))
    plt.hlines(base, 0, 1, colors="gray", linestyles=":", label=f"Baseline {base:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall")
    plt.legend(loc="lower left")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def save_confusion(cm: np.ndarray, out_png: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["normal", "anomaly"])
    plt.yticks([0, 1], ["normal", "anomaly"])
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if val > cm.max() / 2 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def reliability_curve(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(y_prob, edges[:-1], right=False) - 1
    idx = np.clip(idx, 0, bins - 1)
    prob_bin = np.zeros(bins)
    acc_bin = np.zeros(bins)
    cnt = np.zeros(bins)
    for i in range(bins):
        m = idx == i
        if m.any():
            prob_bin[i] = y_prob[m].mean()
            acc_bin[i] = y_true[m].mean()
            cnt[i] = m.sum()
    return edges, prob_bin, acc_bin, cnt


def b64_img(pil: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_attention(img: Image.Image, weights_1d: np.ndarray, patch_size: int, IMG_SIZE: int) -> Image.Image:
    gh = IMG_SIZE // patch_size
    gw = IMG_SIZE // patch_size
    w = weights_1d.reshape(gh, gw)
    w = (w - w.min()) / (w.max() - w.min() + 1e-6)
    heat = Image.fromarray((w * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert("L")
    import matplotlib.cm as cm
    color = (np.array(cm.get_cmap("jet")(np.array(heat) / 255.0))[:, :, :3] * 255).astype(np.uint8)
    heat_rgb = Image.fromarray(color).resize(img.size, Image.BILINEAR)
    base = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return Image.blend(base, heat_rgb, alpha=0.45)


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------


def make_html_from_cached(
    head: nn.Module,
    cache_csv: str,
    patch: int,
    metrics: Dict,
    dataset_stats: Dict,
    IMG_SIZE: int,
    op_stats: Dict,
    out_html: Path,
):
    df = pd.read_csv(cache_csv)
    scores = []
    with torch.no_grad():
        for _, r in df.iterrows():
            d = torch.load(r["path"], map_location="cpu")
            feat = d["feat"].to(torch.float32).unsqueeze(0).to(DEVICE)
            logit, w = head(feat)
            p = torch.sigmoid(logit)[0].item()
            scores.append((r["key"], p, w[0].detach().cpu().numpy(), int(r["label"])))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    topA = scores_sorted[:12]
    topN = scores_sorted[-8:]

    thr = op_stats["thr"]
    preds = [(k, 1 if p >= thr else 0, y, p, w) for k, p, w, y in [(s[0], s[1], s[2], s[3]) for s in scores_sorted]]
    false_pos = [(k, p, w, y) for k, yp, y, p, w in preds if yp == 1 and y == 0][:12]
    false_neg = [(k, p, w, y) for k, yp, y, p, w in preds if yp == 0 and y == 1][:12]

    y_true_all = dataset_stats["y_true_test"].tolist()
    y_prob_all = dataset_stats["y_prob_test"].tolist()
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    auroc = roc_auc_score(y_true_all, y_prob_all) if len(np.unique(y_true_all)) > 1 else float("nan")
    pr, rc, _ = precision_recall_curve(y_true_all, y_prob_all)
    auprc = average_precision_score(y_true_all, y_prob_all)
    base = float(np.mean(y_true_all))

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUROC={auroc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="No skill"))
    fig_roc.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", template="plotly_white", width=520, height=380)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"PR AUPRC={auprc:.3f}"))
    fig_pr.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines", line=dict(dash="dot"), name=f"Baseline={base:.3f}"))
    fig_pr.update_layout(title="Precision Recall", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white", width=520, height=380)

    _, pbin, abin, _ = reliability_curve(y_true_all, y_prob_all, bins=10)
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(x=pbin, y=abin, mode="lines+markers", name="Empirical"))
    fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot"), name="Ideal"))
    fig_rel.update_layout(title="Reliability", xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                          template="plotly_white", width=520, height=380)

    y_true_np = np.asarray(y_true_all).astype(int)
    y_prob_np = np.asarray(y_prob_all).astype(float)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=y_prob_np[y_true_np == 0], name="normal", nbinsx=50, histnorm="density", opacity=0.75))
    if (y_true_np == 1).any():
        fig_hist.add_trace(go.Histogram(x=y_prob_np[y_true_np == 1], name="anomaly", nbinsx=50, histnorm="density", opacity=0.60))
    fig_hist.update_layout(barmode="overlay", title="Score distribution", xaxis_title="Predicted probability", yaxis_title="Density",
                           template="plotly_white", width=520, height=380)

    roc_html = fig_roc.to_html(full_html=False, include_plotlyjs="cdn")
    pr_html = fig_pr.to_html(full_html=False, include_plotlyjs=False)
    rel_html = fig_rel.to_html(full_html=False, include_plotlyjs=False)
    hist_html = fig_hist.to_html(full_html=False, include_plotlyjs=False)

    style = """
<style>
*{box-sizing:border-box}
body{font-family:Inter,Arial;color:#111;background:#f7fafc;margin:0;padding:24px}
h1,h2,h3{margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;background:white}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px}
.kpi .box{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px;text-align:center}
.kpi .val{font-size:20px;font-weight:600}
.kpi .lbl{font-size:12px;color:#555}
.pill{background:#ebf4ff;color:#2b6cb0;padding:3px 8px;border-radius:10px;font-size:12px}
.pill.ok{background:#ddf7e5;color:#276749}
.pill.bad{background:#fee2e2;color:#c53030}
img{width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb;margin-top:6px}
pre{white-space:pre-wrap;font-size:13px;background:#f1f5f9;padding:8px;border-radius:8px;border:1px solid #e2e8f0}
.container{display:flex;gap:16px;flex-wrap:wrap}
.small{font-size:12px;color:#4b5563}
hr{border:0;border-top:1px solid #e5e7eb;margin:16px 0}
footer{margin-top:16px;color:#4b5563;font-size:12px}
</style>
"""
    lines = []
    lines.append("<!doctype html><meta charset='utf-8'/>")
    lines.append(style)
    lines.append("<h1>SAR Defect Detection</h1>")
    lines.append("<div class='kpi'>")
    lines.append(f"<div class='box'><div class='val'>{metrics['test_auroc']:.3f}</div><div class='lbl'>Test AUROC</div></div>")
    lines.append(f"<div class='box'><div class='val'>{metrics['test_auprc']:.3f}</div><div class='lbl'>Test AUPRC</div></div>")
    lines.append(f"<div class='box'><div class='val'>{metrics['test_acc']:.3f}</div><div class='lbl'>Accuracy</div></div>")
    lines.append(f"<div class='box'><div class='val'>{metrics['test_thr']:.4f}</div><div class='lbl'>Threshold</div></div>")
    lines.append(f"<div class='box'><div class='val'>{metrics['brier']:.4f}</div><div class='lbl'>Brier</div></div>")
    lines.append(f"<div class='box'><div class='val'>{op_stats['precision']:.3f}</div><div class='lbl'>Precision</div></div>")
    lines.append(f"<div class='box'><div class='val'>{op_stats['recall']:.3f}</div><div class='lbl'>Recall</div></div>")
    lines.append(f"<div class='box'><div class='val'>{op_stats['fpr']:.3f}</div><div class='lbl'>FPR</div></div>")
    lines.append("</div>")

    lines.append("<h2>Dataset</h2>")
    lines.append(
        f"<pre>Train {dataset_stats['n_train']}   Val {dataset_stats['n_val']}   Test {dataset_stats['n_test']}\n"
        f"Anomalies: {dataset_stats['n_pos_train']}/{dataset_stats['n_pos_val']}/{dataset_stats['n_pos_test']}  (train val test)\n"
        f"Group leakage check: PASSED</pre>"
    )

    lines.append("<h2>Interactive operating point</h2>")
    y_true_json = json.dumps(y_true_all)
    y_prob_json = json.dumps(y_prob_all)
    lines.append(f"""
<div class='card'>
  <div class='small'>Move the slider to change threshold and see metrics</div>
  <input id="thr" type="range" min="0" max="1000" value="{int(metrics['test_thr']*1000)}" style="width:100%;" oninput="updateThr(this.value/1000)">
  <div id="thrval" class='small'>Threshold: {metrics['test_thr']:.4f}</div>
  <div id="opmetrics" class='small'></div>
</div>
<script>
const y_true = {y_true_json};
const y_prob = {y_prob_json};
function calc(thr) {{
  let tp=0, fp=0, tn=0, fn=0;
  for (let i=0;i<y_true.length;i++) {{
    const y=y_true[i]; const p=y_prob[i]>=thr?1:0;
    if (y===1 && p===1) tp++;
    else if (y===1 && p===0) fn++;
    else if (y===0 && p===1) fp++;
    else tn++;
  }}
  const prec = tp+fp>0 ? tp/(tp+fp) : 0.0;
  const rec  = tp+fn>0 ? tp/(tp+fn) : 0.0;
  const fpr  = fp+tn>0 ? fp/(fp+tn) : 0.0;
  return [prec, rec, fpr, tp, fp, tn, fn];
}}
function updateThr(t) {{
  const m = calc(t);
  document.getElementById("thrval").innerText = "Threshold: " + t.toFixed(4);
  document.getElementById("opmetrics").innerText =
    "Precision " + m[0].toFixed(4) + "   Recall " + m[1].toFixed(4) + "   FPR " + m[2].toFixed(4) +
    "   |   TP " + m[3] + "   FP " + m[4] + "   TN " + m[5] + "   FN " + m[6];
}}
updateThr({metrics['test_thr']});
</script>
""")

    lines.append("<h2>Curves</h2>")
    lines.append("<div class='container'>")
    lines.append(f"<div class='card'>{roc_html}</div>")
    lines.append(f"<div class='card'>{pr_html}</div>")
    lines.append(f"<div class='card'>{rel_html}</div>")
    lines.append(f"<div class='card'>{hist_html}</div>")
    lines.append("</div>")

    def section(title, items):
        lines.append(f"<h3>{title}</h3><div class='grid'>")
        for key, prob, weights_1d, yt in items:
            img = read_image_any(CFG.S3_BUCKET, key)
            if img is None:
                lines.append(f"<div class='card'><div class='row'><span class='pill'>skip</span><span class='pill'>{key}</span></div></div>")
                continue
            disp = img.copy().resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            att = overlay_attention(disp, weights_1d, patch, IMG_SIZE)
            truth_pill = "<span class='pill ok'>actual normal</span>" if yt == 0 else "<span class='pill bad'>actual anomaly</span>"
            lines.append(
                f"<div class='card'><div class='row'><span class='pill'>p={prob:.3f}</span>{truth_pill}</div>"
                f"<div class='row'><span class='pill'>{key}</span></div>"
                f"<img src='data:image/jpeg;base64,{b64_img(disp)}'>"
                f"<img src='data:image/jpeg;base64,{b64_img(att)}'></div>"
            )
        lines.append("</div>")

    section("Top Anomalies", topA)
    section("Most Normal", topN)
    section("Top False Positives", false_pos)
    section("Top False Negatives", false_neg)

    lines.append("<footer>Report generated by SAR MIL pipeline</footer>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------


def tta_extract_logits_and_probs(
    backbone, head: nn.Module, keys: Sequence[str], labels: Sequence[int], IMG_SIZE: int, PATCH: int, REGS: int
) -> Tuple[np.ndarray, np.ndarray]:
    probs: List[float] = []
    logits_mean: List[float] = []
    ytrue: List[int] = []
    for k, y in zip(keys, labels):
        img = read_image_any(CFG.S3_BUCKET, k)
        if img is None:
            continue
        ft_list: List[torch.Tensor] = []
        for kind in CFG.TTA_KINDS:
            z = img.copy()
            if kind == "hflip":
                z = ImageOps.mirror(z)
            elif kind == "vflip":
                z = ImageOps.flip(z)
            x = preprocess_pil(z, train=False, IMG_SIZE=IMG_SIZE, DESPECKLE=CFG.DESPECKLE).unsqueeze(0)
            ft = extract_features(backbone, x, skip_tokens=1 + REGS, hidden_layers=CFG.HIDDEN_LAYERS)[0].to(torch.float32)
            if CFG.L2_NORMALIZE_FEATS:
                ft = F.normalize(ft, dim=-1)
            ft_list.append(ft)
        with torch.no_grad():
            logits_aug = []
            for ft in ft_list:
                lg, _ = head(ft.unsqueeze(0).to(DEVICE))
                logits_aug.append(lg)
            mean_logit = torch.stack(logits_aug).mean()
            p = torch.sigmoid(mean_logit).item()
            logits_mean.append(mean_logit.item())
        probs.append(p)
        ytrue.append(y)
    return np.array(ytrue), np.array(probs), np.array(logits_mean)


# ---------------------------------------------------------------------
# Export and calibration
# ---------------------------------------------------------------------


def make_export_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    export_dir = Path(CFG.OUT_DIR) / "exports" / ts
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def save_yaml_summary(export_dir: Path, cfg: Config, metrics: Dict, ds_stats: Dict, op_stats: Dict, files: Dict[str, str]) -> Path:
    summary = {
        "run_info": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(DEVICE)},
        "config": asdict(cfg),
        "metrics": metrics,
        "dataset": {
            "n_train": int(ds_stats["n_train"]),
            "n_val": int(ds_stats["n_val"]),
            "n_test": int(ds_stats["n_test"]),
            "n_pos_train": int(ds_stats["n_pos_train"]),
            "n_pos_val": int(ds_stats["n_pos_val"]),
            "n_pos_test": int(ds_stats["n_pos_test"]),
        },
        "operating_point": {
            "threshold": float(op_stats["thr"]),
            "precision": float(op_stats["precision"]),
            "recall": float(op_stats["recall"]),
            "fpr": float(op_stats["fpr"]),
        },
        "files": files,
    }
    out_yaml = export_dir / "run_summary.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return out_yaml


class TemperatureScaler(nn.Module):
    """Single parameter temperature for binary logits"""
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.t.clamp_min(1e-3)


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit temperature on validation by minimizing BCE"""
    device = DEVICE
    ts = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(ts.parameters(), lr=0.1, max_iter=100)
    x = torch.tensor(logits_val, dtype=torch.float32, device=device).unsqueeze(-1)
    y = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(-1)

    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(ts(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(ts.t.detach().cpu().item())


def apply_calibration(
    kind: str,
    val_prob: np.ndarray,
    val_logits: np.ndarray,
    val_y: np.ndarray,
    test_prob: np.ndarray,
    test_logits: np.ndarray,
):
    """
    Returns calibrated probabilities for val and test plus info dict and a callable that transforms new probs or logits.
    """
    info = {}
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_prob, val_y)

        def transform_test(p: np.ndarray, logits_unused: Optional[np.ndarray] = None) -> np.ndarray:
            return iso.transform(p)

        return iso.transform(val_prob), iso.transform(test_prob), {"kind": "isotonic"}, transform_test

    if kind == "temperature":
        T = fit_temperature(val_logits, val_y)
        info = {"kind": "temperature", "T": float(T)}

        def transform_test(p_unused: Optional[np.ndarray], logits_arr: np.ndarray) -> np.ndarray:
            return torch.sigmoid(torch.tensor(logits_arr) / T).numpy()

        val_p = torch.sigmoid(torch.tensor(val_logits) / T).numpy()
        test_p = torch.sigmoid(torch.tensor(test_logits) / T).numpy()
        return val_p, test_p, info, transform_test

    def identity(x: np.ndarray, _: Optional[np.ndarray] = None) -> np.ndarray:
        return x

    return val_prob, test_prob, {"kind": "none"}, identity


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAR defect detection pipeline")
    parser.add_argument("--csv-path", type=str, default=CFG.CSV_PATH, help="CSV path with scene_id and labels")
    parser.add_argument("--s3-bucket", type=str, default=CFG.S3_BUCKET)
    parser.add_argument("--s3-region", type=str, default=CFG.S3_REGION)
    parser.add_argument("--s3-prefix", type=str, nargs="*", default=CFG.S3_PREFIXES)
    parser.add_argument("--out-dir", type=str, default=CFG.OUT_DIR)
    parser.add_argument("--hf-token", type=str, default=CFG.HF_TOKEN)
    parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
    parser.add_argument("--calib", type=str, default=CFG.CALIB_KIND, choices=["isotonic", "temperature", "none"])
    args = parser.parse_args()

    # Bind CLI to config
    global CFG
    CFG = replace(
        CFG,
        CSV_PATH=args.csv_path,
        S3_BUCKET=args.s3_bucket,
        S3_REGION=args.s3_region,
        S3_PREFIXES=args.s3_prefix,
        OUT_DIR=args.out_dir,
        HF_TOKEN=args.hf_token or CFG.HF_TOKEN,
        TTA_TEST=bool(args.tta),
        CALIB_KIND=args.calib,
        CALIBRATE=(args.calib != "none"),
    )
    Path(CFG.OUT_DIR).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building dataset")
    keys_all, labels_all = build_dataset_from_csv()

    (tr_keys, tr_labels), (va_keys, va_labels), (te_keys, te_labels) = group_stratified_split(
        keys_all, labels_all, CFG.TRAIN_RATIO, CFG.VAL_RATIO, CFG.TEST_RATIO, CFG.SEED
    )
    LOGGER.info("[data] train %d   val %d   test %d", len(tr_keys), len(va_keys), len(te_keys))
    LOGGER.info("[pos]  train %d   val %d   test %d", sum(tr_labels), sum(va_labels), sum(te_labels))

    backbone, HIDDEN, PATCH, NUM_REGS = load_dinov3(CFG.DINOV3_ID)
    IMG_SIZE = CFG.IMG_SIZE
    if IMG_SIZE % PATCH != 0:
        IMG_SIZE = max(PATCH, (IMG_SIZE // PATCH) * PATCH)
        LOGGER.info("Aligning IMG_SIZE to %d for patch %d", IMG_SIZE, PATCH)

    cache_tr = precompute_split_features(backbone, tr_keys, tr_labels, "train", IMG_SIZE, PATCH, NUM_REGS)
    cache_va = precompute_split_features(backbone, va_keys, va_labels, "val", IMG_SIZE, PATCH, NUM_REGS)
    cache_te = precompute_split_features(backbone, te_keys, te_labels, "test", IMG_SIZE, PATCH, NUM_REGS)

    final_dim = HIDDEN * len(CFG.HIDDEN_LAYERS)
    head = GatedAttnHead(D=final_dim, D_att=CFG.ATT_DIM, n_heads=CFG.N_HEADS, dropout=CFG.ATT_DROPOUT).to(DEVICE)
    ema_head = AveragedModel(head)

    tr_csv = Path(cache_tr) / "index.csv"
    va_csv = Path(cache_va) / "index.csv"
    te_csv = Path(cache_te) / "index.csv"

    df_tr = pd.read_csv(tr_csv)
    tr_labels_list = df_tr["label"].tolist()
    from collections import Counter
    ctr = Counter(tr_labels_list)
    w_pos = 0.5 / max(1, ctr.get(1, 1))
    w_neg = 0.5 / max(1, ctr.get(0, 1))
    weights = torch.tensor([w_pos if y == 1 else w_neg for y in tr_labels_list], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(tr_labels_list), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        FeatDataset(str(tr_csv)),
        batch_size=CFG.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_feat,
    )
    val_loader = torch.utils.data.DataLoader(
        FeatDataset(str(va_csv)),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_feat,
    )
    test_loader = torch.utils.data.DataLoader(
        FeatDataset(str(te_csv)),
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_feat,
    )

    opt = make_optimizer(head)
    sched = make_scheduler(opt, CFG.EPOCHS, CFG.WARMUP_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    pos_count = int(df_tr["label"].sum())
    neg_count = int(len(df_tr) - pos_count)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / max(1, pos_count)], device=DEVICE))

    best_auprc = -1.0
    best_epoch = -1
    no_imp = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        train_loss = train_epoch_on_feats(head, train_loader, scaler, opt, loss_fn)
        with torch.no_grad():
            ema_head.update_parameters(head)

        val_loss, val_prob_raw, val_y, _, val_logits_raw = evaluate_on_feats(ema_head, val_loader)
        val_stats_raw = compute_metrics(val_y, val_prob_raw, thr=None, policy=CFG.POLICY, policy_value=CFG.POLICY_VALUE)
        LOGGER.info(
            "Epoch %03d | Train %.4f | Val AUPRC %.4f | AUROC %.4f | ACC %.4f | LR %.6f",
            epoch, train_loss, val_stats_raw["auprc"], val_stats_raw["auroc"], val_stats_raw["acc"], sched.get_last_lr()[0],
        )
        sched.step()

        if val_stats_raw["auprc"] > best_auprc + 1e-6:
            best_auprc = val_stats_raw["auprc"]
            best_epoch = epoch
            no_imp = 0
            torch.save(ema_head.module.state_dict(), Path(CFG.OUT_DIR) / "mil_head_best.pt")
            np.savez(Path(CFG.OUT_DIR) / "val_raw_for_calib.npz", prob=val_prob_raw, logits=val_logits_raw, y=val_y)
        else:
            no_imp += 1
            if no_imp >= CFG.PATIENCE:
                LOGGER.info("[early] no improvement for %d epochs", CFG.PATIENCE)
                break

    if Path(CFG.OUT_DIR, "mil_head_best.pt").exists():
        head.load_state_dict(torch.load(Path(CFG.OUT_DIR, "mil_head_best.pt"), map_location=DEVICE))
        ema_head = AveragedModel(head)
        LOGGER.info("[model] loaded best epoch %d", best_epoch)

    val_loss, val_prob_raw, val_y, _, val_logits_raw = evaluate_on_feats(ema_head, val_loader)
    test_loss, test_prob_raw, test_y, test_keys, test_logits_raw = evaluate_on_feats(ema_head, test_loader)

    # Optional TTA on test before calibration so we can calibrate the final probabilities
    if CFG.TTA_TEST:
        yt_tta, p_tta, lg_tta = tta_extract_logits_and_probs(backbone, ema_head, te_keys, te_labels, IMG_SIZE, PATCH, NUM_REGS)
        test_y = yt_tta
        test_prob_raw = p_tta
        test_keys = te_keys
        test_logits_raw = lg_tta

    # Calibration
    if CFG.CALIBRATE:
        val_prob, test_prob, calib_info, transform_fn = apply_calibration(
            CFG.CALIB_KIND, val_prob_raw, val_logits_raw, val_y, test_prob_raw, test_logits_raw
        )
        if calib_info["kind"] == "temperature":
            with open(Path(CFG.OUT_DIR) / "temperature.json", "w") as f:
                json.dump({"T": calib_info.get("T", 1.0)}, f)
    else:
        val_prob, test_prob = val_prob_raw, test_prob_raw
        calib_info = {"kind": "none"}

        def transform_fn(p: np.ndarray, _: Optional[np.ndarray] = None) -> np.ndarray:
            return p

    # Threshold from calibrated val
    val_stats = compute_metrics(val_y, val_prob, thr=None, policy=CFG.POLICY, policy_value=CFG.POLICY_VALUE)
    chosen_thr = val_stats["thr"]

    # Test metrics at chosen threshold
    test_stats = compute_metrics(test_y, test_prob, thr=chosen_thr, policy=CFG.POLICY, policy_value=CFG.POLICY_VALUE)
    y_pred = (test_prob >= test_stats["thr"]).astype(int)

    tp = int(((y_pred == 1) & (np.array(test_y) == 1)).sum())
    fp = int(((y_pred == 1) & (np.array(test_y) == 0)).sum())
    tn = int(((y_pred == 0) & (np.array(test_y) == 0)).sum())
    fn = int(((y_pred == 0) & (np.array(test_y) == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)

    out_dir = Path(CFG.OUT_DIR)
    pd.DataFrame({"key": test_keys, "y_true": test_y, "y_prob": test_prob}).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "classification_report.txt").write_text(test_stats["report"], encoding="utf-8")

    export_dir = make_export_dir()
    save_curves(test_y, test_prob, export_dir / "roc_pr_curves.png")
    save_confusion(test_stats["cm"], export_dir / "confusion_matrix.png")

    ci = bootstrap_ci(test_y, test_prob, n_boot=CFG.N_BOOT, seed=CFG.SEED)
    auroc_mean, auroc_lo, auroc_hi = ci["auroc"]
    auprc_mean, auprc_lo, auprc_hi = ci["auprc"]
    brier = brier_score_loss(test_y, test_prob)
    ece, mce = expected_calibration_error(test_y, test_prob, bins=15)

    metrics_store = {
        "best_epoch": int(best_epoch),
        "val_auprc": float(val_stats["auprc"]),
        "val_auroc": float(val_stats["auroc"]),
        "val_acc": float(val_stats["acc"]),
        "val_thr": float(val_stats["thr"]),
        "test_auprc": float(test_stats["auprc"]),
        "test_auroc": float(test_stats["auroc"]),
        "test_acc": float(test_stats["acc"]),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_thr": float(test_stats["thr"]),
        "test_brier": float(brier),
        "test_ece": float(ece),
        "test_mce": float(mce),
        "confusion_matrix": test_stats["cm"].tolist(),
        "model_tag": CFG.DINOV3_ID,
        "img_size": int(IMG_SIZE),
        "batch_size": int(CFG.BATCH_SIZE),
        "hidden_layers": CFG.HIDDEN_LAYERS,
        "threshold_criterion": CFG.POLICY,
        "temperature_T": float(json.load(open(Path(CFG.OUT_DIR) / "temperature.json"))["T"]) if (Path(CFG.OUT_DIR) / "temperature.json").exists() else None,
        "split_sizes": {"train": len(tr_keys), "val": len(va_keys), "test": len(te_keys)},
        "split_pos": {"train": int(sum(tr_labels)), "val": int(sum(va_labels)), "test": int(sum(te_labels))},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_store, indent=2), encoding="utf-8")

    ds_stats = {
        "n_train": len(tr_keys),
        "n_val": len(va_keys),
        "n_test": len(te_keys),
        "n_pos_train": int(sum(tr_labels)),
        "n_pos_val": int(sum(va_labels)),
        "n_pos_test": int(sum(te_labels)),
        "y_true_test": test_y,
        "y_prob_test": test_prob,
    }
    html_metrics = {
        "test_auroc": test_stats["auroc"],
        "test_auprc": test_stats["auprc"],
        "test_acc": test_stats["acc"],
        "test_thr": test_stats["thr"],
        "auroc_lo": auroc_lo,
        "auroc_hi": auroc_hi,
        "auprc_lo": auprc_lo,
        "auprc_hi": auprc_hi,
        "test_report": test_stats["report"],
        "brier": float(brier),
    }
    op_stats = {"thr": test_stats["thr"], "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}

    make_html_from_cached(
        ema_head,
        str(Path(CFG.OUT_DIR) / "feat_cache" / "test" / "index.csv"),
        PATCH,
        html_metrics,
        ds_stats,
        IMG_SIZE,
        op_stats,
        export_dir / "xai_report.html",
    )

    files = {
        "html_report": str(export_dir / "xai_report.html"),
        "roc_pr_curves": str(export_dir / "roc_pr_curves.png"),
        "confusion_matrix": str(export_dir / "confusion_matrix.png"),
        "predictions_csv": str(out_dir / "test_predictions.csv"),
        "metrics_json": str(out_dir / "metrics.json"),
        "classification_report_txt": str(out_dir / "classification_report.txt"),
    }
    out_yaml = save_yaml_summary(export_dir, CFG, html_metrics, ds_stats, op_stats, files)

    LOGGER.info("HTML report   : %s", files["html_report"])
    LOGGER.info("YAML summary  : %s", out_yaml)
    LOGGER.info("All artifacts saved under %s", export_dir)


if __name__ == "__main__":
    main()