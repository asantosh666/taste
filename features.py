"""Per-frame CV features for Taste.

Real math only, no model calls here:
- sharpness: variance of the Laplacian (grayscale, analysis-size image).
- exposure: luminance histogram stats (mean, std, clipped-pixel fraction).
- dHash: 9x8 gradient hash, hand-rolled with Pillow + numpy (no imagehash dep).
- near-duplicate grouping: union-find over pairs with Hamming distance <= threshold.

Feature computation is deterministic and cached by file content hash (see main.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ANALYSIS_MAX_DIM = 1024
THUMB_MAX_DIM = 640
DUPE_HAMMING_THRESHOLD = 8


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_analysis_gray(path: Path) -> np.ndarray:
    """Load an image, downscale to ANALYSIS_MAX_DIM, return grayscale float array."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, ANALYSIS_MAX_DIM / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32).mean(axis=2)


def sharpness_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher = sharper. Standard focus measure."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def exposure_stats(gray: np.ndarray) -> dict:
    """Luminance histogram stats on a 0..255 scale."""
    g = np.clip(gray, 0, 255)
    mean = float(g.mean())
    std = float(g.std())
    clipped = float(((g <= 5) | (g >= 250)).mean())
    # Exposure quality: penalize clipping and extreme brightness/darkness.
    # 1.0 = no clipping, mean near mid-grey; degrades smoothly from there.
    mean_penalty = abs(mean - 127.5) / 127.5
    quality = max(0.0, (1.0 - clipped) * (1.0 - 0.5 * mean_penalty))
    return {"mean": mean, "std": std, "clipped_frac": clipped, "quality": quality}


def dhash(path: Path) -> int:
    """64-bit difference hash: 9x8 grayscale, bit set when pixel > left neighbor."""
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_near_dupes(hashes: dict[str, int], threshold: int = DUPE_HAMMING_THRESHOLD) -> list[list[str]]:
    """Union-find grouping; returns groups with 2+ members only."""
    parent = {k: k for k in hashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = list(hashes)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if hamming(hashes[keys[i]], hashes[keys[j]]) <= threshold:
                ri, rj = find(keys[i]), find(keys[j])
                if ri != rj:
                    parent[ri] = rj
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def extract_features(path: Path) -> dict:
    """Full per-frame feature dict. `path` may be any readable image file."""
    gray = load_analysis_gray(path)
    sharp = sharpness_laplacian_variance(gray)
    expo = exposure_stats(gray)
    return {
        "sharpness": sharp,
        "sharpness_log": float(np.log1p(sharp)),
        "exposure_mean": expo["mean"],
        "exposure_std": expo["std"],
        "exposure_clipped": expo["clipped_frac"],
        "exposure_quality": expo["quality"],
        "dhash": dhash(path),
    }


def make_thumb(src: Path, dest: Path) -> Path:
    """Write a JPEG thumbnail (max THUMB_MAX_DIM) for grid display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, THUMB_MAX_DIM / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im.save(dest, "JPEG", quality=85)
    return dest
