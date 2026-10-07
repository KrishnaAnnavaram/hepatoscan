"""Overlay of a label mask on one CT slice, as an RGB array or PNG bytes (PNG needs the ``image`` extra)."""
from __future__ import annotations

import io

import numpy as np

from .labels import LIVER, TUMOR
from .preprocess import LIVER_WINDOW, apply_window

COLORS = {LIVER: (60, 180, 75), TUMOR: (230, 25, 75)}


def best_slice(mask: np.ndarray) -> int:
    """The slice with the most tumor voxels, else the most liver voxels, else the middle slice."""
    for label in (TUMOR, LIVER):
        counts = (mask >= label).reshape(mask.shape[0], -1).sum(axis=1) if label == LIVER else \
            (mask == label).reshape(mask.shape[0], -1).sum(axis=1)
        if counts.max() > 0:
            return int(np.argmax(counts))
    return mask.shape[0] // 2


def overlay_rgb(hu: np.ndarray, mask: np.ndarray, index: int | None = None, alpha: float = 0.45) -> np.ndarray:
    k = best_slice(mask) if index is None else index
    gray = (apply_window(hu[k], LIVER_WINDOW) * 255).astype(np.float32)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for label, color in COLORS.items():
        sel = mask[k] == label
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * np.array(color, dtype=np.float32)
    return rgb.clip(0, 255).astype(np.uint8)


def to_png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
