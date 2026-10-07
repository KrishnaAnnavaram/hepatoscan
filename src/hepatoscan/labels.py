"""The label map and label-safe array operations.

LiTS masks use 0 = background, 1 = liver, 2 = tumor. hepatoscan keeps the
three classes. A tumor voxel is also part of the liver region ("liver" in the
metrics means label >= 1). Masks are always resized with nearest-neighbour
sampling, so a resize never makes a value that is not a label.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BACKGROUND, LIVER, TUMOR = 0, 1, 2
LABELS = {BACKGROUND: "background", LIVER: "liver", TUMOR: "tumor"}
NUM_CLASSES = 3


class LabelError(ValueError):
    """A mask has values that are not in the label map."""


def validate_mask(mask: np.ndarray) -> np.ndarray:
    """Return the mask as uint8 or raise ``LabelError``. Fractional or unknown values are errors."""
    arr = np.asarray(mask)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
            raise LabelError("mask has fractional or non-finite values; resize masks with nearest-neighbour only")
    values = np.unique(arr)
    bad = [v for v in values.tolist() if v not in LABELS]
    if bad:
        raise LabelError(f"mask has unknown labels {bad}; expected {sorted(LABELS)}")
    return arr.astype(np.uint8)


def one_hot(mask: np.ndarray) -> np.ndarray:
    """(..., ) labels to (C, ...) float32 one-hot channels."""
    m = validate_mask(mask)
    return np.stack([(m == c).astype(np.float32) for c in range(NUM_CLASSES)])


def liver_region(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) >= LIVER


def tumor_region(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) == TUMOR


def resize_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbour resize. The output has only labels that the input has."""
    m = validate_mask(mask)
    factors = [t / s for t, s in zip(shape, m.shape)]
    out = ndimage.zoom(m, factors, order=0, mode="nearest", grid_mode=True)
    return out.astype(np.uint8)
