"""Anatomical clean-up of a predicted label volume."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .labels import BACKGROUND, LIVER, TUMOR


def largest_component(region: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(region)
    if n <= 1:
        return region.astype(bool)
    sizes = ndimage.sum_labels(region, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def clean_mask(pred: np.ndarray, min_lesion_voxels: int = 5) -> np.ndarray:
    """Keep the largest liver component, keep tumor only inside it, drop tiny lesions, fill holes."""
    organ = largest_component(pred >= LIVER)
    organ = ndimage.binary_fill_holes(organ)
    tumor = (pred == TUMOR) & organ
    lab, n = ndimage.label(tumor)
    if n:
        sizes = ndimage.sum_labels(tumor, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, np.flatnonzero(sizes >= min_lesion_voxels) + 1)
        tumor = tumor & keep
    out = np.full(pred.shape, BACKGROUND, dtype=np.uint8)
    out[organ] = LIVER
    out[tumor] = TUMOR
    return out
