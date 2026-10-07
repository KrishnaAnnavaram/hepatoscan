"""Segmentation metrics, computed per volume.

* Dice and IoU for the liver region (label >= 1) and the tumor region (label 2).
* HD95: the 95th percentile of the symmetric surface distance in mm.
* Lesion detection: connected tumor components. A true lesion counts as
  found if a predicted tumor voxel touches it. A predicted lesion counts as
  correct if it touches a true lesion.
* Pixel accuracy is also reported, to show how little it says: background
  dominates it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .labels import liver_region, tumor_region


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    p, t = pred.astype(bool), truth.astype(bool)
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else float(2.0 * (p & t).sum() / denom)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    p, t = pred.astype(bool), truth.astype(bool)
    union = (p | t).sum()
    return 1.0 if union == 0 else float((p & t).sum() / union)


def _surface(region: np.ndarray) -> np.ndarray:
    return region & ~ndimage.binary_erosion(region, border_value=0)


def hd95(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """0.0 if both regions are empty, ``inf`` if only one region is empty."""
    p, t = pred.astype(bool), truth.astype(bool)
    if not p.any() and not t.any():
        return 0.0
    if not p.any() or not t.any():
        return math.inf
    sp, st = _surface(p), _surface(t)
    dt_t = ndimage.distance_transform_edt(~st, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_t[sp], dt_p[st]])
    return float(np.percentile(d, 95))


@dataclass(frozen=True)
class LesionStats:
    true_lesions: int
    found_lesions: int
    predicted_lesions: int
    correct_predictions: int

    @property
    def recall(self) -> float:
        return 1.0 if self.true_lesions == 0 else self.found_lesions / self.true_lesions

    @property
    def precision(self) -> float:
        return 1.0 if self.predicted_lesions == 0 else self.correct_predictions / self.predicted_lesions

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def lesion_stats(pred_tumor: np.ndarray, true_tumor: np.ndarray) -> LesionStats:
    structure = ndimage.generate_binary_structure(3, 3)
    t_lab, n_t = ndimage.label(true_tumor, structure)
    p_lab, n_p = ndimage.label(pred_tumor, structure)
    found = len(set(np.unique(t_lab[pred_tumor.astype(bool)])) - {0})
    correct = len(set(np.unique(p_lab[true_tumor.astype(bool)])) - {0})
    return LesionStats(n_t, found, n_p, correct)


@dataclass(frozen=True)
class VolumeMetrics:
    case_id: str
    liver_dice: float
    liver_iou: float
    liver_hd95_mm: float
    tumor_dice: float
    tumor_iou: float
    tumor_hd95_mm: float
    lesion_recall: float
    lesion_precision: float
    lesion_f1: float
    true_lesions: int
    predicted_lesions: int
    pixel_accuracy: float
    liver_volume_error_ml: float
    tumor_volume_error_ml: float

    def as_dict(self) -> dict:
        return asdict(self)


def volume_metrics(case_id: str, pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]) -> VolumeMetrics:
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {truth.shape}")
    voxel_ml = float(np.prod(spacing)) / 1000.0
    pl, tl = liver_region(pred), liver_region(truth)
    pt, tt = tumor_region(pred), tumor_region(truth)
    les = lesion_stats(pt, tt)
    return VolumeMetrics(
        case_id=case_id,
        liver_dice=dice(pl, tl), liver_iou=iou(pl, tl), liver_hd95_mm=hd95(pl, tl, spacing),
        tumor_dice=dice(pt, tt), tumor_iou=iou(pt, tt), tumor_hd95_mm=hd95(pt, tt, spacing),
        lesion_recall=les.recall, lesion_precision=les.precision, lesion_f1=les.f1,
        true_lesions=les.true_lesions, predicted_lesions=les.predicted_lesions,
        pixel_accuracy=float((pred == truth).mean()),
        liver_volume_error_ml=float((pl.sum() - tl.sum()) * voxel_ml),
        tumor_volume_error_ml=float((pt.sum() - tt.sum()) * voxel_ml),
    )


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval over volumes. Non-finite values are left out."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(v.mean()), float(lo), float(hi)
