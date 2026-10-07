"""The segmenter interface and the result summary that serving and the assistant use."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import ndimage

from .labels import LIVER, TUMOR
from .preprocess import PreprocessConfig, Prepared, preprocess_volume, restore_shape
from .volume import Volume


@runtime_checkable
class Segmenter(Protocol):
    name: str
    preprocess: PreprocessConfig

    def predict_prepared(self, prep: Prepared) -> np.ndarray:  # pragma: no cover - protocol
        """Labels (D, H, W) on the grid of ``prep``."""
        ...


@dataclass(frozen=True)
class SegmentationSummary:
    model: str
    liver_volume_ml: float
    tumor_volume_ml: float
    lesion_count: int
    largest_lesion_ml: float
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "liver_volume_ml": round(self.liver_volume_ml, 1),
            "tumor_volume_ml": round(self.tumor_volume_ml, 1),
            "lesion_count": self.lesion_count,
            "largest_lesion_ml": round(self.largest_lesion_ml, 2),
            "shape": list(self.shape),
            "spacing_mm": [round(s, 3) for s in self.spacing_mm],
            "note": "Automatic measurement for research and education. Not a diagnosis. A clinician must review it.",
        }

    def as_text(self) -> str:
        return (f"Automatic measurement by {self.model} (not a diagnosis): liver region {self.liver_volume_ml:.0f} mL, "
                f"lesion candidates {self.lesion_count}, total lesion volume {self.tumor_volume_ml:.1f} mL.")


def summarize(mask: np.ndarray, spacing: tuple[float, float, float], model: str) -> SegmentationSummary:
    voxel_ml = float(np.prod(spacing)) / 1000.0
    tumor = mask == TUMOR
    lab, n = ndimage.label(tumor, ndimage.generate_binary_structure(3, 3))
    largest = float(ndimage.sum_labels(tumor, lab, np.arange(1, n + 1)).max()) * voxel_ml if n else 0.0
    return SegmentationSummary(model, float((mask >= LIVER).sum()) * voxel_ml, float(tumor.sum()) * voxel_ml,
                               int(n), largest, tuple(mask.shape), tuple(spacing))  # type: ignore[arg-type]


def segment(segmenter: Segmenter, vol: Volume) -> tuple[np.ndarray, SegmentationSummary]:
    """Preprocess with the segmenter's own config, predict, and return labels on the original grid."""
    prep = preprocess_volume(vol, segmenter.preprocess)
    pred = segmenter.predict_prepared(prep)
    mask = restore_shape(pred, vol.image.shape)
    return mask, summarize(mask, vol.spacing, segmenter.name)
