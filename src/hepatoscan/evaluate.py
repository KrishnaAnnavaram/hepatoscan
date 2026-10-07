"""Evaluate a segmenter on held-out volumes with per-volume metrics and bootstrap intervals."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .metrics import VolumeMetrics, bootstrap_mean_ci, volume_metrics
from .segmenter import Segmenter, segment
from .volume import Volume

SUMMARY_KEYS = ("liver_dice", "tumor_dice", "liver_hd95_mm", "tumor_hd95_mm", "lesion_recall",
                "lesion_precision", "lesion_f1", "pixel_accuracy", "liver_volume_error_ml", "tumor_volume_error_ml")


def evaluate(segmenter: Segmenter, volumes: Iterable[Volume]) -> dict:
    rows: list[VolumeMetrics] = []
    for vol in volumes:
        if vol.mask is None:
            raise ValueError(f"{vol.case_id} has no ground-truth mask")
        pred, _ = segment(segmenter, vol)
        rows.append(volume_metrics(vol.case_id, pred, vol.mask, vol.spacing))
    summary = {}
    for key in SUMMARY_KEYS:
        values = np.array([getattr(r, key) for r in rows], dtype=float)
        mean, lo, hi = bootstrap_mean_ci(values)
        summary[key] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                        "n_finite": int(np.isfinite(values).sum()), "n": int(values.size)}
    # tumor Dice only on volumes that have a tumor: an empty-empty pair scores 1 and inflates the mean
    with_tumor = np.array([r.tumor_dice for r in rows if r.true_lesions > 0], dtype=float)
    mean, lo, hi = bootstrap_mean_ci(with_tumor)
    summary["tumor_dice_cases_with_tumor"] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                                              "n_finite": int(with_tumor.size), "n": int(with_tumor.size)}
    return {"model": segmenter.name, "volumes": len(rows), "summary": summary,
            "per_volume": [{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in r.as_dict().items()}
                           for r in rows]}
