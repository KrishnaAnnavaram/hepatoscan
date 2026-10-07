"""The one preprocessing path. Training, evaluation and serving all call ``preprocess_volume``.

Steps:

1. Resample to the target spacing (linear for the image, nearest for the mask).
2. Clip the Hounsfield units and make one channel per CT window, scaled to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .labels import resize_mask
from .volume import Volume, from_arrays


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    level: float
    width: float = Field(gt=0)

    @property
    def low(self) -> float:
        return self.level - self.width / 2

    @property
    def high(self) -> float:
        return self.level + self.width / 2


LIVER_WINDOW = Window(level=60, width=200)
ABDOMEN_WINDOW = Window(level=40, width=400)


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_spacing: tuple[float, float, float] | None = (2.5, 1.5, 1.5)
    windows: tuple[Window, ...] = (LIVER_WINDOW, ABDOMEN_WINDOW)
    hu_clip: tuple[float, float] = (-1024.0, 1024.0)


def apply_window(hu: np.ndarray, window: Window) -> np.ndarray:
    return np.clip((hu - window.low) / (window.high - window.low), 0.0, 1.0).astype(np.float32)


def resample(vol: Volume, target_spacing: tuple[float, float, float]) -> Volume:
    factors = [s / t for s, t in zip(vol.spacing, target_spacing)]
    if np.allclose(factors, 1.0):
        return vol
    image = ndimage.zoom(vol.image, factors, order=1, mode="nearest")
    mask = None if vol.mask is None else resize_mask(vol.mask, image.shape)
    return from_arrays(image, target_spacing, vol.case_id, mask)


@dataclass(frozen=True)
class Prepared:
    channels: np.ndarray  # (C, D, H, W) float32 in [0, 1]
    hu: np.ndarray  # (D, H, W) clipped HU after resampling
    spacing: tuple[float, float, float]
    mask: np.ndarray | None
    original_shape: tuple[int, int, int]
    case_id: str


def preprocess_volume(vol: Volume, cfg: PreprocessConfig = PreprocessConfig()) -> Prepared:
    original_shape = vol.image.shape
    if cfg.target_spacing is not None:
        vol = resample(vol, cfg.target_spacing)
    hu = np.clip(vol.image, *cfg.hu_clip).astype(np.float32)
    channels = np.stack([apply_window(hu, w) for w in cfg.windows])
    return Prepared(channels, hu, vol.spacing, vol.mask, original_shape, vol.case_id)  # type: ignore[arg-type]


def restore_shape(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Bring a predicted mask back to the original voxel grid (nearest neighbour)."""
    if mask.shape == tuple(shape):
        return mask.astype(np.uint8)
    return resize_mask(mask, shape)


def slice_stack(channels: np.ndarray, index: int, context: int = 1) -> np.ndarray:
    """2.5-D input for slice ``index``: the channels of slices index-context .. index+context.

    Slices outside the volume repeat the edge slice. Output shape: (C * (2 * context + 1), H, W).
    """
    D = channels.shape[1]
    idx = np.clip(np.arange(index - context, index + context + 1), 0, D - 1)
    return channels[:, idx].reshape(-1, *channels.shape[2:])
