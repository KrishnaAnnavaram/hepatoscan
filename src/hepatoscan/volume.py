"""CT volumes: the in-memory type, schema checks and file input/output.

Supported files:

* ``.npz`` with ``image`` (D, H, W) in Hounsfield units, ``spacing`` (3,) in mm
  and, for training data, ``mask`` (D, H, W) with labels 0/1/2.
* ``.nii`` and ``.nii.gz`` through nibabel (extra ``nifti``). The axes are
  reordered to (slice, row, column).
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import validate_mask

MIN_DIM, MAX_DIM = 8, 1024
HU_MIN, HU_MAX = -2048.0, 4096.0


class VolumeError(ValueError):
    """A volume does not match the expected schema."""


@dataclass(frozen=True)
class Volume:
    image: np.ndarray  # (D, H, W) float32 HU
    spacing: tuple[float, float, float]  # mm along (D, H, W)
    case_id: str = "case"
    mask: np.ndarray | None = None  # (D, H, W) uint8 labels

    def __post_init__(self) -> None:
        img = self.image
        if img.ndim != 3:
            raise VolumeError(f"image must be 3-D, got shape {img.shape}")
        if any(d < MIN_DIM or d > MAX_DIM for d in img.shape):
            raise VolumeError(f"each image axis must be in [{MIN_DIM}, {MAX_DIM}], got {img.shape}")
        if not np.all(np.isfinite(img)):
            raise VolumeError("image has non-finite values")
        lo, hi = float(img.min()), float(img.max())
        if lo < HU_MIN or hi > HU_MAX:
            raise VolumeError(f"values [{lo:.0f}, {hi:.0f}] are not CT Hounsfield units")
        if len(self.spacing) != 3 or any(not (0 < s <= 20) for s in self.spacing):
            raise VolumeError(f"spacing must be 3 values in (0, 20] mm, got {self.spacing}")
        if self.mask is not None:
            if self.mask.shape != img.shape:
                raise VolumeError(f"mask shape {self.mask.shape} differs from image shape {img.shape}")
            validate_mask(self.mask)

    @property
    def voxel_ml(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0


def from_arrays(image: np.ndarray, spacing, case_id: str = "case", mask: np.ndarray | None = None) -> Volume:
    sp = tuple(float(s) for s in np.asarray(spacing, dtype=float).ravel())
    m = None if mask is None else validate_mask(mask)
    return Volume(np.asarray(image, dtype=np.float32), sp, case_id, m)  # type: ignore[arg-type]


def save_npz(vol: Volume, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"image": vol.image.astype(np.float32), "spacing": np.asarray(vol.spacing, dtype=np.float32)}
    if vol.mask is not None:
        arrays["mask"] = vol.mask.astype(np.uint8)
    np.savez_compressed(path, **arrays)
    return path


def load_npz_bytes(data: bytes, case_id: str = "upload") -> Volume:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            if "image" not in z.files or "spacing" not in z.files:
                raise VolumeError("npz file needs the arrays 'image' and 'spacing'")
            image, spacing = z["image"], z["spacing"]
            mask = z["mask"] if "mask" in z.files else None
    except VolumeError:
        raise
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        raise VolumeError(f"not a readable npz file: {exc}") from exc
    return from_arrays(image, spacing, case_id, mask)


def load_nifti_bytes(data: bytes, case_id: str = "upload", gz: bool = False) -> Volume:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise VolumeError("NIfTI input needs the 'nifti' extra (nibabel)") from exc
    import gzip

    raw = gzip.decompress(data) if gz else data
    try:
        img = nib.Nifti1Image.from_bytes(raw)
    except Exception as exc:  # nibabel raises several types
        raise VolumeError(f"not a readable NIfTI file: {exc}") from exc
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    if arr.ndim != 3:
        raise VolumeError(f"NIfTI image must be 3-D, got shape {arr.shape}")
    zooms = img.header.get_zooms()[:3]
    # NIfTI stores (x, y, z). hepatoscan works with (slice, row, column) = (z, y, x).
    image = np.transpose(arr, (2, 1, 0))
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return from_arrays(image, spacing, case_id)


def load_file(path: str | os.PathLike) -> Volume:
    path = Path(path)
    data = path.read_bytes()
    name = path.name.lower()
    if name.endswith(".npz"):
        return load_npz_bytes(data, path.stem)
    if name.endswith(".nii.gz"):
        return load_nifti_bytes(data, path.name[:-7], gz=True)
    if name.endswith(".nii"):
        return load_nifti_bytes(data, path.stem)
    raise VolumeError(f"unsupported file type: {path.name}")
