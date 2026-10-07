"""Synthetic abdominal CT phantoms with exact liver and tumor masks.

A phantom is not a real CT scan. It has air, a fat layer, soft tissue, a spine,
a liver and zero to three hypodense lesions, plus Gaussian noise. The demo and
the tests use phantoms, so they need no download.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .labels import LIVER, TUMOR
from .volume import Volume, from_arrays, load_file, save_npz


def make_phantom(rng: np.random.Generator, shape: tuple[int, int, int] = (24, 64, 64),
                 case_id: str = "phantom", max_lesions: int = 3) -> Volume:
    D, H, W = shape
    z, y, x = np.meshgrid(np.linspace(-1, 1, D), np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    image = np.full(shape, -1000.0, dtype=np.float32)
    body = (y / 0.85) ** 2 + (x / 0.95) ** 2 <= 1.0
    inner = (y / 0.78) ** 2 + (x / 0.88) ** 2 <= 1.0
    image[body] = -100.0  # fat layer
    image[inner] = 40.0  # soft tissue
    spine = ((y - 0.55) / 0.12) ** 2 + (x / 0.12) ** 2 <= 1.0
    image[spine] = 450.0
    # liver: an ellipsoid on the left side of the image (patient right)
    cz, cy, cx = rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.05), rng.uniform(-0.45, -0.3)
    rz, ry, rx = rng.uniform(0.6, 0.85), rng.uniform(0.35, 0.45), rng.uniform(0.3, 0.4)
    liver = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
    liver &= inner
    liver_hu = rng.uniform(55, 70)
    image[liver] = liver_hu
    mask = np.zeros(shape, dtype=np.uint8)
    mask[liver] = LIVER
    n_lesions = int(rng.integers(1, max_lesions + 1)) if rng.random() < 0.8 else 0
    liver_idx = np.argwhere(liver)
    for _ in range(n_lesions):
        center = liver_idx[rng.integers(len(liver_idx))]
        radius = rng.uniform(3.0, 6.0)  # voxels
        dz, dy, dx = (np.indices(shape).transpose(1, 2, 3, 0) - center).transpose(3, 0, 1, 2)
        lesion = (dz * 2.0) ** 2 + dy ** 2 + dx ** 2 <= radius ** 2  # slices are thicker
        lesion &= liver
        image[lesion] = liver_hu - rng.uniform(30, 45)
        mask[lesion] = TUMOR
    image += rng.normal(0.0, rng.uniform(8, 14), size=shape).astype(np.float32)
    spacing = (float(rng.uniform(2.0, 3.0)), float(rng.uniform(1.2, 1.8)), float(rng.uniform(1.2, 1.8)))
    return from_arrays(image, spacing, case_id, mask)


def write_dataset(out_dir: str | Path, n_cases: int = 20, seed: int = 0,
                  shape: tuple[int, int, int] = (24, 64, 64)) -> Path:
    """Write ``case_XXX.npz`` files and ``manifest.csv``. Returns the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        cid = f"case_{i:03d}"
        vol = make_phantom(rng, shape, cid)
        save_npz(vol, out / f"{cid}.npz")
        assert vol.mask is not None
        rows.append({"case_id": cid, "file": f"{cid}.npz", "depth": shape[0], "height": shape[1], "width": shape[2],
                     "tumor_voxels": int((vol.mask == TUMOR).sum())})
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "dataset.json").write_text(json.dumps({"kind": "synthetic_phantom", "n_cases": n_cases, "seed": seed,
                                                  "shape": list(shape)}, indent=2), encoding="utf-8")
    return manifest


REQUIRED_COLUMNS = ("case_id", "file")


def load_dataset(data_dir: str | Path) -> dict[str, Volume]:
    """Read the volumes listed in ``manifest.csv``. Each volume must have a mask."""
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run `hepatoscan synth` or see data/README.md")
    with manifest.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest.csv misses columns {missing}")
        rows = list(reader)
    vols = {}
    for row in rows:
        vol = load_file(data_dir / row["file"])
        if vol.mask is None:
            raise ValueError(f"{row['file']} has no mask")
        vols[row["case_id"]] = Volume(vol.image, vol.spacing, row["case_id"], vol.mask)
    return vols
