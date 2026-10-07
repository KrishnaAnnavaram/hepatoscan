"""Classical baseline: a per-voxel gradient-boosting classifier on local intensity features.

It needs only numpy, scipy and scikit-learn, so the offline demo and CI can
train and evaluate a real segmenter. The scaler and the classifier live in one
scikit-learn ``Pipeline`` that is fit on training volumes only.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labels import NUM_CLASSES, validate_mask
from .postprocess import clean_mask
from .preprocess import PreprocessConfig, Prepared, preprocess_volume
from .volume import Volume

# joblib warns on Windows machines without `wmic`; the warning has no effect on the results
warnings.filterwarnings("ignore", message="Could not find the number of physical cores", category=UserWarning)

FEATURES = ("window_0", "window_1", "mean_3", "std_3", "mean_7", "grad_mag", "z_pos")


def voxel_features(prep: Prepared) -> np.ndarray:
    """(D*H*W, len(FEATURES)) float32 features from the preprocessed channels."""
    c0, c1 = prep.channels[0], prep.channels[1]
    mean3 = ndimage.uniform_filter(c0, size=3)
    sq3 = ndimage.uniform_filter(c0 * c0, size=3)
    std3 = np.sqrt(np.maximum(sq3 - mean3 * mean3, 0.0))
    mean7 = ndimage.uniform_filter(c0, size=(3, 7, 7))
    grad = ndimage.gaussian_gradient_magnitude(c1, sigma=1.0)
    D = c0.shape[0]
    zpos = np.broadcast_to(np.linspace(0.0, 1.0, D)[:, None, None], c0.shape)
    feats = np.stack([c0, c1, mean3, std3, mean7, grad, zpos], axis=-1)
    return feats.reshape(-1, len(FEATURES)).astype(np.float32)


def sample_voxels(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of at most ``per_class`` voxels of each class (class-balanced, seeded)."""
    flat = labels.ravel()
    picks = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(flat == c)
        if idx.size:
            picks.append(rng.choice(idx, size=min(per_class, idx.size), replace=False))
    return np.concatenate(picks)


class VoxelBaseline:
    name = "voxel-gbm"

    def __init__(self, preprocess: PreprocessConfig = PreprocessConfig(), per_class: int = 3000,
                 max_iter: int = 150, seed: int = 0, min_lesion_voxels: int = 10) -> None:
        self.preprocess = preprocess
        self.per_class = per_class
        self.seed = seed
        self.min_lesion_voxels = min_lesion_voxels
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("gbm", HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, random_state=seed)),
        ])
        self.fitted = False

    def fit(self, volumes: Iterable[Volume]) -> "VoxelBaseline":
        rng = np.random.default_rng(self.seed)
        X, y = [], []
        for vol in volumes:
            if vol.mask is None:
                raise ValueError(f"{vol.case_id} has no mask")
            prep = preprocess_volume(vol, self.preprocess)
            assert prep.mask is not None
            labels = validate_mask(prep.mask)
            idx = sample_voxels(labels, self.per_class, rng)
            X.append(voxel_features(prep)[idx])
            y.append(labels.ravel()[idx])
        self.pipeline.fit(np.concatenate(X), np.concatenate(y))
        self.fitted = True
        return self

    def predict_prepared(self, prep: Prepared) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("the baseline is not fitted")
        shape = prep.channels.shape[1:]
        proba = self.pipeline.predict_proba(voxel_features(prep)).reshape(*shape, -1)
        # smooth each class probability over a 3x3x3 neighbourhood: removes isolated noisy voxels
        smooth = np.stack([ndimage.uniform_filter(proba[..., c], size=3) for c in range(proba.shape[-1])], axis=-1)
        raw = self.pipeline.classes_[np.argmax(smooth, axis=-1)].astype(np.uint8)
        return clean_mask(raw, self.min_lesion_voxels)

    # persistence ------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the pickled model and a manifest with its SHA-256 value."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = pickle.dumps(self)
        path.write_bytes(blob)
        path.with_suffix(".json").write_text(json.dumps({
            "model": self.name, "sha256": hashlib.sha256(blob).hexdigest(),
            "preprocess": self.preprocess.model_dump(), "seed": self.seed,
        }, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "VoxelBaseline":
        """Load only a file whose SHA-256 value matches its manifest. Never load a pickle from an unknown source."""
        path = Path(path)
        blob = path.read_bytes()
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(blob).hexdigest() != manifest.get("sha256"):
            raise ValueError(f"{path} does not match the SHA-256 value in its manifest")
        model = pickle.loads(blob)  # noqa: S301 - hash-checked file written by save()
        if not isinstance(model, VoxelBaseline):
            raise TypeError("the file does not hold a VoxelBaseline")
        return model
