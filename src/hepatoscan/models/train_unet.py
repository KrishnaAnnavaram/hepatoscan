"""Seeded training, strict checkpoint loading and the torch segmenter (needs torch)."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from ..labels import validate_mask
from ..metrics import dice
from ..postprocess import clean_mask
from ..preprocess import PreprocessConfig, Prepared, preprocess_volume, slice_stack
from ..volume import Volume
from .unet import AttentionUNet, UNetConfig, build, dice_ce_loss


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(8, ge=1)
    lr: float = Field(1e-3, gt=0)
    patience: int = Field(3, ge=1)
    keep_empty_fraction: float = Field(0.2, ge=0, le=1)  # share of slices without liver that stay in training
    slice_size: int = Field(96, ge=16)  # training slices are centre-cropped or zero-padded to this size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def fit_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop or zero-pad the last two axes to (size, size)."""
    out = arr
    for axis in (-2, -1):
        n = out.shape[axis]
        if n > size:
            start = (n - size) // 2
            out = np.take(out, np.arange(start, start + size), axis=axis)
        elif n < size:
            pad = [(0, 0)] * out.ndim
            pad[axis] = ((size - n) // 2, size - n - (size - n) // 2)
            out = np.pad(out, pad)
    return out


def slices_of(prep: Prepared, context: int, rng: np.random.Generator, keep_empty: float,
              size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """2.5-D inputs and label slices. Slices without liver are kept with probability ``keep_empty``."""
    assert prep.mask is not None
    mask = validate_mask(prep.mask)
    xs, ys = [], []
    for k in range(mask.shape[0]):
        if not mask[k].any() and rng.random() > keep_empty:
            continue
        xs.append(fit_to(slice_stack(prep.channels, k, context), size))
        ys.append(fit_to(mask[k], size))
    if not xs:
        return (np.empty((0, prep.channels.shape[0] * (2 * context + 1), size, size), np.float32),
                np.empty((0, size, size), np.uint8))
    return np.stack(xs), np.stack(ys)


class TorchSegmenter:
    def __init__(self, model: AttentionUNet, preprocess: PreprocessConfig, device: str = "cpu",
                 min_lesion_voxels: int = 5) -> None:
        if len(preprocess.windows) != model.cfg.in_windows:
            raise ValueError(f"the model expects {model.cfg.in_windows} CT windows, the preprocessing gives "
                             f"{len(preprocess.windows)}")
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.device = device
        self.min_lesion_voxels = min_lesion_voxels
        self.name = "attention-unet-2.5d"

    @torch.no_grad()
    def predict_prepared(self, prep: Prepared) -> np.ndarray:
        ctx = self.model.cfg.context
        D = prep.channels.shape[1]
        out = np.zeros(prep.channels.shape[1:], dtype=np.uint8)
        for k in range(D):
            x = torch.as_tensor(slice_stack(prep.channels, k, ctx)[None], device=self.device)
            out[k] = self.model(x).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return clean_mask(out, self.min_lesion_voxels)


def train(volumes_train: Sequence[Volume], volumes_val: Sequence[Volume], ucfg: UNetConfig = UNetConfig(),
          tcfg: TrainConfig = TrainConfig(), pcfg: PreprocessConfig = PreprocessConfig(), device: str = "cpu",
          log: Callable[[str], None] = print) -> tuple[TorchSegmenter, list[dict]]:
    set_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    xs, ys = [], []
    for vol in volumes_train:
        x, y = slices_of(preprocess_volume(vol, pcfg), ucfg.context, rng, tcfg.keep_empty_fraction,
                         tcfg.slice_size)
        xs.append(x)
        ys.append(y)
    X = torch.as_tensor(np.concatenate(xs))
    Y = torch.as_tensor(np.concatenate(ys).astype(np.int64))
    model = build(ucfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(tcfg.epochs):
        model.train()
        order = rng.permutation(len(X))
        losses = []
        for i in range(0, len(order), tcfg.batch_size):
            idx = torch.as_tensor(order[i : i + tcfg.batch_size])
            xb, yb = X[idx].to(device), Y[idx].to(device)
            loss = dice_ce_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        seg = TorchSegmenter(model, pcfg, device)
        val = float(np.mean([_liver_tumor_score(seg, v) for v in volumes_val])) if volumes_val else -float(np.mean(losses))
        model.train()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_score": val})
        log(f"epoch {epoch}  loss {np.mean(losses):.4f}  val mean Dice {val:.4f}")
        if val > best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return TorchSegmenter(model, pcfg, device), history


def _liver_tumor_score(seg: TorchSegmenter, vol: Volume) -> float:
    from ..segmenter import segment

    assert vol.mask is not None
    pred, _ = segment(seg, vol)
    return 0.5 * (dice(pred >= 1, vol.mask >= 1) + dice(pred == 2, vol.mask == 2))


def save_checkpoint(seg: TorchSegmenter, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": seg.model.state_dict(), "unet_config": seg.model.cfg.model_dump(),
                "preprocess": seg.preprocess.model_dump()}, path)
    return path


class CheckpointError(RuntimeError):
    """The weights do not fit the model definition. Serving must stop, never continue with random weights."""


def load_checkpoint(path: str | os.PathLike, device: str = "cpu") -> TorchSegmenter:
    blob = torch.load(path, map_location=device, weights_only=True)
    try:
        ucfg = UNetConfig.model_validate(blob["unet_config"])
        pcfg = PreprocessConfig.model_validate(blob["preprocess"])
        model = build(ucfg)
        model.load_state_dict(blob["state_dict"], strict=True)
        return TorchSegmenter(model, pcfg, device)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"cannot load {path}: {exc}") from exc
