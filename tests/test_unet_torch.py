"""Tests of the deep segmenter. They skip when torch is not installed (the CI installs only the dev extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hepatoscan.models.train_unet import (  # noqa: E402
    CheckpointError, TorchSegmenter, TrainConfig, load_checkpoint, save_checkpoint, train,
)
from hepatoscan.models.unet import AttentionGate, AttentionUNet, UNetConfig, dice_ce_loss  # noqa: E402
from hepatoscan.preprocess import PreprocessConfig, preprocess_volume  # noqa: E402

SMALL = UNetConfig(base=4, depth=2)


def test_forward_shape_for_odd_sizes():
    torch.manual_seed(0)
    m = AttentionUNet(SMALL).eval()
    out = m(torch.randn(2, SMALL.in_channels, 37, 29))
    assert out.shape == (2, 3, 37, 29)


def test_attention_gates_follow_the_config():
    with_gates = AttentionUNet(SMALL)
    without = AttentionUNet(UNetConfig(base=4, depth=2, attention_gates=False))
    assert any(isinstance(x, AttentionGate) for x in with_gates.modules())
    assert not any(isinstance(x, AttentionGate) for x in without.modules())


def test_loss_is_multiclass_and_decreases():
    torch.manual_seed(0)
    m = AttentionUNet(SMALL)
    x = torch.randn(4, SMALL.in_channels, 16, 16)
    y = torch.randint(0, 3, (4, 16, 16))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(15):
        loss = dice_ce_loss(m(x), y)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first


def test_train_save_and_strict_load(tmp_path, phantoms):
    seg, hist = train(phantoms[:3], phantoms[3:4], SMALL, TrainConfig(epochs=1, batch_size=8), log=lambda s: None)
    assert len(hist) == 1
    path = save_checkpoint(seg, tmp_path / "unet.pt")
    again = load_checkpoint(path)
    prep = preprocess_volume(phantoms[5], seg.preprocess)
    assert np.array_equal(seg.predict_prepared(prep), again.predict_prepared(prep))


def test_mismatched_weights_stop_loading(tmp_path, phantoms):
    seg = TorchSegmenter(AttentionUNet(SMALL), PreprocessConfig())
    path = save_checkpoint(seg, tmp_path / "unet.pt")
    blob = torch.load(path, weights_only=True)
    blob["unet_config"]["base"] = 8  # serving definition differs from the trained weights
    torch.save(blob, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_window_count_must_match_the_model():
    with pytest.raises(ValueError):
        TorchSegmenter(AttentionUNet(SMALL), PreprocessConfig(windows=PreprocessConfig().windows[:1]))
