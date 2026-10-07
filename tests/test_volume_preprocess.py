import numpy as np
import pytest

from hepatoscan.preprocess import LIVER_WINDOW, PreprocessConfig, apply_window, preprocess_volume, resample, slice_stack
from hepatoscan.segmenter import segment
from hepatoscan.volume import VolumeError, from_arrays, load_file, load_npz_bytes, save_npz


def test_volume_schema_errors():
    good = np.zeros((10, 10, 10), np.float32)
    with pytest.raises(VolumeError):
        from_arrays(np.zeros((10, 10)), (1, 1, 1))
    with pytest.raises(VolumeError):
        from_arrays(np.full((10, 10, 10), 25500.0), (1, 1, 1))  # not Hounsfield units
    with pytest.raises(VolumeError):
        from_arrays(good, (1, 0, 1))
    with pytest.raises(VolumeError):
        from_arrays(good, (1, 1, 1), mask=np.zeros((9, 10, 10)))
    with pytest.raises(VolumeError):
        from_arrays(np.zeros((4, 10, 10)), (1, 1, 1))


def test_npz_round_trip(tmp_path, phantoms):
    vol = phantoms[0]
    path = save_npz(vol, tmp_path / "v.npz")
    back = load_file(path)
    assert np.allclose(back.image, vol.image) and np.array_equal(back.mask, vol.mask)
    with pytest.raises(VolumeError):
        load_npz_bytes(b"PK\x03\x04garbage")


def test_window_scales_to_unit_range():
    hu = np.array([-1000.0, LIVER_WINDOW.low, LIVER_WINDOW.level, LIVER_WINDOW.high, 3000.0])
    assert apply_window(hu, LIVER_WINDOW).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_resample_keeps_labels(phantoms):
    out = resample(phantoms[1], (1.0, 1.0, 1.0))
    assert set(np.unique(out.mask)) <= {0, 1, 2}
    assert out.image.shape == out.mask.shape


def test_slice_stack_repeats_edges():
    ch = np.arange(2 * 4 * 2 * 2, dtype=np.float32).reshape(2, 4, 2, 2)
    s = slice_stack(ch, 0, context=1)
    assert s.shape == (6, 2, 2)
    assert np.array_equal(s[0], ch[0, 0]) and np.array_equal(s[1], ch[0, 0]) and np.array_equal(s[2], ch[0, 1])


def test_serving_uses_the_segmenter_preprocessing(phantoms):
    seen = []

    class Spy:
        name = "spy"
        preprocess = PreprocessConfig(target_spacing=None)

        def predict_prepared(self, prep):
            seen.append(prep.channels.shape)
            return np.zeros(prep.channels.shape[1:], np.uint8)

    vol = phantoms[0]
    mask, summary = segment(Spy(), vol)
    assert seen == [preprocess_volume(vol, Spy.preprocess).channels.shape]
    assert mask.shape == vol.image.shape and summary.lesion_count == 0
