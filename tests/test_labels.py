import numpy as np
import pytest

from hepatoscan.labels import LabelError, liver_region, one_hot, resize_mask, tumor_region, validate_mask


def test_validate_mask_rejects_unknown_and_fractional_labels():
    with pytest.raises(LabelError):
        validate_mask(np.array([[0, 1, 3]]))
    with pytest.raises(LabelError):
        validate_mask(np.array([[0.0, 0.5, 1.0]]))
    assert validate_mask(np.array([[0.0, 2.0]])).dtype == np.uint8


def test_resize_mask_never_makes_new_values():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=(5, 17, 13)).astype(np.uint8)
    for shape in [(9, 31, 7), (3, 8, 8), (5, 64, 64)]:
        out = resize_mask(m, shape)
        assert out.shape == shape
        assert set(np.unique(out)) <= {0, 1, 2}


def test_three_classes_are_kept_apart():
    m = np.array([0, 1, 2, 2, 1])
    oh = one_hot(m)
    assert oh.shape == (3, 5) and np.array_equal(oh.argmax(0), m)
    assert liver_region(m).tolist() == [False, True, True, True, True]
    assert tumor_region(m).tolist() == [False, False, True, True, False]
