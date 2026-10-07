import numpy as np
import pytest

from hepatoscan.baseline import VoxelBaseline
from hepatoscan.evaluate import evaluate
from hepatoscan.postprocess import clean_mask
from hepatoscan.preprocess import preprocess_volume


def test_baseline_segments_the_liver(phantoms, fitted_baseline):
    report = evaluate(fitted_baseline, phantoms[6:])
    assert report["volumes"] == 2
    assert report["summary"]["liver_dice"]["mean"] > 0.8
    assert {"tumor_hd95_mm", "lesion_f1", "pixel_accuracy", "tumor_dice_cases_with_tumor"} <= set(report["summary"])


def test_unfitted_baseline_refuses_to_predict(phantoms):
    with pytest.raises(RuntimeError):
        VoxelBaseline().predict_prepared(preprocess_volume(phantoms[0]))


def test_save_and_load_checks_the_hash(tmp_path, fitted_baseline):
    path = fitted_baseline.save(tmp_path / "m.pkl")
    again = VoxelBaseline.load(path)
    assert again.fitted and again.seed == fitted_baseline.seed
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        VoxelBaseline.load(path)


def test_clean_mask_keeps_tumor_inside_the_largest_liver():
    pred = np.zeros((10, 20, 20), np.uint8)
    pred[2:8, 2:12, 2:12] = 1
    pred[4:6, 4:8, 4:8] = 2
    pred[0, 18, 18] = 1  # small separate component
    pred[9, 0, 0] = 2  # tumor outside the liver
    out = clean_mask(pred, min_lesion_voxels=5)
    assert out[0, 18, 18] == 0 and out[9, 0, 0] == 0 and (out == 2).sum() == 32
