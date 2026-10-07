import math

import numpy as np

from hepatoscan.metrics import bootstrap_mean_ci, dice, hd95, iou, lesion_stats, volume_metrics


def test_dice_and_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[1:3] = True
    assert dice(a, b) == 0.5 and iou(a, b) == 1 / 3
    assert dice(np.zeros(3), np.zeros(3)) == 1.0


def test_hd95_of_a_shifted_cube():
    a = np.zeros((10, 10, 10), bool)
    a[2:6, 2:6, 2:6] = True
    b = np.roll(a, 2, axis=2)
    assert hd95(a, b, (1.0, 1.0, 2.0)) == 4.0
    assert hd95(a, a, (1, 1, 1)) == 0.0
    assert math.isinf(hd95(a, np.zeros_like(a), (1, 1, 1)))


def test_lesion_detection_counts():
    t = np.zeros((10, 10, 10), bool)
    p = np.zeros_like(t)
    t[1:3, 1:3, 1:3] = True
    t[6:8, 6:8, 6:8] = True
    p[1:2, 1:2, 1:2] = True  # finds lesion 1
    p[4, 0, 9] = True  # false positive
    s = lesion_stats(p, t)
    assert (s.true_lesions, s.found_lesions, s.predicted_lesions, s.correct_predictions) == (2, 1, 2, 1)
    assert s.recall == 0.5 and s.precision == 0.5 and s.f1 == 0.5


def test_pixel_accuracy_hides_a_useless_prediction():
    truth = np.zeros((8, 32, 32), np.uint8)
    truth[3:5, 10:14, 10:14] = 1
    truth[4, 11, 11] = 2
    pred = np.zeros_like(truth)  # predicts background everywhere
    m = volume_metrics("x", pred, truth, (1.0, 1.0, 1.0))
    assert m.pixel_accuracy > 0.99
    assert m.liver_dice == 0.0 and m.lesion_recall == 0.0


def test_bootstrap_ignores_non_finite():
    mean, lo, hi = bootstrap_mean_ci(np.array([1.0, 2.0, math.inf, 3.0]))
    assert mean == 2.0 and lo <= mean <= hi
    assert all(math.isnan(v) for v in bootstrap_mean_ci(np.array([math.inf])))
