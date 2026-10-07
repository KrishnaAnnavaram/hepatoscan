import numpy as np
import pytest

from hepatoscan.labels import LIVER, TUMOR
from hepatoscan.splits import split_cases
from hepatoscan.synthetic import load_dataset, make_phantom, write_dataset


def test_split_is_per_case_disjoint_and_seeded():
    ids = [f"c{i}" for i in range(20)]
    a = split_cases(ids, 0.15, 0.25, seed=1)
    b = split_cases(ids, 0.15, 0.25, seed=1)
    assert a == b
    assert not (set(a.val) & set(a.test)) and not (set(a.train) & set(a.test))
    assert len(a.train) + len(a.val) + len(a.test) == 20 and len(a.test) == 5
    with pytest.raises(ValueError):
        split_cases(["a", "a", "b"])


def test_phantom_tumor_lies_inside_liver():
    rng = np.random.default_rng(3)
    for _ in range(5):
        vol = make_phantom(rng, (16, 32, 32))
        assert (vol.mask == LIVER).any()
        tumor = vol.mask == TUMOR
        if tumor.any():
            assert vol.image[tumor].mean() < vol.image[vol.mask == LIVER].mean()


def test_dataset_write_and_load(tmp_path):
    write_dataset(tmp_path, n_cases=4, seed=0, shape=(12, 24, 24))
    vols = load_dataset(tmp_path)
    assert sorted(vols) == ["case_000", "case_001", "case_002", "case_003"]
    (tmp_path / "manifest.csv").write_text("id,path\n")
    with pytest.raises(ValueError):
        load_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing")
