import numpy as np
import pytest

from hepatoscan.synthetic import make_phantom


@pytest.fixture(scope="session")
def phantoms():
    rng = np.random.default_rng(7)
    return [make_phantom(rng, (16, 40, 40), f"p{i:02d}") for i in range(8)]


@pytest.fixture(scope="session")
def fitted_baseline(phantoms):
    from hepatoscan.baseline import VoxelBaseline

    return VoxelBaseline(per_class=800, max_iter=60, seed=0).fit(phantoms[:6])
