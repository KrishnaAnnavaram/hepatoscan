"""Per-volume splits. All slices of one CT volume stay in one split."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CaseSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def check(self) -> None:
        a, b, c = set(self.train), set(self.val), set(self.test)
        if a & b or a & c or b & c:
            raise ValueError("a case is in more than one split")


def split_cases(case_ids: Sequence[str], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> CaseSplit:
    """Seeded split of case ids into disjoint train, validation and test sets."""
    ids = sorted(set(case_ids))
    if len(ids) != len(case_ids):
        raise ValueError("duplicate case ids")
    if len(ids) < 3:
        raise ValueError("need at least 3 cases")
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("fractions must be in (0, 1) and sum to less than 1")
    perm = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    n_test = max(1, round(len(ids) * test_frac))
    n_val = max(1, round(len(ids) * val_frac))
    if n_test + n_val >= len(ids):
        raise ValueError("too few cases for this split")
    split = CaseSplit(tuple(sorted(perm[n_test + n_val :])), tuple(sorted(perm[n_test : n_test + n_val])),
                      tuple(sorted(perm[:n_test])))
    split.check()
    return split
