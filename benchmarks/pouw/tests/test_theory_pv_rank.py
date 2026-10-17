"""pv's coin-row cross rank in a sequence's first block is (q + 1) + min(q + 1, r), not P6's (q + 1) + r."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import checked_strip as cs  # noqa: E402


def test_pv_coin_rank_in_the_first_block():
    """At (k, r, L) = (32, 32, 32) the coin rows have rank 2 at q = 0 and 64 at q = 31, at the zero and random
    instances, which P6's formula (33 and 64) gets right only at q = r − 1."""
    rng = np.random.default_rng(1)
    for q, want in ((0, (1, 2)), (12, (13, 26)), (31, (32, 64))):
        B = cs.PVBlock(1, 32, 1, 32, 32, q=[q])
        for kind in ("zero", "random"):
            assert cs.cross_ranks(B, *cs.instance(B, kind, rng), mod=1)[:3] == want + (0,)
