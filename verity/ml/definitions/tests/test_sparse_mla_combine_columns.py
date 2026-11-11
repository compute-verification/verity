"""`SparseMlaCombine_v1` is columnwise: its weights read only the splits' lse, and each output column is its own ordered FMA
chain over the splits, so the combine at DV = G on columns [c, c + G) of every split's partial output returns columns
[c, c + G) of the combine at GLM-5.3's DV = 512. A block of G output columns is that Definition at a smaller DV: on
Gaussian rows, an empty split's -inf lse, a split whose lse leaves every other weight 0, partial outputs with NaN, ±inf,
-0 and subnormals, and uniformly random words."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.definitions.registry import sparse_mla_split as SM

NEG_INF, NAN, INF, NEG_ZERO, TINY = 0xFF800000, 0x7FC00001, 0x7F800000, 0x80000000, 0x00000001


def _flat(v) -> list[int]:
    return [w for u in v for w in _flat(u)] if isinstance(v, (tuple, list)) else [int(v)]


def _f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32).view(np.uint32)


def _rows(rng, NSPLIT: int, DV: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(kind, lse, o) of f32 words."""
    o = _f32(rng.standard_normal((NSPLIT, DV)))
    lse = _f32(rng.standard_normal(NSPLIT) * 4)
    empty, dominant, specials = lse.copy(), lse.copy(), o.copy()
    empty[NSPLIT // 2] = NEG_INF
    dominant[0] = _f32(np.array([200.0]))[0]
    for j, v in enumerate((NAN, INF, INF | 0x80000000, NEG_ZERO, TINY)):
        specials[j % NSPLIT, (37 * j + 5) % DV] = v
    return [("gauss", lse, o), ("an empty split's -inf lse", empty, o), ("one split takes every weight", dominant, o),
            ("NaN, inf, -0 and a subnormal in o", lse, specials),
            ("random words", rng.integers(0, 1 << 32, NSPLIT, dtype=np.uint64).astype(np.uint32),
             rng.integers(0, 1 << 32, (NSPLIT, DV), dtype=np.uint64).astype(np.uint32))]


@pytest.mark.parametrize("NSPLIT,G", [(5, 64), (16, 96), (28, 128), (1, 64)])
def test_a_combine_on_a_column_slice_is_that_slice_of_the_combine(NSPLIT, G):
    glm = SM.M.SPARSE_MLA_GLM53
    DV = glm["DV"]
    whole = bind(SM.SparseMlaCombine, NSPLIT=NSPLIT, DV=DV, INV=glm["INV"])
    part = bind(SM.SparseMlaCombine, NSPLIT=NSPLIT, DV=G, INV=glm["INV"])
    cuts = [(c, min(c + G, DV)) for c in range(0, DV, G)]
    tail = bind(SM.SparseMlaCombine, NSPLIT=NSPLIT, DV=cuts[-1][1] - cuts[-1][0], INV=glm["INV"])
    for kind, lse, o in _rows(np.random.default_rng(NSPLIT + G), NSPLIT, DV):
        want = _flat(evaluate(whole, lse.tolist(), o.ravel().tolist()))
        got = [v for c0, c1 in cuts
               for v in _flat(evaluate(part if c1 - c0 == G else tail, lse.tolist(), o[:, c0:c1].ravel().tolist()))]
        assert len(got) == DV and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])
