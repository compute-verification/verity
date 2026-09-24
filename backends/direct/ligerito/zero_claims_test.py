"""The V1 zero claims (``proof.zero_blocks`` / ``zero_points``) on sparse ``w``: numpy only, the same checks as ligerito-verify's
``virtual_rows_get_zero_claims`` / ``zero_claim_columns_must_be_independent``.  The fp8-ada shape: ``m = 3577``, 200 virtual rows
(const 3577, link, start, end, y16 3581, 192 pub:*, next:0..2 = 3774..3776), ``n_i = 12``, ``n_c = 10`` here."""
from __future__ import annotations

import numpy as np

from .proof import aligned_cover, zero_blocks, zero_points
from .proto import ext

P = ext.P
N_I, N_C, M = 12, 10, 3577
VIRT = {"const": M, "link": M + 1, "start": M + 2, "end": M + 3, "y16": M + 4,
        **{f"pub:{k}": M + 5 + k for k in range(192)}, **{f"next:{x}": M + 197 + x for x in range(3)}}


def _rand_ext(rng, shape) -> np.ndarray:
    return rng.integers(0, P, size=tuple(shape) + (6,), dtype=np.int64)


def _eq_at(pt: np.ndarray, idx: int) -> np.ndarray:
    acc = ext.one()
    for t in range(pt.shape[0]):
        acc = ext.mul(acc, pt[t] if (idx >> t) & 1 else ext.sub(ext.one(), pt[t]))
    return acc


def _eval_sparse(w, pt: np.ndarray) -> np.ndarray:
    """``w~(pt)`` for ``w`` = [(row, col, base value)], ``pt`` = (row bits || column bits)."""
    acc = ext.zero()
    for r, c, v in w:
        acc = ext.add(acc, ext.mul(ext.mul(ext.from_base(v % P), _eq_at(pt[:N_I], r)), _eq_at(pt[N_I:], c)))
    return acc


def _first_point(seed=5) -> np.ndarray:
    return _rand_ext(np.random.default_rng(seed), (N_I + N_C,))


def test_cover_matches_rust():
    assert aligned_cover(3577, 3777) == [(0, 3577), (1, 1789), (2, 895), (7, 28), (6, 58), (0, 3776)]
    assert aligned_cover(0, 8) == [(3, 0)]
    assert aligned_cover(5, 5) == []
    assert zero_blocks(VIRT) == aligned_cover(3577, 3777)
    split = dict(VIRT, y16=M + 300)                                   # two runs: each covered on its own, in row order
    assert zero_blocks(split) == aligned_cover(M, M + 4) + aligned_cover(M + 5, M + 200) + [(0, M + 300)]


def test_every_virtual_row_class_hits_its_block():
    blocks = zero_blocks(VIRT)
    pts = zero_points(N_I, _first_point(), blocks)
    first = _first_point()
    for p, (b, pfx) in zip(pts, blocks):
        assert np.array_equal(p[N_I:], first[N_I:]) and np.array_equal(p[:b], first[:b])
        assert np.array_equal(p[b:N_I, 0], [(pfx >> t) & 1 for t in range(N_I - b)]) and not p[b:N_I, 1:].any()
    for name, row in VIRT.items():
        w = [(row, 7, 1)]
        hit = [(b, pfx) for p, (b, pfx) in zip(pts, blocks) if _eval_sparse(w, p).any()]
        assert hit == [(b, pfx) for b, pfx in blocks if row >> b == pfx], name
    # the committed rows and the rows above the virtual block never enter a zero claim
    for row in (0, M - 1, M + 200, (1 << N_I) - 1):
        assert all(not _eval_sparse([(row, 3, 9)], p).any() for p in pts)


def test_zero_claim_columns_must_be_independent():
    """+d at (3584, col 64) and -d at (3648, col 0) (row bit 6 <-> column bit 6) vanish on every (rho[:7] || bits || rho[:n_c])
    but not on (rho[:7] || bits || r_c)."""
    blocks = zero_blocks(VIRT)
    first = _first_point()
    k = blocks.index((7, 28))
    w = [(3584, 64, 12345), (3648, 0, -12345)]
    assert _eval_sparse(w, zero_points(N_I, first, blocks)[k]).any()
    rho = first[:N_I]
    diag = np.concatenate((rho, rho[:N_C]))
    assert not _eval_sparse(w, zero_points(N_I, diag, blocks)[k]).any()
