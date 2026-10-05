"""The three LogUp provers of the A GPU candidate produce ONE transcript on real-instance-shaped leaves (CUDA only).

Regression for lane a-kernel-a100: on ``vy-a100b`` the a-gpu Triton kernels (``gpu/kernels.py``: ``ext_mul``,
``combine_level``, ...) were served from a stale ``~/.triton/cache`` compiled under the previous field constant
``W = 31``; the 63 ``packed/tests`` never exercise those kernels, so nothing caught it.  This test runs, on leaves
shaped like the vector-valued tables (POW / ALIGN / LEAD / NORM: ``p in {1, -m, 0}``, ``q`` full-range extension
elements including values above ``p - 2^27``) and like R16 (identity table, ``Nw`` not a power of two):

* the torch reference with every kernel disabled,
* ``logup.prove_table`` with the a-gpu kernels enabled (``VERITY_GPU_NO_PACKED=1`` behaviour),
* ``logup_packed.prove_ext_table`` / ``prove_range_table`` (eager) and their CUDA-graph variants,

and asserts the message lists and the leaf claims are identical, plus a direct check of the a-gpu ``ext_mul`` and
``combine_level`` kernels on int64 AND int32 inputs (the two argument specialisations are cached separately).
"""
from __future__ import annotations

import copy
import random

import pytest
import torch

from gpu import field as _field
from gpu import logup, logup_packed
from gpu.field import P
from gpu.transcript import Transcript

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not logup_packed.AVAILABLE, reason="CUDA + packed kernels")
DEV = torch.device("cuda") if torch.cuda.is_available() else None
_KERNEL_SLOTS = ("_fast_combine_level", "_fast_eq_outer", "_fast_ext_mul", "_fast_fold")


class _no_kernels:
    def __enter__(self):
        self.saved = {k: getattr(_field, k) for k in _KERNEL_SLOTS}
        self.saved_round = logup._fast_logup_round
        for k in _KERNEL_SLOTS:
            setattr(_field, k, None)
        logup._fast_logup_round = None

    def __exit__(self, *a):
        for k, v in self.saved.items():
            setattr(_field, k, v)
        logup._fast_logup_round = self.saved_round


def _run(fn, *args):
    tr = Transcript()
    tr.absorb(b"a-kernel-a100 regression")
    out: list = []
    lc = fn(*args, tr, out)
    return out, (lc.rho, lc.lam, lc.value)


def _ext_leaves(n: int, n_wit: int, nrows: int, seed: int):
    """Vector-valued table leaves: witness rows point at random table rows; ``q`` = random full-range elements."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    qt = torch.randint(0, P, (nrows, 6), generator=g, dtype=torch.int64)
    qt[: nrows // 8] = torch.randint(P - (1 << 27), P, (nrows // 8, 6), generator=g, dtype=torch.int64)   # the top band
    hits = [rng.randrange(nrows) for _ in range(n_wit)]
    m = torch.bincount(torch.tensor(hits), minlength=nrows)
    p = torch.zeros((1 << n,), dtype=torch.int64)
    q = torch.zeros((1 << n, 6), dtype=torch.int64)
    q[:, 0] = 1
    p[:n_wit] = 1
    q[:n_wit] = qt[torch.tensor(hits)]
    p[n_wit:n_wit + nrows] = (-m) % P
    q[n_wit:n_wit + nrows] = qt
    return p.to(torch.int32).to(DEV), q.to(torch.int32).to(DEV)


@pytest.mark.parametrize("n,n_wit,nrows,seed", [(8, 192, 24, 1), (10, 192, 640, 2), (11, 1536, 400, 3)])
def test_ext_table_paths_agree(n, n_wit, nrows, seed):
    p_leaf, q_leaf = _ext_leaves(n, n_wit, nrows, seed)
    with _no_kernels():
        ref = _run(logup.prove_table, p_leaf, q_leaf, n)
    assert _run(logup.prove_table, p_leaf, q_leaf, n) == ref, "torch path with the a-gpu kernels differs from the reference"
    assert _run(logup_packed.prove_ext_table, p_leaf, q_leaf, n) == ref, "packed eager path differs"
    if logup_packed.GRAPHS:
        assert _run(logup_packed.prove_ext_table_graphed, p_leaf, q_leaf, n) == ref, "packed graphed path differs"


@pytest.mark.parametrize("Nw,seed", [(1000, 1), (56356, 2)])
def test_range_table_paths_agree(Nw, seed):
    Nt = 1 << 16
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, Nt, (Nw,), generator=g, dtype=torch.int64)
    tab = torch.arange(Nt, dtype=torch.int64)
    mult = torch.bincount(w, minlength=Nt)
    n = logup_packed._lg.lref.log2ceil(Nw + Nt)
    z = tuple(random.Random(seed).randrange(P) for _ in range(6))
    # leaves as build_leaves lays them out: [queries | table | pad]
    p = torch.zeros((1 << n,), dtype=torch.int64)
    q = torch.zeros((1 << n, 6), dtype=torch.int64)
    q[:, 0] = 1
    zt = torch.tensor(z, dtype=torch.int64)
    p[:Nw] = 1
    q[:Nw] = zt
    q[:Nw, 0] = (zt[0] - w) % P
    p[Nw:Nw + Nt] = (-mult) % P
    q[Nw:Nw + Nt] = zt
    q[Nw:Nw + Nt, 0] = (zt[0] - tab) % P
    p_leaf, q_leaf = p.to(torch.int32).to(DEV), q.to(torch.int32).to(DEV)
    with _no_kernels():
        ref = _run(logup.prove_table, p_leaf, q_leaf, n)
    w32, tab32 = w.to(torch.int32).to(DEV), tab.to(torch.int32).to(DEV)
    assert _run(logup_packed.prove_range_table, w32, tab32, mult.to(DEV), z, n) == ref
    if logup_packed.GRAPHS:
        assert _run(logup_packed.prove_range_table_graphed, w32, tab32, z, n) == ref


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_agpu_ext_mul_and_combine_level_match_torch(dtype):
    if _field._fast_ext_mul is None:
        pytest.skip("a-gpu kernels disabled (VERITY_GPU_TORCH_ONLY)")
    g = torch.Generator().manual_seed(7)
    n = 1 << 12
    a = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64).to(DEV)
    b = torch.randint(0, P, (n, 6), generator=g, dtype=torch.int64).to(DEV)
    with _no_kernels():
        want = _field.ext_mul(a, b)
    got = _field._fast_ext_mul(a.to(dtype), b.to(dtype))
    assert torch.equal(got.to(torch.int64) % P, want), "a-gpu ext_mul kernel != torch (stale Triton cache / wrong W?)"
    p = torch.randint(0, P, (n,), generator=g, dtype=torch.int64).to(DEV).to(dtype)
    with _no_kernels():
        wp, wq = logup._combine_level(p.to(torch.int32), a.to(torch.int32))
    gp, gq = _field._fast_combine_level(p.to(torch.int32), a.to(torch.int32))
    assert torch.equal(gp, wp) and torch.equal(gq, wq)
