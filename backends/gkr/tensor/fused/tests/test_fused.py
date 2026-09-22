"""The fused kernels == the pure-Python BabyBear^6 reference, bit for bit.

CPU (laptop, < 5 s): the Montgomery scheme the kernels implement (``field.py``) reproduces ``reference.py`` on random
elements and on whole rounds.  Pod (``TC_DEVICE=cuda``): the Triton kernels (SIMT and tensor-core) against the reference
on t <= 4096, all modes (first round base, base -> extension fold, extension rounds, fold-only), odd block tails.
"""
from __future__ import annotations

import os
import random

import pytest
from tensor import reference as ref
from tensor.fused import field

CUDA = bool(os.environ.get("TC_DEVICE")) and __import__("importlib").util.find_spec("torch") is not None
if CUDA:
    import torch
    CUDA = torch.cuda.is_available()


def emulated_round(f, g, h, r_prev, deg_in):
    """The fused kernel's arithmetic in Python: fold with ``r_prev`` (if given), then S over pairs of folded elements."""
    if r_prev is not None:
        rm = field.to_mont(r_prev)
        if deg_in == 1:
            fold = lambda tb: [field.fold_pair_base(tb[2 * k][0], tb[2 * k + 1][0], rm) for k in range(len(tb) // 2)]
        else:
            fold = lambda tb: [field.fold_pair(tb[2 * k], tb[2 * k + 1], rm) for k in range(len(tb) // 2)]
        f, g, h = fold(f), fold(g), fold(h)
    sums = [[0] * 6 for _ in range(4)]
    for k in range(len(f) // 2):
        fx, gx, hx = (field.points(tb[2 * k], tb[2 * k + 1]) for tb in (f, g, h))
        for x in range(4):
            u = field.ext_mmul(field.ext_mmul(fx[x], gx[x]), hx[x])
            for c in range(6):
                sums[x][c] += u[c]
    return [tuple(field.fix_s(v) for v in row) for row in sums], f, g, h


def test_constants():
    assert field.MU == 0x88000001 and field.P == 0x78000001 and field.MU_I32 < 0
    assert (field.P * field.MU) % 2**32 == 1


def test_mmul_is_montgomery_product():
    rng = random.Random(1)
    for _ in range(2000):
        a, b = rng.randrange(field.P), rng.randrange(field.P)
        assert field.mmul(a * field.R_MOD_P % field.P, b) == a * b % field.P
        assert 0 <= field.mmul(a, b) < field.P
    for a in (0, 1, field.P - 1):
        for b in (0, 1, field.P - 1):
            assert field.mmul(a * field.R_MOD_P % field.P, b) == a * b % field.P


def test_ext_mmul_matches_reference():
    rng = random.Random(2)
    for _ in range(300):
        a, b = ref.random_ext(rng), ref.random_ext(rng)
        assert field.ext_mmul(field.to_mont(a), b) == ref.mul(a, b)
        assert field.fold_pair(a, b, field.to_mont(ref.random_ext(rng))) is not None


@pytest.mark.parametrize("t,deg_in,with_fold", [(8, 1, False), (8, 1, True), (8, 6, True), (64, 6, True), (64, 1, True)])
def test_emulated_round_matches_reference(t, deg_in, with_fold):
    rng = random.Random(10 * t + deg_in)
    draw = ref.random_base if deg_in == 1 else ref.random_ext
    f, g, h = ([draw(rng) for _ in range(t)] for _ in range(3))
    r_prev = ref.random_ext(rng) if with_fold else None
    if with_fold:
        f1, g1, h1 = ref.fold(f, r_prev), ref.fold(g, r_prev), ref.fold(h, r_prev)
    else:
        f1, g1, h1 = f, g, h
    S_ref = ref.round_polynomial(f1, g1, h1)
    S, f2, g2, h2 = emulated_round(f, g, h, r_prev, deg_in)
    assert S == S_ref
    assert (f2, g2, h2) == (f1, g1, h1)


# ---------------------------------------------------------------------------------------------------------------------
# pod: the Triton kernels
# ---------------------------------------------------------------------------------------------------------------------


def _kernels():
    from tensor.fused import round_simt
    dev = os.environ["TC_DEVICE"]
    ks = [("simt", round_simt.FusedRound(dev, block=8, num_warps=1)), ("simt", round_simt.FusedRound(dev, block=256, num_warps=4))]
    if os.environ.get("TC_SKIP_TC") != "1":
        from tensor.fused import round_tc
        ks += [("tc", round_tc.FusedRoundTC(dev, block=16, num_warps=1)), ("tc", round_tc.FusedRoundTC(dev, block=32, num_warps=2)),
               ("tc", round_tc.FusedRoundTC(dev, block=64, num_warps=4, reload=False))]
    return ks


@pytest.mark.skipif(not CUDA, reason="CUDA only")
@pytest.mark.parametrize("t,deg_in,mode", [(8, 1, "first"), (8, 1, "fold+eval"), (8, 6, "fold+eval"), (8, 6, "first"), (8, 6, "fold"),
                                            (200, 6, "fold+eval"), (200, 1, "fold+eval"), (4096, 6, "fold+eval"), (4096, 6, "first"), (4096, 1, "first")])
def test_kernels_match_reference(t, deg_in, mode):
    from tensor.fused import round_simt
    rng = random.Random(100 * t + deg_in + len(mode))
    draw = ref.random_base if deg_in == 1 else ref.random_ext
    f, g, h = ([draw(rng) for _ in range(t)] for _ in range(3))
    r_prev = ref.random_ext(rng) if "fold" in mode else None
    if r_prev is not None:
        f1, g1, h1 = ref.fold(f, r_prev), ref.fold(g, r_prev), ref.fold(h, r_prev)
    else:
        f1, g1, h1 = f, g, h
    S_ref = ref.round_polynomial(f1, g1, h1) if "eval" in mode or mode == "first" else None
    dev = os.environ["TC_DEVICE"]
    tf, tg, th = (round_simt.FusedTable.from_ints(x, deg_in, dev) for x in (f, g, h))
    for name, kern in _kernels():
        if name == "tc" and (deg_in == 1 or mode == "first"):
            continue                                                       # the TC kernel folds extension tables only
        S, outs = kern.round(tf, tg, th, r_prev, eval=("eval" in mode or mode == "first"))
        if S_ref is not None:
            assert S == S_ref, (name, kern.block)
        if r_prev is not None:
            assert [o.to_ints() for o in outs] == [f1, g1, h1], (name, kern.block)


@pytest.mark.skipif(not CUDA, reason="CUDA only")
def test_layer_sequence_simt():
    """A whole small layer: first round (base), base -> ext fold + eval, ext rounds down to 2, final fold."""
    from tensor.fused import round_simt
    rng = random.Random(77)
    t = 256
    dev = os.environ["TC_DEVICE"]
    f, g, h = ([ref.random_base(rng) for _ in range(t)] for _ in range(3))
    kern = round_simt.FusedRound(dev, block=32, num_warps=2)
    tf, tg, th = (round_simt.FusedTable.from_ints(x, 1, dev) for x in (f, g, h))
    S, _ = kern.round(tf, tg, th, None)
    assert S == ref.round_polynomial(f, g, h)
    rf, rg, rh = f, g, h
    while len(rf) > 2:
        r = ref.random_ext(rng)
        rf, rg, rh = ref.fold(rf, r), ref.fold(rg, r), ref.fold(rh, r)
        S, (tf, tg, th) = kern.round(tf, tg, th, r)
        assert S == ref.round_polynomial(rf, rg, rh)
        assert tf.to_ints() == rf
    r = ref.random_ext(rng)
    _, (tf, _, _) = kern.round(tf, tg, th, r, eval=False)
    assert tf.to_ints() == ref.fold(rf, r)


@pytest.mark.skipif(not CUDA, reason="CUDA only")
@pytest.mark.parametrize("t,mode", [(8, "first"), (8, "fold+eval"), (8, "fold"), (200, "fold+eval"), (4096, "fold+eval"), (4096, "first")])
def test_logup_kernel_matches_reference(t, mode):
    from tensor.fused import round_logup, round_simt
    rng = random.Random(500 + t + len(mode))
    tabs = [[ref.random_ext(rng) for _ in range(t)] for _ in range(5)]
    lam = ref.random_ext(rng)
    r_prev = ref.random_ext(rng) if "fold" in mode else None
    S_ref, folded_ref = round_logup.logup_reference(*tabs, lam, r_prev)
    dev = os.environ["TC_DEVICE"]
    dt = [round_simt.FusedTable.from_ints(x, 6, dev) for x in tabs]
    for kern in (round_logup.LogUpRound(dev, block=8, num_warps=1), round_logup.LogUpRound(dev, block=256, num_warps=4)):
        S, outs = kern.round(dt, lam, r_prev, eval=(mode != "fold"))
        if mode != "fold":
            assert S == S_ref, kern.block
        if r_prev is not None:
            assert [o.to_ints() for o in outs] == folded_ref, kern.block
