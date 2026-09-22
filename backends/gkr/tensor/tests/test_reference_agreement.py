"""Kernel (CPU path) == pure-Python BabyBear^6 reference, bit for bit, on tiny rounds (t <= 64).

Runs on a laptop in well under 30 s with numpy alone; the torch-CPU path is exercised too when torch is installed.
The same test drives the CUDA path on the pod (``TC_DEVICE=cuda``).
"""
from __future__ import annotations

import os
import random

import pytest

from tensor import reference as ref
from tensor import sumcheck_int8 as k


def _backends():
    out = [k.Ops("numpy")]
    if k.torch is not None:
        out.append(k.Ops("torch", "cpu"))
        dev = os.environ.get("TC_DEVICE")
        if dev and k.torch.cuda.is_available():
            out.append(k.Ops("torch", dev))
    return out


def test_irreducible_modulus():
    assert pow(ref.W, (ref.P - 1) // 2, ref.P) != 1
    assert pow(ref.W, (ref.P - 1) // 3, ref.P) != 1


def test_reference_field_axioms():
    rng = random.Random(1)
    a, b, c = (ref.random_ext(rng) for _ in range(3))
    assert ref.mul(a, b) == ref.mul(b, a)
    assert ref.mul(ref.mul(a, b), c) == ref.mul(a, ref.mul(b, c))
    assert ref.mul(a, ref.add(b, c)) == ref.add(ref.mul(a, b), ref.mul(a, c))
    assert ref.mul(a, ref.ONE) == a


def test_limbs_roundtrip():
    # 4 balanced limbs reach 127 * (2^32 - 1) / 255 = 2139062143 > p - 1; 2^31 - 1 itself does not fit (not needed)
    for v in (0, 1, 255, 256, 2139062143, ref.P - 1, 2**30 + 12345):
        assert ref.from_limbs(ref.limbs_balanced(v)) == v
        assert all(-128 <= x < 128 for x in ref.limbs_balanced(v))


@pytest.mark.parametrize("t", [2, 6, 16, 64])
@pytest.mark.parametrize("deg", [1, 6])
@pytest.mark.parametrize("reduce_in_matrix", [True, False])
def test_round_agrees(t, deg, reduce_in_matrix):
    rng = random.Random(1000 * t + deg)
    draw = ref.random_base if deg == 1 else ref.random_ext
    f, g, h = ([draw(rng) for _ in range(t)] for _ in range(3))
    r = ref.random_ext(rng)
    S_ref, f_ref, g_ref, h_ref = ref.naive_round(f, g, h, r)
    for ops in _backends():
        kern = k.Kernel(ops, chunk=8, reduce_in_matrix=reduce_in_matrix)   # chunk 8 so t = 64 exercises INT64 chunk sums
        tf, tg, th = (kern.table_from_ints(x, deg) for x in (f, g, h))
        stats = k.Stats()
        S, f2, g2, h2 = kern.round_step(tf, tg, th, r, stats)
        got = [tuple(int(x) for x in row) for row in ops.to_numpy(S)]
        assert got == S_ref, (ops.backend, ops.device)
        assert kern.table_to_ints(f2) == f_ref
        assert kern.table_to_ints(g2) == g_ref
        assert kern.table_to_ints(h2) == h_ref
        assert stats.inner_products == 8 and stats.hadamard_products == 2 * t


def test_two_consecutive_rounds_base_then_extension():
    """Round 1 on base tables produces extension tables; round 2 consumes them (the first-fold transition)."""
    rng = random.Random(7)
    t = 32
    f, g, h = ([ref.random_base(rng) for _ in range(t)] for _ in range(3))
    r1, r2 = ref.random_ext(rng), ref.random_ext(rng)
    _, f1, g1, h1 = ref.naive_round(f, g, h, r1)
    S_ref, f_ref, _, _ = ref.naive_round(f1, g1, h1, r2)
    for ops in _backends():
        kern = k.Kernel(ops)
        tf, tg, th = (kern.table_from_ints(x, 1) for x in (f, g, h))
        _, tf1, tg1, th1 = kern.round_step(tf, tg, th, r1)
        assert tf1.deg == 6
        S, tf2, _, _ = kern.round_step(tf1, tg1, th1, r2)
        assert [tuple(int(x) for x in row) for row in ops.to_numpy(S)] == S_ref
        assert kern.table_to_ints(tf2) == f_ref


def test_scalar_matrix_is_multiplication():
    """``limbs(v) @ C_c`` recomposed equals ``c v`` for base and extension ``v`` (both column layouts)."""
    rng = random.Random(3)
    ops = k.Ops("numpy")
    kern = k.Kernel(ops)
    for deg in (1, 6):
        for reduce_in_matrix in (True, False):
            c = ref.random_ext(rng)
            v = ref.random_base(rng) if deg == 1 else ref.random_ext(rng)
            C = k.scalar_matrix(c, deg, reduce_in_matrix)
            tb = kern.table_from_ints([v], deg)
            limbs = k.decompose(ops, tb.coeffs, deg)
            z, _ = ops.mm_int8(limbs, ops.from_numpy(C))
            deg_out = 6 if reduce_in_matrix else 12
            coeffs = k.recompose(ops, z, deg_out, kern.pow2_row)[0].tolist()
            if not reduce_in_matrix:
                coeffs = [(coeffs[d] + ref.W * (coeffs[d + 6] if d + 6 < 12 else 0)) % ref.P for d in range(6)]
            assert tuple(coeffs) == ref.mul(c, v)
