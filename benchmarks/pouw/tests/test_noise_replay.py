"""`noise_replay.py` against the scalar references it stands for: its forming at δ = 1 is `pearl_c.form_v1`, δ scales σ
and the headroom as stated, and its drawn cuts are the scalar cut flags along each word's chain at every G, each
attributed to exactly one cause."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
import noise_replay as NR
from verity.circuits.silicon.term import FP32_SIGNIFICAND_WIDTH, fp32_term
from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.pouw.prover.schemes import pearl_kw as P
from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import H100

K = 256


def _rows(rng, r: int) -> np.ndarray:
    x = rng.standard_normal((r, K)) * rng.uniform(0.01, 3.0, (r, 1))
    x[:, rng.integers(0, K, 3)] *= 40.0
    return x.astype(np.float32).view(np.uint32)


def _codes(rng, rows: int) -> np.ndarray:
    """E4M3 codes with zeros, subnormals and wide exponents in one atom, so that the adder cuts some products."""
    out = rng.integers(0, 0x7F, (rows, K)) | (rng.integers(0, 2, (rows, K)) << 7)
    out[rng.random((rows, K)) < 0.1] = 0
    small = rng.random((rows, K)) < 0.2
    out[small] = rng.integers(1, 16, small.sum())
    return out.astype(np.int64)


def _noise(rng, r: int) -> tuple[list[list[int]], np.ndarray]:
    e = [[int(c) for c in rng.integers(0, 0x7E, C.SLICE)] for _ in range(r)]
    return e, rng.integers(0, 0x7E, (K, C.SLICE)).astype(np.int64)


def _code_term(c: int) -> tuple[bool, int, int]:
    e = (c >> 3) & 0xF
    return bool(c & 0x80), max(e, 1), (c & 7) | (8 if e else 0)


def _cut_flags(acc: int, a, b, atom) -> list[bool]:
    """The adder's cut flags of one atom step from accumulator word `acc` (rev1's `Mend.cuts`), on the group sum."""
    rescale = atom.width - FP32_SIGNIFICAND_WIDTH
    value = None if (acc >> 23) & 0xFF == 0xFF else fp32_term(acc)
    prods = []
    for x, y in zip(a, b):
        (sa, ea, ma), (sb, eb, mb) = _code_term(int(x)), _code_term(int(y))
        prods.append((sa ^ sb, ea + eb - 14, (ma * mb) << 17))
    top = atom.zero_exponent
    for _, e, mag in prods:
        if mag and e > top:
            top = e
    if value is not None and value.magnitude and value.exponent > top:
        top = value.exponent
    out = []
    for _, e, mag in prods:
        scaled = mag << rescale if rescale >= 0 else mag >> -rescale
        out.append((scaled >> max(top - e, 0)) == 0)
    return out


def _chain_cuts(a, b, device) -> int:
    word, cuts = 0, 0
    for t, s in enumerate(range(0, len(a), device.atom.k)):
        if t % device.g == 0:
            word = 0
        cuts += sum(_cut_flags(word, a[s:s + device.atom.k], b[s:s + device.atom.k], device.atom))
        word = device.atom.step(word, [int(v) for v in a[s:s + device.atom.k]], [int(v) for v in b[s:s + device.atom.k]])
    return cuts


def test_delta_one_is_form_v1():
    rng = np.random.default_rng(1)
    x = _rows(rng, 3)
    e, f = _noise(rng, 3)
    stats = [C.s5_stats(row.tolist())[:2] for row in x]
    got = NR.form(x, stats, e, f, 1, H100)
    want = C.form_v1([row.tolist() for row in x], e, f.tolist(), H100)
    assert got["codes"].tolist() == want.codes
    assert got["alpha"] == want.alpha and got["beta"] == want.beta


@pytest.mark.parametrize("delta", [2, 3, 4])
def test_delta_scales_sigma_and_headroom(delta):
    rng = np.random.default_rng(delta)
    x = _rows(rng, 2)
    e, f = _noise(rng, 2)
    stats = [C.s5_stats(row.tolist())[:2] for row in x]
    got, one = NR.form(x, stats, e, f, delta, H100), NR.form(x, stats, e, f, 1, H100)
    for (s, rho), al, be, al1, be1 in zip(stats, got["alpha"], got["beta"], one["alpha"], one["beta"]):
        s, rho = P.f32_of_bits(s), P.f32_of_bits(rho)
        assert P.f32_of_bits(al) == pytest.approx(448 / (s + 4 * delta * rho), rel=1e-6)
        ratio = (P.f32_of_bits(be) / P.f32_of_bits(al)) / (P.f32_of_bits(be1) / P.f32_of_bits(al1))
        assert ratio == pytest.approx(delta, rel=1e-6)


@pytest.mark.parametrize("g", [4, 2, 1])
def test_cuts_are_the_scalar_chain_cuts(g):
    rng = np.random.default_rng(10 + g)
    device = dataclasses.replace(H100, g=g)
    A, B = _codes(rng, 3), _codes(rng, 2)
    ref_a = (np.zeros_like(A), np.zeros_like(A), np.zeros(A.shape, bool), np.zeros_like(A))
    ref_b = (np.zeros_like(B), np.zeros_like(B), np.zeros(B.shape, bool), np.zeros_like(B))
    big_a, big_b = rng.random(A.shape) < 0.1, rng.random(B.shape) < 0.1
    counts, causes = NR.word_counts(A, B, ref_a, ref_b, big_a, big_b, device)
    want = [[_chain_cuts(a, b, device) for b in B] for a in A]
    assert counts["cut"].tolist() == want
    assert int(causes.sum()) == sum(map(sum, want))
    assert causes[0] == sum(int(((a == 0) | (a == 0x80) | (b == 0) | (b == 0x80)).sum()) for a in A for b in B)
