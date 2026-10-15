"""`attacks.py`'s machinery, held to `twins.py` and to the identities its findings rest on: its word steps are the
twins', its inverses invert, mad's sketch reproduces every honest receipt, xsm's top-bit canceller is exact, and the
mum lattice's patterns are integer relations of the keys."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attacks  # noqa: E402
import twins  # noqa: E402

K = twins.keys(0xC0FFEE11)
U = np.uint64


def sample(n: int = 64, seed: int = 7):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 1 << 32, size=n, dtype=np.uint64) for _ in range(4)]


@pytest.mark.parametrize("kind", twins.KINDS)
def test_steps_are_the_twins(kind: str) -> None:
    y, z, p0, p1 = sample()
    a, b = attacks.step(kind, y, z, p0, 4)
    a, b = attacks.step(kind, a, b, p1, 5)
    for i in range(y.size):
        st = (int(y[i]), int(z[i]))
        if kind in twins.PAIRED:
            want = twins.absorb(kind, st, [int(p0[i]), int(p1[i])], 4, K)
        else:
            want = twins.absorb(kind, twins.absorb(kind, st, [int(p0[i])], 4, K), [int(p1[i])], 5, K)
        assert want[0] == int(a[i]) and (kind != "mum" or want[1] == int(b[i]))


@pytest.mark.parametrize("kind", [k for k in twins.KINDS if k != "chi"])
@pytest.mark.parametrize("t", [4, 5])
def test_invert_inverts(kind: str, t: int) -> None:
    y, z, p, _ = sample()
    ty, tz = attacks.step(kind, y, z, p, t)
    q, ok = attacks.invert(kind, y, z, ty, tz, t)
    assert ok.all() and (q == p).all()


def test_mad_sketch_reproduces_every_receipt() -> None:
    r = attacks.mad_sketch(n=64, b=16, L=32)
    assert r["receipts_equal"] and r["blocks"] == 32


def test_ra_carry_free_blocks_are_the_linear_form() -> None:
    r = attacks.carries("int26", 64)
    assert r["blocks_with_none"] > 0 and r["of_carry_free_blocks"] == 1.0


def test_degree2_passes_mad_and_fails_mum() -> None:
    r = attacks.degree2("u32", 8)
    assert r["mad"]["add"]["exact"] == 1.0 and r["mum"]["add"]["exact"] == 0.0 and r["mum"]["xor"]["exact"] == 0.0


def test_xsm_top_bit_cancels() -> None:
    y, z, p0, p1 = sample()
    a, _ = attacks.step("xsm", *attacks.step("xsm", y, z, p0, 4), p1, 5)
    b, _ = attacks.step("xsm", *attacks.step("xsm", y, z, p0 ^ U(1 << 31), 4), p1 ^ U(0x80008000), 5)
    assert (a == b).all()


def test_mum_lattice_patterns_cancel() -> None:
    for m, r in attacks.mum_lattice().items():
        d, n = r["deltas"], len(r["deltas"])
        assert sum(x * int(attacks.KA[4 + i]) for i, x in enumerate(d)) % (1 << 64) == 0
        assert max(abs(x) for x in d) < 1 << 32 and m == f"{n}_words"
