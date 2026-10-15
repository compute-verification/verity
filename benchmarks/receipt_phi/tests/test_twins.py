"""`twins.py`, the CPU statement of `phi.cu`'s candidates that `gate.py` holds the device to, checked for the structure
the attacks rely on: keys as `set_keys` makes them, every candidate's absorb defined, and mad's chain the affine form
over Z/2^32 that makes it fall."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import twins  # noqa: E402

K = twins.keys(0xC0FFEE11)


def test_keys_are_set_keys() -> None:
    assert all(a & 1 for a in K.ka) and all(b & 1 for b in K.kb)
    assert len(set(K.ka)) == twins.NKEY and len(set(K.kk)) == twins.NKEY
    assert set(twins.KINDS) == set(twins.OPS)


@pytest.mark.parametrize("kind", twins.KINDS)
def test_absorb_is_a_bijection_of_the_last_word(kind: str) -> None:
    """φ(y, ·) is injective in the last word absorbed, sampled: two different last words never meet (chi aside)."""
    rng = random.Random(1)
    n = 2 if kind in twins.PAIRED else 1
    for _ in range(200):
        st = (rng.getrandbits(32), rng.getrandbits(32))
        p = [rng.getrandbits(32) for _ in range(n)]
        q = list(p)
        q[-1] ^= 1 << rng.randrange(32)
        a, b = twins.absorb(kind, st, p, 0, K), twins.absorb(kind, st, q, 0, K)
        assert twins.finish(kind, a) != twins.finish(kind, b) or kind == "chi"


def test_mad_is_affine() -> None:
    rng = random.Random(2)
    y0, p = rng.getrandbits(32), [rng.getrandbits(32) for _ in range(16)]
    y, _ = twins.absorb("mad", (y0, 0), p, 0, K)
    want, w = 0, 1
    for i in reversed(range(16)):
        want = (want + p[i] * w) & twins.M32
        w = (w * K.ka[i]) & twins.M32
    assert y == (want + y0 * w) & twins.M32


@pytest.mark.parametrize("kind", twins.KINDS)
def test_pure_chains_shape(kind: str) -> None:
    a = twins.pure_chains(kind, 2, 0x5EED1234, 0xC0FFEE11, 3)
    assert len(a) == twins.PU and a == twins.pure_chains(kind, 2, 0x5EED1234, 0xC0FFEE11, 3)
    assert a != twins.pure_chains(kind, 1, 0x5EED1234, 0xC0FFEE11, 3)
