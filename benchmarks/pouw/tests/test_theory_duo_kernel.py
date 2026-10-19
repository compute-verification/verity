"""duo's all-state kernel at small widths (theory/duo_kernel.c): no three-word kernel for any odd keys, the four-word
family T4 exactly where its key conditions hold, and its 32-bit instance on receipt_folds' duo_word."""
from __future__ import annotations

import itertools
import random
import re
import sys
from pathlib import Path

import pytest

THEORY = Path(__file__).resolve().parents[1] / "theory"
sys.path.insert(0, str(THEORY))
sys.path.insert(0, str(THEORY.parent / "receipts"))

import duo_kernel_sweep as dk
import receipt_folds as rf


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return dk.build(tmp_path_factory.mktemp("duo_kernel"))


def _hit_keys(row: dict) -> set[tuple[int, ...]]:
    return {tuple(map(int, re.match(r"hit a=([\d,]+) ", line).group(1).split(","))) for line in row["hit_lines"]}


@pytest.mark.parametrize("n", [4, 6])
def test_no_three_word_kernel(exe, n):
    """Three words, every odd multiplier at each position and every nonzero rotation: no two runs leave the same
    final state on every start state."""
    rows = [dk.invoke(exe, *job) for job in dk.generic(n, 3, "full")]
    assert len(rows) == (n - 1) ** 2 and sum(r["configs"] for r in rows) == (n - 1) ** 2 * (1 << (n - 1)) ** 2
    assert not any(r["hit_configs"] for r in rows)


def test_four_word_kernel_is_t4(exe):
    """n = 4, four words: the kernel is non-empty exactly at r1 + r3 ≡ 0 (mod 4), a2 ≡ -1 (mod 2^r1) and
    a3 ≡ ±1 (mod 2^(5 - r1)), and every element flips only top bits."""
    n, odd = 4, range(1, 16, 2)
    for r1, r2, r3 in itertools.product(range(1, n), repeat=3):
        row = dk.invoke(exe, n, 4, "full", (0, 1, 2, 3), (r1, r2, r3, 1))
        want = set()
        if (r1 + r3) % n == 0:
            want = {(a1, a2, a3, 1) for a1 in odd for a2 in odd for a3 in odd
                    if (a2 + 1) % (1 << r1) == 0 and a3 % (1 << (n + 1 - r1)) in (1, (1 << (n + 1 - r1)) - 1)}
        assert _hit_keys(row) == want and row["nontop"] == 0, (r1, r2, r3)


def test_t4_at_32_bits():
    """r1 = r3 = 16, a2 = 2^32 - 1, a3 = 1: flipping the top bit of four words, with bit 15 of the third word's XOR
    set, leaves (y, z) unchanged on every start state tried; with that bit clear it doesn't."""
    t, rng = 1 << 31, random.Random(11)
    keys = [(rng.getrandbits(32), rng.getrandbits(32) | 1, 7), (rng.getrandbits(32), 0xFFFFFFFF, rng.randrange(1, 32)),
            (rng.getrandbits(32), 1, 16), (rng.getrandbits(32), rng.getrandbits(32) | 1, 3)]
    keys[0] = keys[0][:2] + (16,)

    def run(y, z, ds):
        for d, (k, a, r) in zip(ds, keys):
            y, z = rf.duo_word(y, z, d, k, a, r)
        return y, z

    for x, equal in ((rng.getrandbits(32) | 1 << 15, True), (rng.getrandbits(32) & ~(1 << 15), False)):
        w = rng.getrandbits(32)
        d = (keys[0][0], keys[1][0] ^ t, keys[2][0] ^ x, w)
        e = (keys[0][0] ^ t, keys[1][0], keys[2][0] ^ x ^ t, w ^ t)
        states = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(2000)]
        assert all(run(y, z, d) == run(y, z, e) for y, z in states) is equal


def test_rotation_zero_is_caught(exe):
    """With the first rotation 0 mod n, two words' top flips cancel: the enumerator finds that kernel."""
    row = dk.invoke(exe, 4, 2, "full", (0, 1), (4, 1), keys=(3, 5))
    assert row["hits"] == 1 and row["nontop"] == 0
    assert not dk.invoke(exe, 4, 2, "full", (0, 1), (1, 1))["hit_configs"]


@pytest.mark.parametrize("mode", ["proj", "proj-K"])
def test_served_receipt_one_and_two_words(exe, mode):
    """The served receipt y ^ rotl(z, n/2) at n = 6: one and two words, every odd key and rotation, have no kernel,
    exactly or up to one constant."""
    rows = [dk.invoke(exe, *job) for job in dk.generic(6, 1, mode) + dk.generic(6, 2, mode)]
    assert len(rows) == 5 + 25 and not any(r["hit_configs"] for r in rows)
