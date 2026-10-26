"""ACCEL v3.1's CPU reference (benchmarks/pouw/accel_v31/v31.py): the binding identity, the row layout, and the stub
challenger's verdicts on an honest call and on calls that skip a noise tile or forge committed values."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import v31  # noqa: E402


def _call(sh: v31.Shape, kind: str = "llm", seed: int = 3):
    A, B = v31.operands(sh, np.random.default_rng(seed), kind)
    seeds = {s: bytes([seed, i]) * 32 for i, s in enumerate(v31.STREAMS)}
    c = v31.Coins.expand(sh, seeds)
    return A, B, seeds, c, v31.form(sh, A, B, c)


@pytest.mark.parametrize("kind", ["llm", "uniform", "zero"])
@pytest.mark.parametrize("lp,t,k", [(256, 3, 1536), (192, 3, 1536), (256, 3, 1280), (512, 2, 2048)])
def test_binding_identity_gives_o(kind, lp, t, k):
    sh = v31.Shape(128, 256, k, lp, t)
    _, _, _, c, f = _call(sh, kind)
    v = v31.values(sh, f)
    assert (v31.binding(sh, f, c, v.T_full) == v.O).all()
    assert (f.X + 256 * f.wX == f.Ad + v31.v_full(sh, c)).all() and np.abs(f.X).max() <= 128


def test_coins_have_their_ranges():
    sh = v31.Shape(128, 128, 768)
    _, _, _, c, _ = _call(sh)
    assert c.U.min() == -80 and c.U.max() == 80 and c.V.min() == -80 and c.V.max() == 80
    assert set(np.unique(c.dA)) == {0, 1}
    assert all(sorted(p) == list(range(sh.lp)) for p in c.perm) and len({tuple(p) for p in c.perm}) == len(c.perm)


def test_rows_hold_each_cells_words_where_word_leaf_says():
    sh = v31.Shape(256, 384, 1536, row=2048)
    _, _, _, _, f = _call(sh)
    v = v31.values(sh, f)
    words, obuf = v31.words_buffer(sh, v.T), v31.o_buffer(v.O)
    assert len(words) == sh.tiles * sh.S * 32768 and len(obuf) == 4 * sh.m * sh.n
    rng = np.random.default_rng(0)
    for _ in range(200):
        i, j, s = int(rng.integers(sh.m)), int(rng.integers(sh.n)), int(rng.integers(1, sh.S + 1))
        p, o = sh.word_leaf(i, j, s)
        assert int.from_bytes(v31.row_bytes(sh, words, obuf, p)[o:o + 2], "little") == v.T[s - 1, i, j]
        p, o = sh.o_leaf(i, j)
        assert int.from_bytes(v31.row_bytes(sh, words, obuf, p)[o:o + 4], "little") == v.O[i, j]
    assert all(len(v31.row_bytes(sh, words, obuf, p)) == sh.row for p in range(sh.leaves))


@pytest.mark.parametrize("cheat", [None, "skip:4", "forge:1:0.5", "forge:2:0.5", "forgeO:0.5"])
def test_verdicts(cheat):
    sh = v31.Shape(256, 256, 1536)
    A, B, _, _, _ = _call(sh)
    ch = v31.StubChallenger(b"\x07" * 64)
    seeds = ch.seeds(0, v31.operand_roots(A, B))
    prover, _ = v31.prove(sh, A, B, seeds, b"\x09" * 64, 0, cheat)
    ok, fails, line = v31.verdict(sh, ch, 0, prover, 48)
    assert ok == (cheat is None), line
    assert line.startswith("ACCEPT" if cheat is None else "REFUSE")


def test_challenger_keeps_the_order_and_never_redraws():
    sh = v31.Shape(128, 128, 768)
    ch = v31.StubChallenger(os.urandom(64))
    with pytest.raises(RuntimeError):
        ch.check(0, b"r" * 64, sh, 4)
    ch.seeds(0, b"o" * 128)
    with pytest.raises(RuntimeError):
        ch.seeds(0, b"o" * 128)
    ch.check(0, b"r" * 64, sh, 4)
    with pytest.raises(RuntimeError):
        ch.check(0, b"r" * 64, sh, 4)
