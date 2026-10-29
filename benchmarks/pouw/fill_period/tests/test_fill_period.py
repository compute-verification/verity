"""One filler period end to end on a local gate (`fill_period.run`): the auditor opens the fill period on the filler
circuit, the window's salt, call and draw coins are its session's live coins, and `fill.period_fill` credits the fill only
for the window's own value roots, fixed before the draw and accepted block by block."""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F
from verity.core.protocols.pouw.circuit import dither as D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fill_period as FP


def test_block_of_walks_every_block_once():
    """`fill.block_of` takes the draw's units onto every (call, cell, segment) of the window once, call-major."""
    sh = D.Shape(128, 256, 128, lp=32, L=64, o=False)
    got = [F.block_of(sh, u) for u in range(2 * F.blocks(sh))]
    assert len(set(got)) == len(got) == 2 * sh.m * sh.n * sh.S
    assert got[0] == (0, 0, 0, 1) and got[1] == (0, 0, 0, 2) and got[sh.S] == (0, 0, 1, 1)
    assert got[F.blocks(sh)] == (1, 0, 0, 1) and got[-1] == (1, sh.m - 1, sh.n - 1, sh.S)


def test_the_stand_in_accepts_honest_blocks_and_rejects_a_wrong_word():
    """The stand-in accepts a block of an honest call and rejects one whose committed T_s is off by one, though its rows
    still open at the (wrong) values' root."""
    rng = random.Random(7)
    sh = D.Shape(128, 128, 32, lp=32, L=32, o=False)
    B = [[rng.randrange(-128, 128) for _ in range(sh.k)] for _ in range(sh.n)]
    salt, coin, ks = rng.randbytes(64), rng.randbytes(64), rng.randbytes(64)
    seeds, A, tree = FP.call(sh, salt, coin, B, ks, 0)
    circuit = rng.randbytes(64)
    units = [0, 129, sh.m * sh.n - 1]
    assert [p["verdict"] for p in FP.stand_in(sh, units, [(seeds, A, tree)], B, circuit, salt, [coin])] == ["accept"] * 3
    T, _ = D.values(sh, seeds, A, B)
    T[0][1][1] = (T[0][1][1] + 1) & D.M16
    bad = D.commit(sh, D.values_buffer(sh, T), D.salts(ks))
    got = FP.stand_in(sh, units, [(seeds, A, bad)], B, circuit, salt, [coin])
    assert [p["verdict"] for p in got] == ["accept", "reject", "accept"]
    assert got[1]["program"] == F.statement(circuit, salt, coin, (0, 1, 1, 1)).hex()


@pytest.mark.slow
def test_a_filler_period_on_a_local_gate_is_credited_and_its_negatives_refused(tmp_path: Path, monkeypatch):
    """The filler window's coins come from its session (the stub logged their requests), its draw is the committed K's,
    every drawn block is accepted, and the fill is the window's m·n·K a call at that K; a value root changed after the
    draw coin and a drawn block the verifier rejects each refuse the whole fill.

    Slow (two calls' values in pure Python, about 30 s): it is the one test that runs the gate's period-open, reserve,
    open and period-close around a filler window, which the difftest's cases (`pouw_fill_cases`) build by hand."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = [FP.WS.a5_coins().on_main([])]
    g = FP.WS.local(tmp_path / "gate", main)
    monkeypatch.setattr(g.A, "serve_argv", g.A.serve_argv)
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    FP.stub(g, tmp_path / "coins.log")
    doc = FP.run(g, main, tmp_path / "out", "demo")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    assert doc["period"] == "demo" + F.SUFFIX and doc["period_open"][Cr.K_FIELD] == FP.DRAW_K
    assert len(doc["drawn"]) == FP.DRAW_K and doc["verdicts"] == ["accept"] * FP.DRAW_K
    sh = FP.SHAPE
    assert doc.get("fill") == {"work": [FP.CALLS * sh.m * sh.n * sh.k], "k": FP.DRAW_K, "delta": str(Cr.DELTA),
                               "eps": doc["fill"]["eps"]}, doc.get("refused")
    stale, rejected = doc["negatives"]["stale-root"], doc["negatives"]["block-rejected"]
    assert stale["refused"][0] == "fill" and ": draw: " in stale["refused"][1], stale
    assert rejected["refused"][0] == "fill" and ": failed: " in rejected["refused"][1], rejected
