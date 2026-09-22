"""Laptop-sized end-to-end check of the v2 Ligero protocol on a toy circuit that exercises every code path (a
product, two lookups into small tables incl. one whose key is a linear form, the state chain, the claimed word).
Runs on the CPU in a few seconds; the checker-v2 system itself is exercised on the pod (``run.py gate``)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from verity_numerical.checker.ctx import Table  # noqa: E402
from verity_numerical.checker.params import REAL  # noqa: E402
from verity_numerical.gkr_export.circuit import CircuitCtx  # noqa: E402

from backends.direct.ligero.field import P  # noqa: E402
from backends.direct.ligero.v2 import fc  # noqa: E402
from backends.direct.ligero.v2.protocol import Config, prove, verify  # noqa: E402
from backends.direct.ligero.v2.system import system_from_ctx  # noqa: E402

STEPS = 4


def toy_system():
    """State (s, e, M, z); unit: y.s = x.s, y.e = x.e + a, y.M = x.M * a (product), y.z = x.z; lookups: SQ(a) = a^2 for
    a < 8, and DBL(key = y.e - x.e) = 2 key for key < 16; ovf column = 0; y16 = y.M mod 2^16 (as a hint column, bound
    by SPLIT lookup on y.M = y16 + 2^16 hi is skipped: toy uses y16 = y.M directly)."""
    ctx = CircuitCtx()
    idx = {}

    def col(name):
        c = ctx.column(name)
        idx[name] = len(ctx.cols) - 1
        return c

    xs = [col(f"x.{n}") for n in ("s", "e", "M", "z")]
    a = col("a")
    ys = [col(f"y.{n}") for n in ("s", "e", "M", "z")]
    y16 = col("y16")
    ovf = col("g0.ovf")
    ctx.eq("y.s", ys[0], xs[0])
    ctx.eq("y.e", ys[1], xs[1] + a)
    ctx.eq("y.M", ys[2], xs[2] * a + a)      # a product wire
    ctx.eq("y.z", ys[3], xs[3])
    ctx.eq("ovf0", ovf, 0 * a)
    sq = Table("SQ", 0, 8, ("sq",), lambda k: (k * k,))
    dbl = Table("DBL", 0, 70, ("dbl",), lambda k: (2 * k,))   # 70 entries > l: 5 multiplicity rows
    (asq,) = ctx.lookup("sq", sq, a)
    (d,) = ctx.lookup("dbl", dbl, ys[1] - xs[1])
    (asq2,) = ctx.lookup("sq", sq, ys[1] - xs[1])       # a second query into SQ (multiplicities accumulate)
    ctx.eq("sq_dbl", asq + d, a * a + 2 * a)
    ctx.eq("sq_twice", asq2, asq)
    ctx.eq("y16", y16, ys[2])
    idx["__ovf"] = [idx["g0.ovf"]]
    idx["__sgn"] = []
    from verity_numerical.gkr_export.v2 import derive

    sysm = system_from_ctx(REAL, ctx, [], [], idx)
    vals = [None] * len(ctx.cols)
    vals[0] = 1
    for n, v in zip(("s", "e", "M", "z"), (0, REAL.floor, 0, 1)):
        vals[idx[f"x.{n}"]] = vals[idx[f"y.{n}"]] = v
    vals[idx["a"]] = vals[idx["g0.ovf"]] = vals[idx["y16"]] = 0
    sysm.pad = derive(ctx, vals)
    return sysm


def toy_units(sysm, n_vus: int, seed: int = 0):
    from verity_numerical.gkr_export.v2 import derive

    rng = np.random.default_rng(seed)
    units, words = [], []
    for _ in range(n_vus):
        st = [0, REAL.floor, 0, 1]
        for i in range(STEPS):
            a = int(rng.integers(1, 8))
            nxt = [st[0], st[1] + a, st[2] * a + a, st[3]]
            vals = [None] * len(sysm.ctx.cols)
            vals[0] = 1
            for k, n in enumerate(("s", "e", "M", "z")):
                vals[sysm.col_idx[f"x.{n}"]] = st[k]
                vals[sysm.col_idx[f"y.{n}"]] = nxt[k]
            vals[sysm.col_idx["a"]] = a
            vals[sysm.col_idx["g0.ovf"]] = 0
            vals[sysm.col_idx["y16"]] = nxt[2]
            units.append(derive(sysm.ctx, vals))
            st = nxt
        words.append(st[2])
    return units, np.array(words, dtype=np.int64)


def test_fc_inverse():
    x = torch.randint(0, P, (50, 6), dtype=torch.int64)
    one = fc.fc_mul(x, fc.fc_inv(x))
    assert torch.equal(one, fc.fc_one((50,), x.device))


def test_toy_accepts_and_rejects():
    sysm = toy_system()
    cfg = Config(l=16, n=64, t=8, target_log2=-10.0, steps=STEPS, n_vus=3)
    units, words = toy_units(sysm, 3)
    dev = torch.device("cpu")
    proof = prove(sysm, cfg, units, words, dev, check=True)
    ok, msg = verify(sysm, cfg, words, proof, dev)
    assert ok, msg
    # wrong claimed word
    bad = words.copy()
    bad[0] ^= 1
    ok, msg = verify(sysm, cfg, bad, proof, dev)
    assert not ok
    # a mutated lookup output (the SQ output of unit 0 + 1) is rejected by the verifier
    row = sysm.col_idx.get("sq.sq") or sysm.ctx.cols.index("sq.sq")

    def mut(W1):
        W1[row, 0] = (W1[row, 0] + 1) % P
    proof2 = prove(sysm, cfg, units, words, dev, mutate=mut)
    ok, msg = verify(sysm, cfg, words, proof2, dev)
    assert not ok, msg
    # a broken chain: unit 1's incoming state != unit 0's outgoing state
    row = sysm.col_idx["x.e"]

    def mut2(W1):
        W1[row, 1] = (W1[row, 1] + 1) % P
    proof3 = prove(sysm, cfg, units, words, dev, mutate=mut2)
    ok, msg = verify(sysm, cfg, words, proof3, dev)
    assert not ok, msg
    # a multiplicity tampered with
    lay = sysm.layout(cfg.l)
    r0 = lay.mult_base["SQ"]

    def mut3(W1):
        W1[r0, 1] = (W1[r0, 1] + 1) % P
    proof4 = prove(sysm, cfg, units, words, dev, mutate=mut3)
    ok, msg = verify(sysm, cfg, words, proof4, dev)
    assert not ok, msg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
