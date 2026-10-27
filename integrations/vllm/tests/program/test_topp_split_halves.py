"""The split top-p reference is defined on every row (`sampling.topp_keep`): on a constructed row whose two warp halves disagree on
stopping, each lane follows its own half -- kept when its half stops, masked by its half's pivot when its half goes on.

The rows are ordinary logits, their stats from the model's stats kernel.  The step kernels' partials are constructed: every pivot of
every round holds mass >= p and none passes the lower bound, so the search narrows onto the row's top probability, where each half's
own scalars decide `nomask`.  The seeds give both orders."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import topp_split as TS

f32 = np.float32
V, S, P = 3000, 16, f32(0.5)
F, R = TS.SPLIT_FANOUT, TS.SPLIT_ROUNDS
#: (seed, the halves' stops)
ROWS = [(35, [False, True]), (50, [False, True]), (17, [True, False]), (47, [True, False])]


def _row(seed: int) -> np.ndarray:
    return (-np.random.default_rng(seed).random(V) * f32(2.5e-5)).astype(f32)


def _narrowing_parts() -> np.ndarray:
    parts = np.zeros((R, S, F, 3), f32)
    parts[..., 0] = f32(1.0 / S)          # mass per slice: every pivot holds mass >= p
    parts[..., 1] = f32(1e-12)            # min prob above the pivot: the lower bound mass - min * count stays >= p
    parts[..., 2] = f32(1.0)
    return parts


def _expected(x: np.ndarray, stats: np.ndarray, parts: np.ndarray, cfg, stopped: list[bool]) -> np.ndarray:
    """Per lane: its half stopped, or its logit above its half's pivot logit (no lane's probability equals the constructed dup)."""
    ch = [TS._mask_chain(stats, parts, P, S, F, R, cfg, h) for h in (0, 1)]
    slice_len = (V + S - 1) // S                                  # each slice is one mask block at this V and S
    i = np.arange(V)
    half = TS.lane_half(i - (i // slice_len) * slice_len, cfg, S)
    pivot_logit = np.array([np.inf if stopped[h] else ch[h]["pivot_logit"] for h in (0, 1)], f32)[half]
    return np.array(stopped)[half] | (x > pivot_logit)


@pytest.mark.parametrize("seed,stopped", ROWS, ids=[f"seed{s}-stops{int(a)}{int(b)}" for s, (a, b) in ROWS])
def test_each_lane_follows_its_own_half(seed, stopped):
    x, cfg = _row(seed), TS.Cfg.for_vocab(V, S)
    stats, parts = TS.stats_kernel(x, P, S, TS.STATS_BLOCK, cfg), _narrowing_parts()
    trace: dict = {}
    out = TS.mask_kernel(x, stats, parts, P, S, F, R, TS.MASK_BLOCK, cfg, trace)
    keep = _expected(x, stats, parts, cfg, stopped)
    assert trace["stopped"] == stopped and trace["half_divergence"]
    assert np.array_equal(trace["keep"], keep) and keep.any() and not keep.all()
    assert np.array_equal(out.view(np.uint32), np.where(keep, x, TS.NEG_INF).astype(f32).view(np.uint32))


def test_the_keep_word_is_defined_on_such_a_row(monkeypatch):
    def step(x, stats, parts, p, ROUND, S_, F_, R_, BLOCK, cfg):     # the constructed partials in place of the step kernels'
        parts[...] = _narrowing_parts()

    monkeypatch.setattr(TS, "step_kernel", step)
    word = SM.topp_mask_word(V)
    for seed, stopped in ROWS:
        x, cfg = _row(seed), TS.Cfg.for_vocab(V, S)
        keep = _expected(x, TS.stats_kernel(x, P, S, TS.STATS_BLOCK, cfg), _narrowing_parts(), cfg, stopped)
        assert np.array_equal(SM.topp_keep(x, P, S), keep)
        bits = word.evaluate(*x.view(np.uint32).tolist(), int(P.view(np.uint32)), S)
        assert bits == int.from_bytes(np.packbits(keep, bitorder="little").tobytes(), "little")


def test_a_half_that_stops_at_the_uniform_test_carries_no_scalars(monkeypatch):
    """The same reading when the stopped half's chain ends before any scalar (only its `early` is known)."""
    real = TS._mask_chain

    def chain(stats, parts, p, S_, F_, R_, cfg, half):
        return {"early": "uniform-or-empty"} if half == 1 else real(stats, parts, p, S_, F_, R_, cfg, half)

    monkeypatch.setattr(TS, "_mask_chain", chain)
    seed, _ = ROWS[0]
    x, cfg = _row(seed), TS.Cfg.for_vocab(V, S)
    stats, parts = TS.stats_kernel(x, P, S, TS.STATS_BLOCK, cfg), _narrowing_parts()
    trace: dict = {}
    TS.mask_kernel(x, stats, parts, P, S, F, R, TS.MASK_BLOCK, cfg, trace)
    assert trace["stopped"] == [False, True]
    assert np.array_equal(trace["keep"], _expected(x, stats, parts, cfg, [False, True]))
