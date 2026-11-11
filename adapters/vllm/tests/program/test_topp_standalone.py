"""vLLM's standalone top-p kernel path (`registry.topp_standalone`, the exact model; `registry.topp_standalone_words`, its word
gates): the builder computes the model's keep bits on every row, the Definitions evaluate to them through the circuit evaluator
with no dead gate, the compaction keeps every flagged value in lane order, the sigma cut is the kernel's index at every boundary,
and the select is `GumbelSelectF32_v3` over the standalone mask.  The rows are served-shaped (bf16 logits, some masked to -inf,
ties) and reach both searches, the duplicate trim and p >= 1."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.liveness import dead_gates
from verity.circuits.program import Program
from verity.definitions.registry import dead_free as DF
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import topp_standalone as TS
from verity.definitions.registry import topp_standalone_words as TSW
from verity.definitions.registry.topp_keep_builder import Engine

f32 = np.float32
P = (0.95, 0.5, 0.9, 0.95, 0.8, 1.0, 0.3, 0.99)


def _bf16(x: np.ndarray) -> np.ndarray:
    """``x`` rounded to bf16 (to nearest even), as f32."""
    w = np.asarray(x, f32).view(np.uint32).astype(np.uint64)
    w = ((w + 0x7FFF + ((w >> 16) & 1)) >> 16 << 16).astype(np.uint32)
    return np.where(np.isinf(x), np.asarray(x, f32), w.view(f32))


def _rows(V: int, n: int, seed: int):
    """(logits, p): peaked, flat, tied, a first block above the rest (the full-softmax search), half masked, Gumbel."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            x = rng.standard_normal(V) * 2
            x[rng.integers(0, V, 5)] += 8
        elif kind == 1:
            x = rng.standard_normal(V) * 0.05
        elif kind == 2:
            x = np.round(rng.standard_normal(V) * 2) / 2
            x[rng.integers(0, V, 6)] = 5.0
        elif kind == 3:
            x = rng.standard_normal(V) * 0.1 + 3.0
            x[:min(V, TS.BLOCK)] = rng.standard_normal(min(V, TS.BLOCK)) + 5.0
        elif kind == 4:
            x = rng.standard_normal(V) * 3
            x[rng.random(V) < 0.5] = -np.inf
        else:
            x = -np.log(-np.log(rng.random(V))) * 2
        out.append((_bf16(x.astype(f32)), f32(P[i % len(P)])))
    return out


def _words(x: np.ndarray) -> list[int]:
    return [int(w) for w in np.asarray(x, f32).view(np.uint32)]


@pytest.mark.parametrize("V", [16, 300, pytest.param(9000, marks=pytest.mark.slow)])
def test_the_builder_is_the_model(V):
    rows = _rows(V, 8, V)
    X = np.stack([x for x, _ in rows])
    E = Engine(len(rows))
    x = E.free(V, 32, X.view(np.uint32).astype(np.uint64))
    p = E.free(1, 32, np.array([[p.view(np.uint32)] for _, p in rows], np.uint64))
    got = TSW.standalone_keep(E, x, p).d.astype(bool)
    branches = set()
    for t, (xr, pr) in enumerate(rows):
        tr: dict = {}
        assert (got[t] == TS.standalone_row(xr, pr, tr)).all(), (V, t)
        branches.add(tr.get("outliers"))
    assert branches == {True, False, None}


@pytest.mark.parametrize("V", [1, 16, pytest.param(64, marks=pytest.mark.slow)])
def test_the_keep_word_is_the_model_with_no_dead_gate(V):
    fn = bind(TSW.TopPStandaloneKeep, V=V)
    assert dead_gates(Program(fn), transitive=True).computed == 0
    for x, p in _rows(V, 6, V + 1):
        got = np.array(evaluate(fn, _words(x), int(p.view(np.uint32))), dtype=bool)
        assert (got == TS.standalone_row(x, p)).all(), (V, float(p))


def test_the_select_is_the_gumbel_select_over_the_standalone_mask():
    V, rng = 16, np.random.default_rng(5)
    fn = bind(TSW.GumbelTopPStandaloneSelect, V=V)
    assert dead_gates(Program(fn), transitive=True).computed == 0
    for i, (x, p) in enumerate(_rows(V, 4, 7)):
        xb = [w >> 16 for w in _words(x)]
        seed, pos, temp = int(rng.integers(0, 1 << 63)), int(rng.integers(0, 1 << 20)), int(f32([1.0, 0.7][i % 2]).view(np.uint32))
        scaled = np.array(evaluate(bind(SA.TemperatureScale, V=V), xb, temp), np.uint32).view(f32)
        masked = TS.standalone_mask_row(scaled, p)
        want = evaluate(bind(DF.GumbelSelectF32V3, V=V), _words(masked), seed, pos, temp)
        assert evaluate(fn, xb, int(p.view(np.uint32)), seed, pos, temp) == want


@pytest.mark.parametrize("V", [1, 2, 3, 17, 257])
def test_the_compaction_keeps_the_flagged_values_in_order(V):
    rng = np.random.default_rng(V)
    dens = (0.0, 1.0, 0.5, 0.05, 0.95, 0.3)
    vals = rng.integers(1, 1 << 31, (len(dens), V)).astype(np.uint64)
    flags = np.stack([rng.random(V) < d for d in dens]).astype(np.uint64)
    E = Engine(len(dens))
    got = TSW.compact(E, E.free(V, 32, vals), E.free(V, 1, flags)).d
    for t in range(len(dens)):
        kept = vals[t][flags[t] == 1]
        assert (got[t, :len(kept)] == kept).all() and not got[t, len(kept):].any()


def test_the_cut_is_the_kernels_index():
    """``clamp(int(p * 200), 0, 199)`` (cvt.rzi.s32.f32, NaN to 0) at every k / 200 and the float just below it, and outside
    [0, 1]."""
    ps = [f32(k / 200) for k in range(201)] + [np.nextafter(f32(k / 200), f32(-1)) for k in range(1, 201)]
    ps += [f32(v) for v in (-0.0, -1e-30, -3.0, 1e-45, 5.0, np.inf, -np.inf, np.nan)]
    E = Engine(len(ps))
    got = TSW._cut(E, E.free(1, 32, np.array(ps, f32).view(np.uint32).astype(np.uint64)[:, None])).d[:, 0]
    for t, pv in enumerate(ps):
        assert got[t] == TSW.SIGMA_CUT[max(0, min(TS._s32(f32(pv * f32(200))), 199))], pv


def test_p_of_one_or_more_keeps_every_lane():
    x, _ = _rows(300, 1, 0)[0]
    for p in (1.0, 1.5, np.inf, np.nan):
        assert TS.standalone_row(x, p).all()
