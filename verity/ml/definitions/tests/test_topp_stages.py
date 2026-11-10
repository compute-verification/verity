"""`GumbelTopPTokenSelect_v1{V}` cut into units of at most a Flock block each (`registry.topp_stages`), and their twin
(`numerics.topp_stage_twins`), from which the serving reader commits a row's units:

* every unit's Definition on the twin's inputs for it (its lanes of the row, the Call's operands and its writers' outputs)
  computes the twin's outputs, and the twin's token is the Call's in both its forms, as the circuit evaluator computes them: on
  rows with ties, -inf and NaN lanes, at temperature 0, 0.7 and 1, at every split count;
* the twin's keep bits are `topp_keep`'s at every split count on the edge rows (NaN and +inf lanes, a few finite lanes, a
  uniform row, p = 0 and 1);
* `plan` wires each parameter to an earlier unit's output of its type and `cut` counts its units, at small and served statics;
* the twin's fast word gates are their primitives' references.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.ml.circuits.codec import lookup_primitive
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.types import Tuple
from verity.ml.definitions.numerics import topp_stage_twins as TW
from verity.ml.definitions.registry import sampling as SA
from verity.ml.definitions.registry import topp_keep_builder as K
from verity.ml.definitions.registry import topp_stages as T

OPERANDS = ("top_p", "seed", "pos", "temp", "splits")
SERVED_V = 154880


def f32(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def bf16_row(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def row(V: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(V) * 3).astype(np.float32)
    if kind == "ties":
        x = np.round(x).astype(np.float32)
    elif kind == "neginf":
        x[rng.choice(V, V // 4, replace=False)] = -np.inf
    elif kind == "nan":
        x[rng.choice(V, 2, replace=False)] = np.nan
    return bf16_row(x)


def fields(defn) -> list[int]:
    _, ret = defn.signature()
    fs = [f[1] if isinstance(f, tuple) else f for f in ret.fields] if isinstance(ret, Tuple) else [ret]
    return [len(list(f.leaf_widths())) for f in fs]


def unit_args(u: T.Unit, logits: np.ndarray, ops: dict, outs: dict) -> list:
    wires = {p: (k, j) for p, k, j in u.wires}
    args = []
    for name, _ty in u.defn.signature()[0]:
        if name == "logits":
            args.append([int(v) for v in logits[u.lanes[0]:u.lanes[1]]])
        elif name in wires:
            args.append([int(v) for v in outs[wires[name][0]][wires[name][1]]])
        else:
            args.append(ops[name])
    return args


# (row kind, temperature, top_p, the split count the units are evaluated at)
ROWS = [("plain", 0.7, 0.9, 32), ("ties", 0.0, 0.5, 1), ("neginf", 1.0, 0.95, 8), ("nan", 0.7, 0.3, 16)]


@pytest.mark.slow
@pytest.mark.parametrize("kind,temp,top_p,at", ROWS, ids=[r[0] for r in ROWS])
def test_the_units_compute_the_twin_s_words_whose_token_is_the_call_s(kind, temp, top_p, at):
    V = 64
    lg = row(V, kind, seed=V + len(kind))
    ops = {"top_p": f32(top_p), "seed": 0x1234_5678_9ABC_DEF0, "pos": 41, "temp": f32(temp)}
    for S in K.SPLIT_COUNTS:
        tok, outs = TW.stage_rows(V, lg, splits=S, **ops)
        for fam in (SA.GumbelTopPTokenSelect, SA.GumbelTopPTokenSelectSharedGreedy):
            assert evaluate(bind(fam, V=V), [int(v) for v in lg], *(ops[o] for o in OPERANDS[:4]), S) == (tok,), (fam, S)
        if S != at:
            continue
        bad = []
        for u in T.plan(V):
            got = [int(v) for v in evaluate(u.defn, *unit_args(u, lg, {**ops, "splits": S}, outs))]
            if got != [int(v) for f in outs[u.key] for v in f] or len(got) != sum(fields(u.defn)):
                bad.append(u.key)
        assert bad == []
        assert [int(v) for v in outs[T.plan(V)[-1].key][0]] == [tok]


EDGE = ["ties", "nan", "few", "uniform", "posinf", "plain", "neginf"]


@pytest.mark.slow
def test_the_twin_s_keep_bits_are_topp_keep_s():
    from verity.ml.definitions.registry import sampling_rows as SR

    V = 200
    rng = np.random.default_rng(5)
    for i, kind in enumerate(EDGE):
        x = (rng.standard_normal(V) * 3).astype(np.float32)
        if kind == "ties":
            x = np.round(x).astype(np.float32)
        elif kind == "nan":
            x[rng.choice(V, 2, replace=False)] = np.nan
        elif kind == "few":
            x[:] = -np.inf
            x[rng.choice(V, 3, replace=False)] = 1.0
        elif kind == "uniform":
            x[:] = 1.0
        elif kind == "posinf":
            x[rng.choice(V, 2, replace=False)] = np.inf
        elif kind == "neginf":
            x[rng.choice(V, V // 4, replace=False)] = -np.inf
        lg, p, temp = bf16_row(x), np.float32([0.9, 0.5, 0.3, 0.99, 1.0, 0.95, 0.0][i]), f32([1.0, 0.7, 0.0][i % 3])
        xs = SR.temperature_scale_row(lg, temp).view(np.float32)
        for S in K.SPLIT_COUNTS:
            want = np.asarray(SA.topp_keep(xs, p, S), bool)
            assert np.array_equal(TW.keep_bits(V, lg, int(p.view(np.uint32)), temp, S), want), (kind, S)


@pytest.mark.parametrize("V", [16, 64, 1000, SERVED_V])
def test_the_plan_wires_each_parameter_to_an_earlier_unit_s_output_of_its_type(V):
    units = T.plan(V)
    seen: dict[tuple, object] = {}
    outputs: dict[tuple[object, str], set[int]] = {}
    for u in units:
        params = dict(u.defn.signature()[0])
        wired = {p for p, _k, _j in u.wires}
        assert set(params) - wired - {"logits"} <= set(OPERANDS), u.key
        assert u.defn.bindings["N"] == u.lanes[1] - u.lanes[0] and 0 <= u.lanes[0] < u.lanes[1] <= V
        for p, k, j in u.wires:
            _, ret = seen[k].signature()
            fs = [f[1] if isinstance(f, tuple) else f for f in ret.fields] if isinstance(ret, Tuple) else [ret]
            assert fs[j] == params[p], (u.key, p, k, j)
            outputs.setdefault((u.defn, p), set()).add(j)
        seen[u.key] = u.defn
    # a serving window wires a template's parameter to one output port of its writers (`stage_rows.wire`)
    assert all(len(js) == 1 for js in outputs.values()), {(d.id, p): js for (d, p), js in outputs.items() if len(js) > 1}
    assert len(seen) == len(units)
    assert sum(n for _d, n in T.cut(bind(SA.GumbelTopPTokenSelect, V=V))) == len(units)
    assert units[-1].defn.signature()[1] == bind(SA.GumbelTopPTokenSelect, V=V).signature()[1]
    sel = [u for u in units if u.key[0] == "select"]
    assert [u.lanes for u in sel] == [(lo, min(lo + T.chunk(V), V)) for lo in range(0, V, T.chunk(V))]


def test_the_served_cut_s_units():
    """At GLM-5.3's vocabulary the Call is 3,924 units of 85 Definitions: per split count its stats blocks and their parts,
    its step blocks' probabilities and five rounds of steps, and 605 select chunks of at most 256 lanes."""
    cut = T.cut(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V))
    by = Counter()
    for d, n in cut:
        by[d.defn.name] += n
    assert (len(cut), sum(by.values())) == (85, 3924)
    assert by == {"TopPStatsHead": 147, "TopPStatsExp": 268, "TopPProbs": 484, "TopPStep": 2420, "GumbelTopPSelect": 605}
    assert T.cut(bind(SA.GumbelTopPTokenSelect, V=SERVED_V)) == cut
    assert {SA.CUTS[f] for f in ("GumbelTopPTokenSelect_v1", "GumbelTopPTokenSelectSharedGreedy_v1")} == {T.__name__}


SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 1, 0x807FFFFF, 0x3F800000, 0xBF800000,
           0x7F7FFFFF, 0x00800000, 0xFFFFFFFF]


def test_the_fast_word_gates_are_their_primitives():
    rng = np.random.default_rng(11)
    pairs = np.array(SPECIAL, np.uint64)[np.indices((len(SPECIAL), len(SPECIAL))).reshape(2, -1)]
    for pid, fast in TW.FAST.items():
        prim = lookup_primitive(REGISTRY, pid)
        widths = [next(iter(set(t.leaf_widths()))) for _, t in prim.signature()[0]]
        cols = []
        for w in widths:
            if w == 1:
                cols.append(rng.integers(0, 2, 600).astype(np.uint64))
            else:
                c = rng.integers(0, 1 << 32, 600, dtype=np.uint64)
                c[:pairs.shape[1]] = pairs[sum(x != 1 for x in widths[:len(cols)]) % 2]
                cols.append(c)
        got = np.asarray(fast(prim, *cols), np.uint64)
        want = np.array([prim.evaluate(*(int(c[i]) for c in cols)) for i in range(600)], np.uint64)
        assert np.array_equal(got, want), (pid, np.flatnonzero(got != want)[:4])
