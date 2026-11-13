"""`GumbelTopPTokenSelect_v1{V}` cut into units of at most a Flock block each (`registry.topp_stages`), and their twin
(`numerics.topp_stage_twins`), from which the serving reader commits a row's units:

* every unit's Definition on the twin's inputs for it (its lanes of the row, the Call's operands and its writers' outputs)
  computes the twin's outputs, and the twin's token is the Call's in both its forms, as the circuit evaluator computes them: on
  rows with ties, -inf and NaN lanes, at temperature 0, 0.7 and 1, at every split count;
* a select chunk of `SELECT_N` lanes that a slice ends inside counts its ties as the row does, on rows where that count
  decides the Call's token, and gives a lane past the end its new slice's warp half, on a row where a tie's probability is
  its half's;
* every select chunk of a served row gives its lanes their own slice's positions (`topp_stages.positions`), at every split
  count that has slices, and the next chunk its first lane's;
* the twin's keep bits are `topp_keep`'s at every split count on the edge rows (NaN and +inf lanes, a few finite lanes, a
  uniform row, p = 0 and 1);
* `plan` wires each parameter to an earlier unit's output of its type and `cut` counts its units, at small and served statics;
* the twin's fast word gates are their primitives' references.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.circuits.codec import lookup_primitive
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.types import Tuple
from verity.definitions.numerics import topp_stage_twins as TW
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import topp_keep_builder as K
from verity.definitions.registry import topp_stages as T
from verity.definitions.registry.topp_word_gates import CW

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


@pytest.mark.slow
def test_a_select_chunk_counts_its_ties_across_a_slice_s_end():
    """A V = 1000 row's select chunks hold `SELECT_N` lanes, and at split counts 2, 4, 8 and 16 a slice ends inside the chunk
    of lane 504 before it, which no chunk of a V = 64 row has (its two lanes end with their slices).  On these two rows the
    lane's tie count decides the token: it ties with lanes 3 and 470 and its noise beats theirs, so the Call's token is lane 3
    when top_p keeps two of the ties (it is the third along the row) and lane 504 when it keeps three.  Every select chunk
    computes the twin's words, which count ties along the row: a chunk that read a lane past the slice's end with the ties
    before the slice it began in keeps the lane on the first row, one that read the slice after's drops it on the second, and
    one that didn't restart its count at the slice's end carries another count.  Half a minute, most of it the twin."""
    V, lane = 1000, 504
    sel = [u for u in T.plan(V) if u.key[0] == "select"]
    at = sel[lane // T.SELECT_N]
    assert T.chunk(V) == T.SELECT_N and at.lanes[0] <= lane < at.lanes[1]
    for S in (2, 4, 8, 16):
        assert any(at.lanes[0] < end <= lane for end in range(T.Geo(V, S).L, V, T.Geo(V, S).L)), S
    for ties, top_p, tok in ((6, 0.3, 3), (7, 0.4, lane)):
        x = np.full(V, 1.0, np.float32)
        x[[3, 470, lane, 600, 700, 800, 900][:ties]] = 8.0
        lg = bf16_row(x)
        ops = {"top_p": f32(top_p), "seed": 0x1234_5678_9ABC_DEF0, "pos": 42, "temp": f32(0.7)}
        for S in (2, 4, 8, 16):
            assert evaluate(bind(SA.GumbelTopPTokenSelect, V=V), [int(v) for v in lg], *(ops[o] for o in OPERANDS[:4]), S) == (tok,)
            got, outs = TW.stage_rows(V, lg, splits=S, **ops)
            bad = [u.key for u in sel if [int(v) for v in evaluate(u.defn, *unit_args(u, lg, {**ops, "splits": S}, outs))]
                   != [int(v) for f in outs[u.key] for v in f]]
            assert (got, bad) == (tok, []), (top_p, S)


@pytest.mark.slow
def test_a_lane_past_a_slice_s_end_inside_a_chunk_takes_its_new_slice_s_warp_half():
    """The two warp halves hold values of Z an ulp apart when the slices' exp sums round differently in the halves' two orders
    (`topp_geometry.z_group`); a tie's probability is then its half's, and a lane counts as a tie only at its own half's.  On
    such a row of V = 1000 (a lone top lane, seven ties under it, a floor in steps of 1/64), lane 504 is a tie past a slice's
    end inside its chunk at split counts 4, 8 and 16, in the other half than its offset would have in the slice that ended.
    Every select chunk computes the twin's words, each split count's tie count among them, and the twin's token is the Call's:
    a chunk that took the half of its lanes past the end from the slice they left carries a tie count one off at all three
    split counts, which the rows of equal ties above don't see, their halves' Z being one word.  A quarter of a minute."""
    V, lane, ties = 1000, 504, [3, 470, 504, 600, 700, 800, 900]
    x = ((np.arange(V) * 5 % 257) / 64 - 2).astype(np.float32)
    x[ties], x[40] = 8.0, 9.0
    lg = bf16_row(x)
    sel = [u for u in T.plan(V) if u.key[0] == "select"]
    at = sel[lane // T.SELECT_N]
    ops = {"top_p": f32(0.5), "seed": 0x1234_5678_9ABC_DEF0, "pos": 42, "temp": f32(0.7)}
    for S in (4, 8, 16):
        geo, off = T.Geo(V, S), lane % T.Geo(V, S).L
        assert at.lanes[0] < lane - off <= lane < at.lanes[1] and geo.h1(off) != geo.h1(off + geo.L), S
        got, outs = TW.stage_rows(V, lg, splits=S, **ops)
        pr = {v: int(w) for u in T.plan(V) if u.key[:2] == ("probs", S) for v, w in zip(range(*u.lanes), outs[u.key][0])}
        assert len({pr[t] for t in ties}) == 2, S
        assert evaluate(bind(SA.GumbelTopPTokenSelect, V=V), [int(v) for v in lg], *(ops[o] for o in OPERANDS[:4]), S) == (got,)
        bad = [u.key for u in sel if [int(v) for v in evaluate(u.defn, *unit_args(u, lg, {**ops, "splits": S}, outs))]
               != [int(v) for f in outs[u.key] for v in f]]
        assert bad == [], S


def test_every_served_select_chunk_s_lanes_take_their_own_slice_s_positions():
    """At GLM-5.3's vocabulary, `topp_stages.positions` on the first lane's position of every select chunk, at every split
    count that has slices: each lane's warp half, whether a mask block of its slice starts at it (its tie count restarts) and
    whether it is past a slice's end are those of its own offset in its own slice, counted along the row as the twin counts
    them, and the position of the lane past the chunk is the next chunk's first.  A slice ends inside 8 chunks at S = 16 and
    24 at S = 32, and inside none at a smaller split count.  At S = 32 each of the 384 lanes past such an end is in the other
    half than its offset would have in the slice that ended, so a chunk that took those lanes' half from the slice they left
    is wrong in all 24; at S = 16 the two halves are one, and the row of 1,000 lanes above holds that split count.  The
    chunks are the rows of one engine, the twin's, which runs one row."""
    V, n = SERVED_V, T.chunk(SERVED_V)
    assert n == T.SELECT_N and V % n == 0
    first = np.arange(0, V, n)
    lanes = first[:, None] + np.arange(n)
    E = TW.ArrayEngine()
    E.T = len(first)

    def pos(geo, lane):
        off = lane % geo.L
        return np.stack([off, off % K.MASK_BLOCK, off % geo.P, lane // geo.L], 1)

    def words(v):
        return np.where(v.k, v.kv, v.d)

    inside, other = {}, {}
    for S in K.SPLIT_COUNTS[1:]:
        geo = T.Geo(V, S)
        h1, reset, wr, nxt = T.positions(E, geo, n, tuple(E.free(1, CW, w) for w in pos(geo, first).T))
        off, past = lanes % geo.L, lanes // geo.L != first[:, None] // geo.L
        want = {"h1": geo.h1(off), "reset": off % K.MASK_BLOCK == 0, "past": past, "next": pos(geo, first + n)}
        got = {"h1": words(h1), "reset": words(reset), "past": words(wr), "next": np.concatenate([words(w) for w in nxt], 1)}
        bad = {name: np.nonzero((got[name] != want[name]).any(1))[0].tolist() for name in want}
        assert not any(bad.values()), (S, {name: (len(ks), ks[:4]) for name, ks in bad.items() if ks})
        inside[S], other[S] = int(past.any(1).sum()), int((past & (geo.h1(off) != geo.h1(off + geo.L))).sum())
    assert len(first) == 4840 and inside == {2: 0, 4: 0, 8: 0, 16: 8, 32: 24} and other == {2: 0, 4: 0, 8: 0, 16: 0, 32: 384}


EDGE = ["ties", "nan", "few", "uniform", "posinf", "plain", "neginf"]


@pytest.mark.slow
def test_the_twin_s_keep_bits_are_topp_keep_s():
    from verity.definitions.registry import sampling_rows as SR

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
    """At GLM-5.3's vocabulary the Call is 8,159 units of 85 Definitions: per split count its stats blocks and their parts,
    its step blocks' probabilities and five rounds of steps, and 4,840 select chunks of 32 lanes (`SELECT_N`)."""
    cut = T.cut(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=SERVED_V))
    by = Counter()
    for d, n in cut:
        by[d.defn.name] += n
    assert (len(cut), sum(by.values())) == (85, 8159)
    assert by == {"TopPStatsHead": 147, "TopPStatsExp": 268, "TopPProbs": 484, "TopPStep": 2420, "GumbelTopPSelect": 4840}
    assert {d.bindings["N"] for d, _n in cut if d.defn.name == "GumbelTopPSelect"} == {T.SELECT_N} == {T.chunk(SERVED_V)}
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
