"""The top-p keep word as word gates (`registry.topp_words`): `TopPKeepWord_v1{V,S}` is `topp_keep(x, p, S)` on every row, as the circuit's
evaluator computes it from its word gates, including the constructed rows whose warp halves disagree on stopping;
`GumbelTopPTokenSelect_v2{V,S}` is v1 wherever `splits` is S; the program codec round-trips them; `Q_word` cuts the keep word into
word-sized units with no recomputed value; the new word gates are exact."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.codec import decode_program, encode_program
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import topp_split as TS
from verity.ml.definitions.registry import topp_keep_builder as TKB
from verity.ml.definitions.registry import topp_word_gates as TWGates
from verity.ml.definitions.registry import topp_words as TWG
from verity.ml.definitions.registry.prims import BIT, F32

f32 = np.float32
#: the constructed rows (`test_topp_split_halves`): ordinary logits at V = 3000, S = 16, p = 0.5, the step kernels' partials
#: constructed so that each warp half's own scalars decide `nomask`; {seed: the halves' stops}, both orders
HALVES = {35: [False, True], 50: [False, True], 17: [True, False], 47: [True, False]}


def _rows(V, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = (rng.standard_normal(V) * rng.choice([0.5, 3, 10])).astype(np.float32)
        if i % 5 == 1:
            x = np.round(x).astype(np.float32)                  # ties
        if i % 7 == 2:
            x[rng.choice(V, 2)] = np.nan
        if i % 6 == 3:
            x[:] = -np.inf
            x[rng.choice(V, 3)] = 1.0
        if i % 9 == 4:
            x[:] = 1.0                                           # a uniform row
        if i % 8 == 5:
            x[rng.choice(V, 2)] = np.inf
        p = np.float32([0.9, 0.5, 0.3, 0.99, 1.0, 0.95, 0.0][i % 7])
        out.append((x, p))
    return out


def _halves(monkeypatch):
    """The keep word's own body (`topp_words.keep_word_body`) with their partials in place of its step kernels', and the rows,
    `topp_split`'s step kernels given the same partials."""
    V, S, P = 3000, 16, f32(0.5)
    parts = np.zeros((TS.SPLIT_ROUNDS, S, TS.SPLIT_FANOUT, 3), f32)
    parts[..., 0], parts[..., 1], parts[..., 2] = f32(1.0 / S), f32(1e-12), f32(1.0)

    def step(x, stats, parts_, p, ROUND, S_, F_, R_, BLOCK, cfg):
        parts_[...] = parts

    def parts_of(E, r, S_):
        n = TKB.F * S_
        return [E.const([int(f32(1.0 / S_).view(np.uint32))] * n), E.const([int(f32(1e-12).view(np.uint32))] * n),
                E.const([1] * n, TWGates.CW)]

    monkeypatch.setattr(TS, "step_kernel", step)
    rows = []
    for seed, stopped in HALVES.items():
        x = (-np.random.default_rng(seed).random(V) * f32(2.5e-5)).astype(f32)
        assert TS.topp_split_row(x, float(P), S)[3]["stopped"] == stopped
        rows.append((x, P))
    fn = CompositeDefinition("TopPKeepWordHalves", 1, (), lambda st: ((("logits", Array(V, F32)), ("top_p", F32)), Array(V, BIT)),
                             lambda B, st, logits, top_p: TWG.keep_word_body(B, S, logits, top_p, parts_of), register=False)
    return bind(fn), rows, S


@pytest.mark.parametrize("V,S,n", [(16, 4, 10)] + [pytest.param(*a, marks=pytest.mark.slow)
                                       for a in [(64, 32, 8), (300, 8, 8), (1025, 32, 6), (3000, 16, "halves")]])
def test_the_keep_word_as_word_gates_is_topp_keep(V, S, n, monkeypatch):
    if n == "halves":
        fn, rows, S = _halves(monkeypatch)
    else:
        fn, rows = bind(TWG.TopPKeepWord, V=V, S=S), _rows(V, n, V + S)
    for x, p in rows:
        got = np.array(evaluate(fn, [int(w) for w in x.view(np.uint32)], int(p.view(np.uint32))), dtype=bool)
        ref = SM.topp_keep(x, float(p), S)
        assert (got == ref).all(), (V, S, float(p))
        assert n != "halves" or (ref.any() and not ref.all())


@pytest.mark.slow
def test_the_select_at_a_constant_split_count_is_v1_there():
    V, rng = 64, np.random.default_rng(3)
    for i in range(3):
        xb = ((rng.standard_normal(V) * 3).astype(np.float32).view(np.uint32) >> 16).tolist()
        args = (xb, 0x3F733333, 1234 + i, 5 + i, 0x3F4CCCCD)
        for spl in (4, 3):                                       # the constant S, and a count no launch has (v1 masks every lane)
            assert evaluate(bind(TWG.GumbelTopPTokenSelectV2, V=V, S=4), *args, spl) == \
                evaluate(bind(SM.GumbelTopPTokenSelect, V=V), *args, spl)


@pytest.mark.slow
@pytest.mark.parametrize("fam,V,S", [("TopPKeepWord", 300, 8), ("TopPKeepWord", 1025, 32), ("GumbelTopPTokenSelectV2", 300, 8)])
def test_the_codec_round_trips_it(fam, V, S):
    """The Definition as one Call of a Program, through `encode_program` and `decode_program` as a Build's workload compose reads it.
    Its NaN tests (`x == x`) read one multi-part sequence twice: the writer must not alias a node's second argument to its first.
    (At V = 16, S = 4 each such sequence is one run, so the smallest shapes never exercised the alias.)"""
    fn = bind(getattr(TWG, fam), V=V, S=S)
    params, ret = fn.signature()
    P = Program(CompositeDefinition(f"OneCall[{fn.id}]", 1, (), lambda st: (tuple(params), ret), lambda B, st, *x: B.call(fn, *x),
                                    register=False))
    D = decode_program(json.loads(json.dumps(encode_program(P))), REGISTRY)
    assert D.gates == P.gates and D.root.id == P.root.id


@pytest.mark.slow
def test_q_word_cuts_it_into_word_units():
    from verity.ml.definitions import word as WQ

    r = WQ.unit_rule(bind(TWG.TopPKeepWord, V=300, S=8), 16, 32)
    assert not r["violations"] and r["max_out_bits"] <= 32 and r["units_per_call"] > 1000


def test_the_new_word_gates():
    f = lambda x: int(np.float32(x).view(np.uint32))
    assert TWGates.TopPNumkeep.evaluate(10, f(3.7)) == 7 and TWGates.TopPNumkeep.evaluate(10, f(12.0)) == 1
    assert TWGates.TopPNumkeep.evaluate(0, f(3.0)) == 0 and TWGates.TopPNumkeep.evaluate(5, f(float("nan"))) == 5
    assert TWGates.TopPNumkeep.evaluate(5, f(-2.0)) == 5 and TWGates.TopPNumkeep.evaluate(5, f(float("inf"))) == 5
    assert TWGates.I32ToF32.evaluate(7) == f(7.0) and TWGates.I32ToF32.evaluate(0xFFFFFFFF) == f(-1.0)
    assert TWGates.I32ToF32.evaluate(16777217) == f(16777216.0)      # ties to even
    assert TWGates.NvLogf.evaluate(f(1.0)) == 0 and TWGates.NvLogf.evaluate(0xBF800000) == 0xFFC00000
    assert TWGates.BitXor.evaluate(1, 1) == 0 and TWGates.U32Xor.evaluate(0xF0F0, 0xFFFF) == 0x0F0F and TWGates.SelectBit.evaluate(0, 1, 0) == 0
