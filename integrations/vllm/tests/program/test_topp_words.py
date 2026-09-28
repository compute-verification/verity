"""The top-p keep word as word gates (`registry.topp_words`): `TopPKeepWord_v1{V,S}` is `topp_keep(x, p, S)` on every row, as the IR's
evaluator computes it from its word gates; `GumbelTopPTokenSelect_v2{V,S}` is v1 wherever `splits` is S; `Q_word` cuts the keep word into
word-sized units with no recomputed value; the new word gates are exact."""
from __future__ import annotations

import numpy as np
import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import topp_word_gates as TWGates
from verity_vllm.program.registry import topp_words as TWG


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


@pytest.mark.parametrize("V,S,n", [(16, 4, 10), (64, 32, 8), (300, 8, 8), (1025, 32, 6)])
def test_the_keep_word_as_word_gates_is_topp_keep(V, S, n):
    fn = bind(TWG.TopPKeepWord, V=V, S=S)
    for x, p in _rows(V, n, V + S):
        got = np.array(evaluate(fn, [int(w) for w in x.view(np.uint32)], int(p.view(np.uint32))), dtype=bool)
        assert (got == SM.topp_keep(x, float(p), S)).all(), (V, S, float(p))


def test_the_select_at_a_constant_split_count_is_v1_there():
    V, rng = 64, np.random.default_rng(3)
    for i in range(3):
        xb = ((rng.standard_normal(V) * 3).astype(np.float32).view(np.uint32) >> 16).tolist()
        args = (xb, 0x3F733333, 1234 + i, 5 + i, 0x3F4CCCCD)
        for spl in (4, 3):                                       # the constant S, and a count no launch has (v1 masks every lane)
            assert evaluate(bind(TWG.GumbelTopPTokenSelectV2, V=V, S=4), *args, spl) == \
                evaluate(bind(SM.GumbelTopPTokenSelect, V=V), *args, spl)


def test_q_word_cuts_it_into_word_units():
    from verity_vllm.query import word as WQ

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
