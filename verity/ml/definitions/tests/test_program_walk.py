"""`numerics.program_walk`: a Program walked against given words, call by call, on small Programs of `Bf16Add_v2`."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.ops.boolean import elementwise as BE
from verity.ml.circuits.refs import Coll, Explicit, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.numerics import program_walk as PW

BF16 = Array(16, BL.BIT)
ADD = bind(BE.Bf16Add)
M = 6


def _probe(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


def _two_adds():
    """s = x + y, then t = s + x, each a batch over M words."""
    def body(B, S, x, y):
        s = B.batch(ADD, x, y, axes=(0, 0), name="s")
        return B.batch(ADD, s, x, axes=(0, 0), name="t")
    return _probe("WalkTwoAdds", (("x", Array(M, BF16)), ("y", Array(M, BF16))), Array(M, BF16), body)


def _add(a, b):
    return evaluate_words(ADD, a, b).reshape(-1)


@pytest.fixture
def words():
    rng = np.random.default_rng(5)
    x, y = (rng.integers(0x3C00, 0x4400, M, dtype=np.uint64).astype(np.uint16) for _ in range(2))
    s = _add(x, y)
    return {"x": x, "y": y, "s": s, "t": _add(s, x)}


def test_each_named_call_matches_on_bits(words) -> None:
    w = PW.walk(_two_adds(), words)
    assert [(c.path, c.via, c.members, c.words, c.mismatches) for c in w.checks] == [("s", "bits", M, M, 0), ("t", "bits", M, M, 0)]
    assert w.checks[0].family == PW.family(ADD) and not w.unchecked and not w.computed


def test_a_wrong_output_is_that_calls_mismatch(words) -> None:
    """The run's t wrong at one word: t's check finds it, at its member; s is untouched."""
    words["t"] = words["t"].copy()
    words["t"][3] ^= 0x0001
    s, t = PW.walk(_two_adds(), words).checks
    assert s.mismatches == 0 and t.mismatches == 1 and t.first == [(3, 0, int(words["t"][3] ^ 1), int(words["t"][3]))]


def test_a_call_reads_the_runs_words_not_its_own(words) -> None:
    """The run's s wrong at one word: s's check finds it, and t, computed from the run's s, disagrees with the run's t there."""
    words["s"] = words["s"].copy()
    words["s"][2] ^= 0x0100
    s, t = PW.walk(_two_adds(), words).checks
    assert (s.mismatches, [f[0] for f in s.first]) == (1, [2])
    assert (t.mismatches, [f[0] for f in t.first]) == (1, [2])


def test_a_model_takes_the_call(words) -> None:
    """A family's model is used over its bits, member axis first; a wrong model is found."""
    seen = []

    def model(statics, a, b):
        seen.append((statics, a.shape, a.dtype, b.shape))
        return _add(a, b)
    w = PW.walk(_two_adds(), words, {PW.family(ADD): model})
    assert [(c.via, c.mismatches) for c in w.checks] == [("model", 0), ("model", 0)]
    assert seen[0] == ({}, (M,), np.dtype(np.uint16), (M,))
    w = PW.walk(_two_adds(), words, {PW.family(ADD): lambda S, a, b: a})
    assert [c.mismatches for c in w.checks] == [int(np.sum(words["x"] != words["s"])), int(np.sum(words["s"] != words["t"]))]


def test_over_the_bit_budget_it_walks_the_bodies_down_to_gates(words) -> None:
    """With no bit budget the walk descends: a call of the two adds, each add's members walked gate by gate."""
    inner = _two_adds()

    def body(B, S, x, y):
        return B.call(inner, x, y, name="inner")
    outer = _probe("WalkOuter", (("x", Array(M, BF16)), ("y", Array(M, BF16))), Array(M, BF16), body)
    values = {"x": words["x"], "y": words["y"], "inner/s": words["s"], "inner/t": words["t"], "inner": words["t"]}
    w = PW.walk(outer, values, bits_max=0)
    assert [(c.path, c.via, c.mismatches) for c in w.checks] == [("inner/s", "body", 0), ("inner/t", "body", 0), ("inner", "body", 0)]
    assert any(p.startswith("inner/s[5]/#") for p in w.computed) and not w.unchecked
    assert PW.walk(outer, values, bits_max=0, descend_max=M - 1).unchecked[0][0] == "inner/s"


def test_a_call_with_no_parameter_is_walked_through_its_body(words) -> None:
    """A call of constants, as the training step's -100 and 1.0: its body's constant bits, then an add that reads them."""
    one = 0x3F80

    def consts(B, S):
        bit = (B.call(BL.constant(0)).refs[0], B.call(BL.constant(1)).refs[0])
        return Coll(BF16, Explicit(tuple(bit[(one >> i) & 1] for i in range(16))))
    k = _probe("WalkConstants", (), BF16, consts)

    def body(B, S, x):
        c = B.call(k, name="k")
        return B.batch(ADD, x, Coll(Array(M, BF16), concat([c.refs] * M)), axes=(0, 0), name="s")
    prog = _probe("WalkReadsConstants", (("x", Array(M, BF16)),), Array(M, BF16), body)
    s = _add(words["x"], np.full(M, one, np.uint16))
    w = PW.walk(prog, {"x": words["x"], "k": np.uint16(one), "s": s})
    assert [(c.path, c.via, c.members, c.mismatches) for c in w.checks] == [("k", "body", 1, 0), ("s", "bits", M, 0)]
    assert not w.unchecked


def test_a_scan_runs_its_step_per_iteration(words) -> None:
    """A running sum: carry' = carry + x_i, y_i = carry; on bits and body by body."""
    step = _probe("WalkSumStep", (("c", BF16), ("x", BF16)), Tuple(BF16, BF16), lambda B, S, c, x: tuple_of(B.call(ADD, c, x), c))

    def body(B, S, c0, xs):
        return B.scan(step, c0, xs=(xs,), name="sum")
    prog = _probe("WalkSum", (("c0", BF16), ("xs", Array(M, BF16))), Tuple(BF16, Array(M, BF16)), body)
    c, ys = np.uint16(0x3F80), []
    for x in words["x"]:
        ys.append(c)
        c = _add(np.array([c]), np.array([x]))[0]
    values = {"c0": np.uint16(0x3F80), "xs": words["x"], "sum": (np.array([c]), np.array(ys))}
    for budget in (PW.BITS_MAX, 0):
        (chk,) = PW.walk(prog, values, bits_max=budget).checks
        assert (chk.path, chk.via, chk.members, chk.words, chk.mismatches) == ("sum", "scan", M, M + 1, 0)


def test_the_walk_refuses_missing_and_misshapen_words(words) -> None:
    with pytest.raises(ValueError, match=r"parameters \['y'\]"):
        PW.walk(_two_adds(), {"x": words["x"]})
    with pytest.raises(ValueError, match="5 words given"):
        PW.walk(_two_adds(), {**words, "s": words["s"][:5]})
    with pytest.raises(ValueError, match="wider than its element"):
        PW.walk(_two_adds(), {**words, "x": words["x"].astype(np.uint32) | 0x10000})
