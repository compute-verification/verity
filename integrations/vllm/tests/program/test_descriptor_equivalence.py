"""The graph/reference correspondence test across codec transitions (`verity_vllm.program.descriptor_equivalence`,
descriptor-codec.md §C): representation-independent reference runs, an exhaustive definition comparison, and
sampled gates / calls / boundaries / evaluation. It must accept the same program under different encodings and
reject programs that differ in gates, producers or reference order."""

from __future__ import annotations

import json
import random

import pytest

from verity.ml.circuits import Array, Program, Value, bind, composite, decode_program, encode_program, REGISTRY
from verity.ml.circuits.codec import SCHEMA_V0, SCHEMA_V1
from verity_vllm.program.descriptor_equivalence import Mismatch, canonical_runs, canonical_runs_by_leaf, compare
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided, concat
from verity_vllm.program.registry import b1

from tests.program.test_running_example import Add16, Id16, Mul16, make_program

V16 = Value(16)


def _random_part(rng: random.Random):
    kind = rng.choice(["affine", "strided", "explicit", "broadcast"])
    space, idx = rng.choice(["p", "n"]), rng.randrange(4)
    if kind == "affine":
        return Affine(space, idx, rng.randrange(20), rng.randrange(1, 8))
    if kind == "broadcast":
        return Strided(space, idx, rng.randrange(20), 1, 0, 0, rng.randrange(1, 8))
    if kind == "strided":
        inner = rng.randrange(1, 4)
        return Strided(space, idx, rng.randrange(20), inner, rng.randrange(-3, 6), rng.randrange(-2, 3), rng.randrange(1, 12))
    return Explicit(tuple((space, idx, rng.randrange(20)) for _ in range(rng.randrange(1, 6))))


def test_canonical_runs_equal_the_leaf_by_leaf_reference_and_ignore_representation():
    rng = random.Random(7)
    for _ in range(300):
        parts = [_random_part(rng) for _ in range(rng.randrange(1, 6))]
        if rng.random() < 0.3:  # nest
            parts = [Concat(parts[: len(parts) // 2 + 1]), *parts[len(parts) // 2 + 1:]]
        r = Concat(parts) if len(parts) > 1 else parts[0]
        runs = canonical_runs(r)
        assert runs == canonical_runs_by_leaf(r)
        # the same sequence, split differently or rebuilt through `concat`, has the same runs
        n = r.count
        cut = rng.randrange(n + 1)
        assert canonical_runs(Concat([r.slice(0, cut), r.slice(cut, n)])) == runs
        assert canonical_runs(concat([Explicit(tuple(r))])) == runs
        assert sum(c for *_, c in runs) == n
    # and they are not blind: reordering or repeating a reference changes them
    a = Concat([Affine("n", 0, 0, 3), Affine("n", 1, 0, 3)])
    assert canonical_runs(a) != canonical_runs(Concat([Affine("n", 1, 0, 3), Affine("n", 0, 0, 3)]))
    assert canonical_runs(Affine("n", 0, 0, 3)) != canonical_runs(Strided("n", 0, 0, 1, 0, 0, 3))


def _round_trip(P, schema):
    return decode_program(json.loads(json.dumps(encode_program(P, schema=schema))), REGISTRY)


def test_the_same_program_under_v0_and_v1_encodings_is_equivalent_with_boundaries_and_evaluation():
    P = make_program()
    rep = compare(_round_trip(P, SCHEMA_V0), _round_trip(P, SCHEMA_V1), gate_samples=50, call_samples=8)
    assert rep["verdict"] == "equivalent"
    assert rep["definitions"] > 0 and rep["reference_sequences"] > 0
    assert rep["boundaries_compared"] > 0  # `S.out` (escaping, whole-circuit consumer scan) of sampled calls
    assert rep["calls_evaluated"] > 0
    # AttentionHead{T}: the zero-stride pad (v1 and post-B.1 v0) against the pre-B.1 explicit single-leaf parts
    sp = bind(b1.AttentionHead, T=5, D=64)
    Q = Program(sp)
    old_style = _explode_broadcasts(encode_program(Q, schema=SCHEMA_V0))
    assert len(json.dumps(old_style)) > 3 * len(json.dumps(encode_program(Q)))
    rep = compare(decode_program(old_style, REGISTRY), _round_trip(Q, SCHEMA_V1), gate_samples=50, call_samples=8)
    assert rep["verdict"] == "equivalent" and rep["boundaries_compared"] > 0


def _explode_broadcasts(desc: dict) -> dict:
    """The pre-B.1 writing of a descriptor: every zero-stride `strided` inside a `concat` becomes `count`
    single-leaf `affine` parts (the same ordered references, one produced value)."""
    def fix(r):
        if "concat" in r:
            out = []
            for p in r["concat"]:
                if "strided" in p and p["strided"][4] == 0 and p["strided"][5] == 0:
                    s, i, b, _, _, _, c = p["strided"]
                    out.extend({"affine": [s, i, b, 1]} for _ in range(c))
                else:
                    out.append(fix(p))
            return {"concat": out}
        if "strided" in r and r["strided"][4] == 0 and r["strided"][5] == 0 and r["strided"][6] > 1:
            s, i, b, _, _, _, c = r["strided"]
            return {"concat": [{"affine": [s, i, b, 1]} for _ in range(c)]}
        return r
    desc = json.loads(json.dumps(desc))
    for d in desc["definitions"].values():
        if d["form"] == "composite":
            for n in d["body"]["nodes"]:
                for a in n["args"]:
                    a["refs"] = fix(a["refs"])
            d["body"]["ret"] = fix(d["body"]["ret"])
    return desc


@composite("EqPair", 1, [], lambda S: ((("x", Array(2, V16)),), V16))
def Pair(B, S, x):
    return B.call(Mul16, x[0], x[1])


@composite("EqPairSwapped", 1, [], lambda S: ((("x", Array(2, V16)),), V16))
def PairSwapped(B, S, x):
    return B.call(Mul16, x[1], x[0])


@composite("EqTop", 1, ["f"], lambda S: ((("x", Array(2, V16)),), V16))
def Top(B, S, x):
    return B.call(Add16, B.call(S.f, x), B.call(Id16, x[0]))


def test_programs_that_differ_are_rejected():
    with pytest.raises(Mismatch, match="definition sets differ"):
        compare(Program(bind(Top, f=Pair)), Program(bind(Top, f=PairSwapped)))
    # same definition ids, different reference order inside one body: the definitions are compared exhaustively
    P, Q = Program(bind(Top, f=Pair)), Program(bind(Top, f=PairSwapped))
    dq = json.loads(json.dumps(encode_program(Q)).replace("EqPairSwapped", "EqPair"))  # Q's bodies under P's ids
    assert dq["definitions"].keys() == encode_program(P)["definitions"].keys()
    with pytest.raises(Mismatch, match="reference"):
        compare(P, decode_program(dq, REGISTRY))
    # a program that is the same everywhere is accepted, whatever its encoding
    assert compare(P, _round_trip(P, SCHEMA_V0))["verdict"] == "equivalent"
