"""The graph/reference correspondence test across codec transitions (`tests.program.descriptor_equivalence`,
descriptor-codec.md §C): representation-independent reference runs, an exhaustive definition comparison, and
sampled gates / calls / boundaries / evaluation. It must accept the same program under different encodings and
reject programs that differ in gates, producers or reference order."""

from __future__ import annotations

import json
import random

import pytest
from verity.circuits import (
    REGISTRY,
    Array,
    Program,
    Value,
    bind,
    composite,
    decode_program,
    encode_program,
)
from verity.circuits.refs import Affine, Concat, Explicit, Strided, concat

from tests.program.descriptor_equivalence import (
    Mismatch,
    canonical_runs,
    canonical_runs_by_leaf,
    compare,
)
from tests.program.test_running_example import Add16, Id16, Mul16

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


def _round_trip(P):
    return decode_program(json.loads(json.dumps(encode_program(P))), REGISTRY)


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
    assert compare(P, _round_trip(P))["verdict"] == "equivalent"
