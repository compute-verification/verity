"""`word.call_groups` on a columnar body (its inputs-committed test as numpy passes over the read columns) groups the Calls exactly as
over the same Calls held as a tuple, whatever is committed and whatever the numpy step."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from verity_vllm.query.units import ValueRef
from verity.ml.definitions import word as W
from verity_vllm.query.module_body import literal_calls
from verity_vllm.query.program_view import CallView, ProgramView, from_instances

PARAMS = [["prompt", "Array<4,Value<32>>"], ["weights", "{m.a.weight: Array<4,Value<16>>, m.b.weight: Array<4,Value<16>>}"]]
EPS = '{"f64":"0x1p-17"}'


def rows(n: int, seed: int) -> list[dict]:
    """A fused norm with two ports (Call 1), then `n` Calls that each read a random set of earlier Calls (either norm port), a weight,
    the prompt, or nothing (a literal)."""
    rnd = random.Random(seed)
    out = [{"i": 0, "spec": "SiluMul_v1{I=4}", "form": "call", "name": "m.e/x[0]", "args": [[["p", 0, 0, 1]]]},
           {"i": 1, "spec": f"RMSNormFusedCuda_v2{{N=4,EPS={EPS}}}", "form": "call", "name": "m.n/fused[0]",
            "args": [[["n", 0, 0, 4]], [["n", 0, 0, 4]], [["p", 1, 0, 4]]]}]
    for i in range(2, 2 + n):
        if rnd.random() < 0.2:
            args = []
        else:
            args = [[["n", j, 0, 4]] if j != 1 else [["n", 1, *rnd.choice([(0, 4), (4, 8), (2, 6)])]] for j in rnd.sample(range(i), min(i, rnd.randint(1, 10)))]
            if rnd.random() < 0.3:
                args.append([["p", 1, *rnd.choice([(0, 4), (4, 8)])]])
        out.append({"i": i, "spec": "SiluMul_v1{I=4}", "form": "call", "name": f"m.l{rnd.randrange(4)}/x[{i}]", "args": args})
    return out


def population(P: ProgramView, committed: frozenset) -> SimpleNamespace:
    return SimpleNamespace(required=SimpleNamespace(values=committed), literals=literal_calls(P),
                           program_path=lambda cid: P.call(cid).name.split("/")[0])


@pytest.mark.parametrize("step", [1, 3, 1 << 24])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_columnar_call_groups_are_the_tuple_bodys(monkeypatch, step, seed):
    monkeypatch.setattr(W, "_READS_PER_STEP", step)
    P = from_instances({"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": PARAMS, "rows": rows(50, seed)})
    T = ProgramView(tuple(CallView(c.id, c.definition, c.family, c.form, c.name, c.port_table, tuple(c.reads)) for c in P.calls),
                    P.params, P.outputs, P.digest, P.source)
    assert P.columns is not None and T.columns is None
    read = sorted({v for c in P.calls for v in c.reads})
    rnd = random.Random(seed)
    for share in (0.0, 0.5, 0.9, 1.0):
        committed = frozenset(v for v in read if rnd.random() < share) | {ValueRef(0, "no-such-port"), ValueRef(-9, "out")}
        Q = population(P, committed)
        got = W._inputs_committed(P, committed, Q.literals)
        assert got == [all(r in committed or r.call in Q.literals for r in c.reads) for c in P.calls]
        assert W.call_groups(P, Q) == W.call_groups(T, Q)
    assert W._inputs_committed(T, committed, Q.literals) is None
    assert all(got), "every Value read is committed: every Call's inputs are"
    some = frozenset(read[: len(read) // 2])
    flags = W._inputs_committed(P, some, Q.literals)
    assert True in flags and False in flags, "a half-committed body has Calls of both kinds"
