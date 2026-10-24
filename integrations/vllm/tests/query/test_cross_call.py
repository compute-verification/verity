"""Cross-Call recomputes (`verity_vllm.query.cross_call`): a Call computed twice, a subset of a Call's gates computed twice by
another Call of its spec, one gate computed in two Calls of different specs, each caught and named; structure (values from constants
alone) not counted; the honest synthetic Program passes; the hierarchical support analysis equals the gate-by-gate one."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from verity.ml.circuits.cut import separable
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.registry import prims as P
from verity_vllm.query import cross_call as X
from verity_vllm.query import word as W

F32 = P.F32


def _honest():
    from tests.check.test_sampled_replay import _program
    inst, _w = _program(attn=True, scale=True)
    return inst


def _append(inst: dict, rows: list[dict]) -> dict:
    doc = copy.deepcopy(inst)
    n = len(doc["rows"])
    for k, r in enumerate(rows):
        doc["rows"].append(dict(r, i=n + k))
    return doc


def test_the_honest_program_computes_every_value_in_one_call():
    r = X.check_program(_honest())
    assert r["ok"] and r["recomputes"] == [] and r["recomputed_gates"] == 0 and r["unrefined"] == {}


def test_a_call_computed_twice_is_caught_and_named():
    inst = _honest()
    norm = next(r for r in inst["rows"] if r["spec"].startswith("RMSNormTriton"))
    gemm = next(r for r in inst["rows"] if r["spec"].startswith("Gemm") and r["args"][0][0][:2] == ["n", norm["i"]])
    n = len(inst["rows"])
    copy_gemm = dict(gemm, name=gemm["name"] + "/copy", args=[[["n", n, *gemm["args"][0][0][2:]]]] + gemm["args"][1:])
    doc = _append(inst, [dict(norm, name=norm["name"] + "/copy"), copy_gemm])     # the Gemm reads the copy: computed again too
    r = X.check_program(doc, strict=False)
    got = {a["definition"].split("{")[0]: a for a in r["recomputes"]}
    assert set(got) == {"RMSNormTriton_v1", "Gemm_v1"} and all(a["level"] == "call" and a["calls"] == 1 for a in got.values())
    assert got["RMSNormTriton_v1"]["examples"] == [[norm["name"] + "/copy", norm["name"]]]
    assert got["Gemm_v1"]["gates"] == X.analyze(W.specialization("Gemm_v1", {"K": 16, "N": 80})).total > 0
    with pytest.raises(X.RecomputeViolation, match="RMSNormTriton_v1.*" + norm["name"].replace("[", r"\[").replace("]", r"\]")):
        X.check_program(doc)


def _defs():
    """`Scaled(w, x)` = (w0 + w1) * x0: the sum depends on w alone.  `Sum2(v, y)` / `Diff2(v, y)`: v0 + v1, then times / minus y0."""
    def scaled(B, S, w, x):
        return B.call(P.F32Mul, B.call(P.F32Add, w[0], w[1]), x[0])

    def sum2(B, S, v, y):
        return B.call(P.F32Mul, B.call(P.F32Add, v[0], v[1]), y[0])

    def diff2(B, S, v, y):
        s = B.call(P.F32Add, v[0], v[1])
        return tuple_of(B.call(P.F32Mul, s, s), B.call(P.F32Add, s, y[0]))

    two = (("w", Array(2, F32)), ("x", Array(1, F32)))
    return {"Scaled": bind(CompositeDefinition("Scaled", 1, (), lambda S: (two, F32), scaled, register=False)),
            "Sum2": bind(CompositeDefinition("Sum2", 1, (), lambda S: (two, F32), sum2, register=False)),
            "Diff2": bind(CompositeDefinition("Diff2", 1, (), lambda S: (two, Tuple(F32, F32)), diff2, register=False))}


def _rows(*calls):
    return [{"i": i, "spec": s, "form": "call", "name": f"m{i}/{s}", "args": a} for i, (s, a) in enumerate(calls)]


def test_gates_that_read_only_a_shared_operand_are_caught_in_every_further_call_of_the_spec():
    D = _defs()
    rows = _rows(("Scaled", [[["p", 0, 0, 2]], [["p", 1, 0, 1]]]), ("Scaled", [[["p", 0, 0, 2]], [["p", 1, 1, 2]]]),
                 ("Scaled", [[["p", 0, 0, 2]], [["p", 1, 2, 3]]]))
    assert X.analyze(D["Scaled"]).gates == {1: 1, 3: 1}
    r = X.check_program(rows, strict=False, specialize=lambda f, s: D[f])
    (a,) = r["recomputes"]
    assert (a["level"], a["calls"], a["gates"], a["originals"]) == ("operands", 2, 2, 1)
    assert a["examples"][0] == ["m1/Scaled", "m0/Scaled"]


def test_one_gate_computed_in_calls_of_two_specs_is_caught_at_gate_level():
    D = _defs()
    rows = _rows(("Sum2", [[["p", 0, 0, 2]], [["p", 1, 0, 1]]]), ("Diff2", [[["p", 0, 0, 2]], [["p", 1, 1, 2]]]))
    r = X.check_program(rows, strict=False, specialize=lambda f, s: D[f])
    (a,) = r["recomputes"]
    assert (a["definition"], a["level"], a["calls"], a["gates"]) == ("Diff2", "gates", 1, 1)
    assert r["refined"]["calls"] == 2
    ok = X.check_program(_rows(("Sum2", [[["p", 0, 0, 2]], [["p", 1, 0, 1]]]), ("Diff2", [[["p", 0, 1, 3]], [["p", 1, 1, 2]]])),
                         specialize=lambda f, s: D[f])
    assert ok["ok"] and ok["refined"]["calls"] == 2                                           # v overlaps in one leaf: no common sum


def test_values_from_constants_alone_are_structure():
    D = _defs()
    rows = _rows(("Scaled", []), ("Scaled", []), ("Scaled", [[["n", 0, 0, 1], ["n", 1, 0, 1]], [["n", 0, 0, 1]]]),
                 ("Scaled", [[["n", 0, 0, 1], ["n", 1, 0, 1]], [["n", 0, 0, 1]]]))
    r = X.check_program(rows, specialize=lambda f, s: D[f])
    assert r["ok"] and r["structure_calls"] == 4


@pytest.mark.parametrize("fam,statics", [("Attention_v3", {"T": 150, "NH": 4, "KVH": 2, "D": 64, "BN": 128}), ("RoPE_v1", {"NHEADS": 3, "D": 16}),
                                         ("Gemm_v1", {"K": 64, "N": 8}), ("TokenSelect_v1", {"V": 64}), ("RMSNormTriton_v1", {"N": 64, "EPS": 1e-05}),
                                         ("Embedding_v1", {"V": 64, "H": 16})])
def test_the_hierarchical_analysis_is_the_gate_by_gate_one(fam, statics):
    fn = W.specialization(fam, statics)
    a = X.analyze(fn, {})
    b = X._expand(fn, np.repeat(np.array([1 << k for k in range(len(X._params(fn)))], np.int64), [int(t.leaves) for _, t in X._params(fn)]))
    assert a.gates == b.gates and a.prims == b.prims and np.array_equal(a.out, b.out)
    assert a.total == W.Graph(fn).n


def test_the_manifest_build_runs_the_check(tmp_path):
    import gzip
    import json
    from verity_vllm.pipeline import manifest as M
    inst = _honest()
    (tmp_path / "instances.json.gz").write_bytes(gzip.compress(json.dumps(inst).encode()))
    assert M.cross_call_check(str(tmp_path))["ok"]
    (tmp_path / "instances.json.gz").write_bytes(gzip.compress(json.dumps(_append(inst, [dict(inst["rows"][2], name="dup")])).encode()))
    with pytest.raises(X.RecomputeViolation, match="dup"):
        M.cross_call_check(str(tmp_path))
    assert M.BuildOptions.__dataclass_fields__["cross_call_check"].default is False            # opt-in


def test_members_of_a_separable_definition_are_compared():
    # word.unit_rule cuts each member of a body that reads only its parameters on its own: the FP8 block GEMM's coordinates each
    # compute their block's scale product sx[kb] * sw[kb], 127 copies per (weight block, kb)
    fp8 = W.specialization("ScaledMmFp8Block_v1", {"K": 256, "N": 256, "G": 128})
    (v,) = [v for v in W.unit_rule(fp8)["violations"] if v["class"] == "cut"]
    assert v["codes"] == ["gate-recomputed"] and v["detail"]["gate-recomputed"]["n"] == 2 * 2 * 127
    (a,) = v["detail"]["gate-recomputed"]["classes"]
    assert a["level"] == "operands" and a["definition"].startswith("ScaledMmFp8BlockCoordinate") and a["calls"] == 2 * 127
    with pytest.raises(W.QueryRuleViolation, match="ScaledMmFp8Block_v1.*cut"):
        W.check_calls([{"family": "ScaledMmFp8Block_v1", "statics": {"K": 256, "N": 256, "G": 128}, "definition": fp8.id, "calls": 1}])

    def twice(B, S, x):
        return tuple_of(B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1]))

    f = bind(CompositeDefinition("TwiceSeparable", 1, (), lambda S: ((("x", Array(2, F32)),), Tuple(F32, F32)), twice, register=False))
    assert separable(f) is not None
    (v,) = W.unit_rule(f)["violations"]
    assert v["class"] == "cut" and v["detail"]["gate-recomputed"]["n"] == 1
    for fam, st in (("Gemm_v1", {"K": 64, "N": 8}), ("Attention_v3", {"T": 21, "NH": 4, "KVH": 2, "D": 64, "BN": 128}), ("RoPE_v1", {"NHEADS": 3, "D": 16})):
        assert W.unit_rule(W.specialization(fam, st))["violations"] == []
