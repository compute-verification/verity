"""`Q_template_instance` v0 and `Q_template_instances` v0 (`verity.ml.circuits.partition_object.TemplatePartition`): the object, its digest and
the evaluation -- units, their instances' gates, inputs and outputs, the derived committed set -- from descriptor bytes, and every
refusal, against `template_instance_vectors.json`.  `python verity/ml/circuits/tests/test_template_instance.py --write` rewrites it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, Registry, bind
from verity.ml.circuits.population import population_program
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

PATH = Path(__file__).parent / "template_instance_vectors.json"
V32 = Value(32)
M32 = (1 << 32) - 1


def _prim(name, params):
    return PrimitiveDefinition(name, 1, tuple(params), V32, lambda *a: sum(a) & M32, register=False)


def _comp(name, version, statics, sig, body):
    return CompositeDefinition(name, version, tuple(statics), sig, body, register=False)


TAdd = _prim("TAdd32", [("a", V32), ("b", V32)])
TMul = _prim("TMul32", [("a", V32), ("b", V32)])
REG = Registry()
for _p in (TAdd, TMul):
    REG.add(_p)

#: x * k, elementwise: a float static, so its descriptor id is not its readable id
ScaleDef = _comp("TScale", 1, ("N", "C"), lambda S: ((("x", Array(S.N, V32)), ("k", V32)), Array(S.N, V32)),
                 lambda B, S, x, k: B.batch(TMul, x, k, axes=(0, None)))
Scale = bind(ScaleDef, N=2, C=1.5)
#: a0 * b0 + a1 * b1
Dot = bind(_comp("TDot", 1, (), lambda S: ((("a", Array(2, V32)), ("b", Array(2, V32))), V32),
                 lambda B, S, a, b: B.call(TAdd, B.call(TMul, a[0], b[0]), B.call(TMul, a[1], b[1]))))
#: (c + x, c + x): a scan step
Step = bind(_comp("TStep", 1, (), lambda S: ((("c", V32), ("x", V32)), Tuple(V32, V32)),
                  lambda B, S, c, x: (lambda s: tuple_of(s, s))(B.call(TAdd, c, x))))
X3 = Array(3, Array(2, V32))


def _mixed(B, S, xs, k, a, b):
    s = B.batch(Scale, xs, k, axes=(0, None))                        # 3 instances of Scale, k shared
    d0 = B.call(Dot, a[0], b[0])                                     # 1 instance of Dot
    ds = B.batch(Dot, a[1:3], b[1:3], axes=(0, 0))                   # 2 instances of Dot
    d3 = B.call(Dot, s[0], b[0])                                     # 1 instance of Dot reading Scale #0's result: a value across units
    return tuple_of(s, d0, ds, d3)


MIXED = Program(_comp("TMix", 0, (), lambda S: ((("xs", X3), ("k", V32), ("a", X3), ("b", X3)), Tuple(X3, V32, Array(2, V32), V32)),
                      _mixed))
POPULATION = population_program(Scale, 3, version=0)  # the pinned vectors' version
SCAN = Program(_comp("TScanProg", 0, (), lambda S: ((("xs", Array(3, V32)), ("c", V32)), Tuple(V32, Array(3, V32))),
                     lambda B, S, xs, c: B.scan(Step, c, xs=[xs])))
PRIM = Program(_comp("TPrimProg", 0, (), lambda S: ((("x", Array(2, V32)),), V32), lambda B, S, x: B.call(TAdd, x[0], x[1])))
EMPTY = Program(_comp("TEmptyProg", 0, (), lambda S: ((("a", Array(0, Array(2, V32))), ("b", Array(0, Array(2, V32)))), Array(0, V32)),
                      lambda B, S, a, b: B.batch(Dot, a, b, axes=(0, 0))))
PROGRAMS = {"population": POPULATION, "mixed": MIXED, "scan": SCAN, "primitive": PRIM, "empty": EMPTY}
SCALE_ID, DOT_ID = codec.descriptor_id(Scale), codec.descriptor_id(Dot)

#: (label, program, query): the query applies
CASES = [
    ("Q_template_instance over a population program", "population", PO.template_instance_query(Scale)),
    ("Q_template_instances over two templates", "mixed", PO.template_instances_query([Scale, Dot])),
    ("the list order is kept: the same units, another digest", "mixed", PO.template_instances_query([Dot, Scale])),
]
#: (label, program, query): the query refuses the program
REFUSED = [
    ("the readable id is not the descriptor id", "population", PO.template_instance_query(Scale.id)),
    ("a node of another function", "mixed", PO.template_instance_query(Scale)),
    ("a listed template without an instance", "mixed", PO.template_instances_query([Scale, Dot, "TOther_v1"])),
    ("a scan at the root", "scan", PO.template_instance_query(Step)),
    ("a primitive at the root", "primitive", PO.template_instance_query(TAdd)),
    ("a batch of 0 members is no instance", "empty", PO.template_instance_query(Dot)),
]
#: (label, query): the object's form is refused (`validate`: query-params or query-unknown)
MALFORMED = [
    ("a template that is not a string", {"name": "Q_template_instance", "version": 0, "params": {"template": 7}}),
    ("templates for Q_template_instance", {"name": "Q_template_instance", "version": 0, "params": {"templates": [DOT_ID]}}),
    ("an empty list", {"name": "Q_template_instances", "version": 0, "params": {"templates": []}}),
    ("a template listed twice", {"name": "Q_template_instances", "version": 0, "params": {"templates": [DOT_ID, DOT_ID]}}),
    ("a string for the list", {"name": "Q_template_instances", "version": 0, "params": {"templates": DOT_ID}}),
    ("an extra parameter", {"name": "Q_template_instance", "version": 0, "params": {"template": DOT_ID, "X": 16}}),
    ("a version not registered", {"name": "Q_template_instance", "version": 1, "params": {"template": DOT_ID}}),
]


def _canon(program) -> str:
    return codec.canonical_json({k: v for k, v in codec.encode_program(program).items() if k != "annotations"}).decode()


def _evaluated(program, q) -> dict:
    obj = PO.build(program, q)
    part = PO.evaluate(program, q)
    return {"object": obj, "digest": PO.digest(obj), "population": part.population, "ranges": [list(r) for r in part.ranges],
            "ranges_digest": PO.ranges_digest(part), "committed": part.committed(), "verify": PO.verify(obj, program).ok,
            "units": [[i, *part.locate(i), *part.gates(i), part.inputs(i), part.outputs(i)] for i in range(part.population)]}


def _refusal(program, q) -> str | None:
    try:
        PO.evaluate(program, q)
    except PO.QueryRefused as e:
        v = PO.verify(PO.build(program, q), program)
        return f"{v.codes} {e}"
    return None


def _order_refused() -> list:
    """The population program whose instances read their own outputs: not topological, so no query applies."""
    d = json.loads(_canon(POPULATION))
    d["definitions"][d["root"]]["body"]["nodes"][2]["args"][0]["refs"] = {"affine": ["n", 2, 0, 6]}
    D = codec.decode_program(d, REG)
    v = PO.verify(PO.build(D, PO.template_instance_query(Scale)), D)
    return [list(v.codes), v.detail["query-inapplicable"]["gates"]]


def build(decode: bool = True) -> dict:
    progs = {k: codec.decode_program(json.loads(_canon(p)), REG) if decode else p for k, p in PROGRAMS.items()}
    return {
        "about": "Pinned vectors of Q_template_instance v0 and Q_template_instances v0 (verity.ml.circuits.partition_object), checked by "
                 "verity/ml/circuits/tests/test_template_instance.py; written by it with --write.  Programs are descriptor bytes; units are "
                 "[unit, root node, member, first gate, end gate, inputs, outputs].",
        "descriptors": {k: _canon(p) for k, p in PROGRAMS.items()},
        "template_ids": {"TScale descriptor id": SCALE_ID, "TScale readable id": Scale.id, "TDot": DOT_ID},
        "cases": [[label, name, q, _evaluated(progs[name], q)] for label, name, q in CASES],
        "refused": [[label, name, q, _refusal(progs[name], q)] for label, name, q in REFUSED],
        "order_refused": _order_refused(),
        "served_unsupported": list(PO.verify(PO.build(POPULATION, PO.template_instance_query(Scale)), POPULATION, served={}).codes),
        "malformed": [[label, q, list(PO.validate({"format": PO.FORMAT, "program": "0" * 128, "query": q}).codes)] for label, q in MALFORMED],
    }


VECTORS = json.loads(PATH.read_text()) if PATH.exists() else {}


@pytest.fixture(scope="module")
def live() -> dict:
    return build()


def test_the_descriptors_and_ids(live):
    assert live["descriptors"] == VECTORS["descriptors"] and live["template_ids"] == VECTORS["template_ids"]
    assert SCALE_ID == 'TScale_v1{N=2,C={"f64":"0x1.8000000000000p+0"}}' and Scale.id == "TScale_v1{N=2,C=1.5}"
    assert PO.template_id(Scale) == PO.template_id(SCALE_ID) == SCALE_ID and PO.template_id(Scale.id) == Scale.id


@pytest.mark.parametrize("n", range(len(CASES)), ids=[c[0] for c in CASES])
def test_the_evaluation_from_the_bytes(live, n):
    assert live["cases"][n] == VECTORS["cases"][n]
    label, name, q, want = VECTORS["cases"][n]
    assert PO.validate(want["object"]).ok and want["verify"]
    assert _evaluated(PROGRAMS[name], q) == want                                       # the built program agrees with its bytes
    assert want["population"] == sum(c for *_, c in want["ranges"]) == len(want["units"])
    tiles = sorted((lo, hi) for _i, _k, _m, lo, hi, _in, _out in want["units"])
    D = codec.decode_program(json.loads(VECTORS["descriptors"][name]), REG)
    first = min(D.root.body.offsets[k] for k, *_ in want["ranges"])
    assert tiles[0][0] == first and all(a[1] == b[0] for a, b in zip(tiles, tiles[1:])) and tiles[-1][1] == D.gates


def test_the_mixed_program_commits_the_value_across_units():
    _, _, _, mixed = VECTORS["cases"][1]
    units = mixed["units"]
    s0_out, d3_in = units[0][6], units[6][5]
    assert set(s0_out) <= set(d3_in) and set(s0_out) <= set(mixed["committed"])
    assert VECTORS["cases"][2][3]["ranges"] == mixed["ranges"] and VECTORS["cases"][2][3]["digest"] != mixed["digest"]


@pytest.mark.parametrize("n", range(len(REFUSED)), ids=[r[0] for r in REFUSED])
def test_the_query_refuses(live, n):
    assert live["refused"][n] == VECTORS["refused"][n] and VECTORS["refused"][n][3].startswith("('query-inapplicable',)")


def test_a_program_that_is_not_topological_and_served(live):
    assert live["order_refused"] == VECTORS["order_refused"] and live["served_unsupported"] == VECTORS["served_unsupported"]
    codes, (detail,) = VECTORS["order_refused"]
    assert codes == ["query-inapplicable"] and detail["node"] == 2 and detail["why"] == "not topological"
    assert VECTORS["served_unsupported"] == ["served-unsupported"]
    for name, text in VECTORS["descriptors"].items():                   # every pinned program is topological
        assert PO.order_violation(codec.decode_program(json.loads(text), REG)) is None, name


@pytest.mark.parametrize("n", range(len(MALFORMED)), ids=[m[0] for m in MALFORMED])
def test_the_object_form(live, n):
    assert live["malformed"][n] == VECTORS["malformed"][n] and VECTORS["malformed"][n][2] in (["query-params"], ["query-unknown"])


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
