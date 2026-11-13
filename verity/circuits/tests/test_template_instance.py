"""`Q_template_instance` v0 and `Q_template_instances` v0 (`verity.circuits.partition_object.TemplatePartition`): the object, its digest and
the evaluation -- units, their instances' gates, inputs and outputs, the derived committed set -- from descriptor bytes, and every
refusal, against `template_instance_vectors.json`.  Its `--write` rewrites it; `REWRITE` names the command that runs it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.circuits import codec, partition_object as PO
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, DefinitionTable, bind
from verity.circuits.population import population_program
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple, Value

PATH = Path(__file__).parent / "template_instance_vectors.json"
REWRITE = ("the vectors differ from the tree's; a change meant to move them rewrites them with "
           "`uv run python tools/verity/check/pins.py --write verity/circuits/tests/template_instance_vectors.json`")
V32 = Value(32)
M32 = (1 << 32) - 1


def _prim(name, params):
    return PrimitiveDefinition(name, 1, tuple(params), V32, lambda *a: sum(a) & M32, register=False)


def _comp(name, version, statics, sig, body):
    return CompositeDefinition(name, version, tuple(statics), sig, body, register=False)


TAdd = _prim("TAdd32", [("a", V32), ("b", V32)])
TMul = _prim("TMul32", [("a", V32), ("b", V32)])
REG = DefinitionTable()
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
#: x * x
Sq = bind(_comp("TSq", 1, (), lambda S: ((("x", V32),), V32), lambda B, S, x: B.call(TMul, x, x)))


def _canon(program) -> str:
    return codec.canonical_json({k: v for k, v in codec.encode_program(program).items() if k != "annotations"}).decode()


def _keyed(key: str, first: bool = False, nested: bool = False) -> str:
    """The bytes of a program of one instance of `Sq` and one call of a composite of one gate whose id is the input id `key`:
    the call reads the instance and the root returns it; `first`, the instance reads the call; `nested`, the root's one node
    is an instance of `TOuter`, whose body is the call. An input id keys only the input primitive, so the decoder refuses
    each (PROTOCOL.md §4.6)."""
    name, version = key.rsplit("_v", 1)
    keyed = bind(_comp(name, int(version), (), lambda S: ((("x", V32),), V32), lambda B, S, x: B.call(TAdd, x, x)))
    assert codec.descriptor_id(keyed) == key
    outer = bind(_comp("TOuter", 1, (), lambda S: ((("x", V32),), V32), lambda B, S, x: B.call(keyed, x)))

    def body(B, S, x):
        if nested:
            return B.call(outer, x)
        return B.call(Sq, B.call(keyed, x)) if first else B.call(keyed, B.call(Sq, x))
    return _canon(Program(_comp("TKeyedProg", 0, (), lambda S: ((("x", V32),), V32), body)))


def _edited(program: Program, edit) -> str:
    """`program`'s bytes with `edit` applied to its descriptor and its root's one `call` node."""
    d = json.loads(_canon(program))
    call, = (nd for nd in d["definitions"][d["root"]]["body"]["nodes"] if nd["form"] == "call")
    edit(d, call)
    return codec.canonical_json(d).decode()


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
#: one instance of `Sq`, which the undecoded programs edit
SQUARE = Program(_comp("TSqProg", 0, (), lambda S: ((("x", V32),), V32), lambda B, S, x: B.call(Sq, x)))
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
KEYS_ONLY = "an input id keys only the input primitive"
#: (label, the program's bytes, query, a substring of the refusal): the decoder refuses the bytes, so no query reads them
UNDECODED = [
    ("a composite keyed with an input id", _keyed("Input8_v1"), PO.template_instance_query(Sq), KEYS_ONLY),
    ("a composite keyed with an input id, which the instance reads", _keyed("Input8_v1", first=True),
     PO.template_instance_query(Sq), KEYS_ONLY),
    ("a composite keyed with a registered input id", _keyed("RegisteredInput8_v1"), PO.template_instance_query(Sq), KEYS_ONLY),
    ("a composite keyed with an input id that the query lists", _keyed("Input8_v1"),
     PO.template_instances_query([Sq, "Input8_v1"]), KEYS_ONLY),
    ("a composite keyed with an input id inside the instance", _keyed("Input8_v1", nested=True),
     PO.template_instance_query("TOuter_v1"), KEYS_ONLY),
    ("a root call node of an input primitive", _edited(SQUARE, lambda d, call: call.update(fn=d["callees"].index("Input32_v1"))),
     PO.template_instance_query(Sq), "a call node does not call a primitive (Input32_v1)"),
    ("a listed callee without a definition", _edited(SQUARE, lambda d, call: d["callees"].__setitem__(call["fn"], "Input8_v1")),
     PO.template_instance_query("Input8_v1"), "KeyError: 'Input8_v1'"),
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


def _undecoded(text: str) -> str | None:
    try:
        codec.decode_program(json.loads(text), REG)
    except (ValueError, KeyError) as e:
        return f"{type(e).__name__}: {e}"
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
        "about": "Pinned vectors of Q_template_instance v0 and Q_template_instances v0 (verity.circuits.partition_object), checked by "
                 "verity/circuits/tests/test_template_instance.py; written by it with --write.  Programs are descriptor bytes; units are "
                 "[unit, root node, member, first gate, end gate, inputs, outputs]; undecoded holds bytes the decoder refuses, each "
                 "with its refusal.",
        "descriptors": {k: _canon(p) for k, p in PROGRAMS.items()},
        "template_ids": {"TScale descriptor id": SCALE_ID, "TScale readable id": Scale.id, "TDot": DOT_ID},
        "cases": [[label, name, q, _evaluated(progs[name], q)] for label, name, q in CASES],
        "refused": [[label, name, q, _refusal(progs[name], q)] for label, name, q in REFUSED],
        "undecoded": [[label, text, q, _undecoded(text)] for label, text, q, _ in UNDECODED],
        "order_refused": _order_refused(),
        "served_unsupported": list(PO.verify(PO.build(POPULATION, PO.template_instance_query(Scale)), POPULATION, served={}).codes),
        "malformed": [[label, q, list(PO.validate({"format": PO.FORMAT, "program": "0" * 128, "query": q}).codes)] for label, q in MALFORMED],
    }


VECTORS = json.loads(PATH.read_text()) if PATH.exists() else {}


@pytest.fixture(scope="module")
def live() -> dict:
    return build()


def test_the_descriptors_and_ids(live):
    assert live["descriptors"] == VECTORS["descriptors"] and live["template_ids"] == VECTORS["template_ids"], REWRITE
    assert SCALE_ID == 'TScale_v1{N=2,C={"f64":"0x1.8000000000000p+0"}}' and Scale.id == "TScale_v1{N=2,C=1.5}"
    assert PO.template_id(Scale) == PO.template_id(SCALE_ID) == SCALE_ID and PO.template_id(Scale.id) == Scale.id


@pytest.mark.parametrize("n", range(len(CASES)), ids=[c[0] for c in CASES])
def test_the_evaluation_from_the_bytes(live, n):
    assert live["cases"][n] == VECTORS["cases"][n], REWRITE
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
    assert live["refused"][n] == VECTORS["refused"][n], REWRITE
    assert VECTORS["refused"][n][3].startswith("('query-inapplicable',)")


@pytest.mark.parametrize("n", range(len(UNDECODED)), ids=[u[0] for u in UNDECODED])
def test_the_decoder_refuses_the_program(live, n):
    assert live["undecoded"][n] == VECTORS["undecoded"][n], REWRITE
    assert VECTORS["undecoded"][n][3] is not None and UNDECODED[n][3] in VECTORS["undecoded"][n][3]


def test_a_program_that_is_not_topological_and_served(live):
    assert live["order_refused"] == VECTORS["order_refused"] and live["served_unsupported"] == VECTORS["served_unsupported"], REWRITE
    codes, (detail,) = VECTORS["order_refused"]
    assert codes == ["query-inapplicable"] and detail["node"] == 2 and detail["why"] == "not topological"
    assert VECTORS["served_unsupported"] == ["served-unsupported"]
    for name, text in VECTORS["descriptors"].items():                   # every pinned program is topological
        assert PO.order_violation(codec.decode_program(json.loads(text), REG)) is None, name


@pytest.mark.parametrize("n", range(len(MALFORMED)), ids=[m[0] for m in MALFORMED])
def test_the_object_form(live, n):
    assert live["malformed"][n] == VECTORS["malformed"][n], REWRITE
    assert VECTORS["malformed"][n][2] in (["query-params"], ["query-unknown"])


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
