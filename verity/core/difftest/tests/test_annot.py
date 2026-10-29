"""The annotation check's port (`verity_difftest.annot`, `annot-difftest`) on small Programs built here:

* `annot.sets` gives, set for set, the gate intervals `verity.ml.circuits.query_ast` evaluates for the same query, on the
  terms both evaluate, and `annot.term` writes `query_codec`'s document back unchanged;
* `annot.check`, at the size of the description, gives `annot.reference`'s verdict, the meaning decided gate by gate, on
  drawn tables, label maps and meanings, and accepts only what the reference accepts;
* refusals carry their codes.

The Programs: `gemm` (`TGemm{K=32,N=6}`: Input batches, a composite batch, nested calls), `synth` (two zip-able batches, a
constant, a scan) and `twice` (one specialization called twice).
"""

from __future__ import annotations

import json
import shutil

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from verity.ml.circuits import query_ast as G
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_codec import to_json
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity_difftest import annot

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

V16, V32 = Value(16), Value(32)


def prim(name: str, params, ret) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *a: 0, register=False)


Add32 = prim("TAdd32", [("a", V32), ("b", V32)], V32)
Mul32 = prim("TMul32", [("a", V32), ("b", V32)], V32)
Zero32 = prim("TZero32", [], V32)
Dot16 = prim("TDot16", [("acc", V32), ("a", Array(16, V16)), ("b", Array(16, V16))], V32)
Cvt16 = prim("TCvt16", [("a", V32)], V16)


def _dot(Bd, S, x, w):
    acc = Bd.call(Zero32)
    for s in range(S.K // 16):
        acc = Bd.call(Dot16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


Dot = CompositeDefinition("TDot", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.K, V16))), V32), _dot,
                          register=False)
Coord = CompositeDefinition("TGemmCoordinate", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("wrow", Array(S.K, V16))), V16),
                            lambda Bd, S, x, wrow: Bd.call(Cvt16, Bd.call(bind(Dot, K=S.K), x, wrow)), register=False)
Gemm = CompositeDefinition("TGemm", 1, ("K", "N"),
                           lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.N, Array(S.K, V16)))), Array(S.N, V16)),
                           lambda Bd, S, x, w: Bd.batch(bind(Coord, K=S.K), x, w, axes=(None, 0)), register=False)
Step = CompositeDefinition("TScanStep", 1, (), lambda S: ((("carry", V32), ("x", V32)), Tuple(V32, V32)),
                           lambda B, S, carry, x: (lambda c: tuple_of(c, B.call(Mul32, c, x)))(B.call(Add32, carry, x)),
                           register=False)


def _synth(B, S, x):
    s = B.batch(Add32, B.batch(Mul32, x, x, axes=(0, 0)), x, axes=(0, 0))
    return B.scan(Step, B.call(Zero32), xs=[s])


Synth = CompositeDefinition("TSynth", 1, ("N",), lambda S: ((("x", Array(S.N, V32)),), Tuple(V32, Array(S.N, V32))), _synth,
                            register=False)
Twice = CompositeDefinition("TTwice", 1, (), lambda S: ((("x", Array(16, V16)), ("w", Array(16, Array(16, V16)))), Array(16, V16)),
                            lambda B, S, x, w: B.call(bind(Gemm, K=16, N=16), B.call(bind(Gemm, K=16, N=16), x, w), w),
                            register=False)

PROGRAMS = {"gemm": Program(bind(Gemm, K=32, N=6)), "synth": Program(bind(Synth, N=4)), "twice": Program(bind(Twice))}
TEXT = {name: canonical_json(encode_program(p)).decode() for name, p in PROGRAMS.items()}

QUERIES = {
    "gemm": ['default(32)', 'default(16)', 'members("//TGemmCoordinate_v1", "@*")', 'instances("//TDot_v1")',
             'members("//TGemmCoordinate_v1{K=32}", "@*", k=2)', 'members(".", "@*", k=4)', 'nodes(".", "@*")',
             'concat(instances("//TDot_v1"), rest())', 'each("//TGemmCoordinate_v1", concat(instances("@call:TDot_v1"), rest()))',
             'rest(instances("//TDot_v1"))', 'where(default(16), form="call")', 'instances("//TGemm_v1")',
             'members("//TGemm_v1", "@2", k=4)', 'instances("//TDot16_v1")', 'instances("@2/@0")',
             'members("//TDot_v1", "@form=primitive")', 'concat(members("@0", "@*"), rest())', 'root()'],
    "synth": ['default(32)', 'zip(".", ["@batch:TMul32_v1", "@batch:TAdd32_v1"])', 'members(".", "@*")',
              'instances("//TScanStep_v1")', 'members("//TScanStep_v1", "@*")', 'concat(zip(".", ["@1", "@2"]), rest())',
              'nodes(".", "@form=batch")', 'members(".", "@1", k=3)', 'where(members(".", "@*"), w_min=33)',
              'each("//TScanStep_v1", concat(members(".", "@0"), rest()))', 'rest(zip(".", ["@1", "@2"]))',
              'zip(".", ["@0", "@3"])', 'each(zip(".", ["@1", "@2"]), default(8))'],
    "twice": ['default(32)', 'instances("//TGemm_v1")', 'instances("//TGemmCoordinate_v1")', 'members("//TGemm_v1", "@*", k=5)',
              'concat(instances("//TDot_v1"), rest())', 'each("//TGemm_v1", concat(members(".", "@*", k=3), rest()))',
              'instances("//TDot_v1/@1")', 'members("//TGemm_v1/@0//TDot_v1", "@*")'],
}


@pytest.fixture(scope="module")
def server():
    return annot.server()


def _call(server, fn: str, value) -> tuple[str, object]:
    kind, body = server.call(fn, value)
    return kind, (json.loads(body) if kind == "output" else body)


@pytest.mark.parametrize("name,q", [(n, q) for n, qs in QUERIES.items() for q in qs])
def test_sets_are_query_ast_families(server, name: str, q: str) -> None:
    doc = to_json(parse(q))
    assert _call(server, "annot.term", doc) == ("output", doc)
    try:
        want = [[list(iv) for iv in s.intervals()] for s in G.evaluate(PROGRAMS[name], parse(q))]
    except G.QueryError as e:
        want = e
    kind, got = _call(server, "annot.sets", annot.instance(TEXT[name], doc))
    if isinstance(want, G.QueryError):
        assert kind == "reject" and got.startswith("plan:"), got
    else:
        assert kind == "output" and got["sets"] == want


def _root_nodes(name: str) -> int:
    return len(PROGRAMS[name].root.body.nodes)


@st.composite
def _instances(draw):
    """A program, a term over its root body (a table of its nodes, or one of `QUERIES`), maybe a label map, and a meaning."""
    name = draw(st.sampled_from(sorted(PROGRAMS)))
    tables = None
    if draw(st.booleans()):
        n = _root_nodes(name)
        labels = draw(st.integers(1, n))
        rows = draw(st.lists(st.one_of(st.none(), st.integers(0, labels - 1)), min_size=n, max_size=n))
        query = {"op": "table", "path": ".", "table": "t"}
        tables = {"t": {json.loads(TEXT[name])["root"]: rows}}
    else:
        query = to_json(parse(draw(st.sampled_from(QUERIES[name]))))
    match draw(st.sampled_from(["none", "affine", "definition", "name"])):
        case "affine":
            query = {"op": "label", "base": query, "map": {"affine": [draw(st.integers(0, 5)), draw(st.integers(0, 5)),
                                                                      draw(st.integers(1, 6))]}}
        case "definition" | "name" as key:
            query = {"op": "group", "base": query, "key": key}
    meaning = draw(st.fixed_dictionaries({}, optional={"tiling": st.sampled_from(["partition", "cover"]),
                                                       "max_count": st.integers(0, 300), "max_gates": st.integers(0, 300)}))
    return annot.instance(TEXT[name], query, tables, meaning)


@settings(max_examples=150, derandomize=True, database=None, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_instances())
def test_check_is_the_reference(server, inst) -> None:
    """The check's verdict is the reference's, but where the check bounds a group's gates by the sum of its sets' sizes:
    there it may refuse what the reference accepts, never the converse."""
    check = _call(server, "annot.check", inst)
    ref = _call(server, "annot.reference", inst)
    q = inst["annotation"]["query"]
    if q["op"] == "group" and "max_gates" in inst["meaning"] and inst["meaning"].get("tiling") != "partition":
        if check[0] == "output" and check[1]["accept"]:
            assert ref == check
        return
    assert check[0] == ref[0] and (check == ref or check[0] == "reject"), (check, ref)


def test_refusals_carry_codes(server) -> None:
    gemm, synth = TEXT["gemm"], TEXT["synth"]
    key = json.loads(synth)["root"]
    cases = [
        (synth, to_json(parse('zip(".", ["@0", "@3"])')), None, "plan:"),
        (synth, {"op": "frontier", "w": 32}, None, "term:"),
        (synth, {"op": "members", "path": ".", "sel": "@*", "k": [2, 2]}, None, "term:"),
        (synth, {"op": "members", "path": ".", "sel": "@*", "k": 0}, None, "term:"),
        (synth, {"op": "table", "path": ".", "table": "t"}, {"t": {key: [0, None]}}, "table:"),
        (synth, {"op": "table", "path": ".", "table": "t"}, {"t": {key: [0, 2, None, None, None]}}, "table:"),
        (synth, {"op": "table", "path": ".", "table": "missing"}, None, "table:"),
        (gemm, {"op": "each", "base": {"op": "instances", "path": "//TDot_v1"},
                "inner": {"op": "label", "base": {"op": "default", "w": 8}, "map": {"affine": [1, 0, 2]}}}, None, "term:"),
        (gemm, {"op": "group", "base": to_json(parse('zip("//TGemm_v1", ["@2"])')), "key": "definition"}, None, "term:"),
        (gemm, {"op": "label", "base": {"op": "default", "w": 8}, "map": {"affine": [1, 0, 0]}}, None, "term:"),
    ]
    for program, query, tables, code in cases:
        for fn in ("annot.check", "annot.reference"):
            kind, why = _call(server, fn, annot.instance(program, query, tables, {"tiling": "cover"}))
            assert kind == "reject" and why.startswith(code), (fn, query, why)
    kind, why = _call(server, "annot.check", annot.instance(
        gemm, {"op": "label", "base": {"op": "default", "w": 8}, "map": {"affine": [1, 0, 3]}}, None, {"max_gates": 5}))
    assert kind == "reject" and why.startswith("meaning-unsupported:")


def test_partition_reads_inputs_as_held_by_none(server) -> None:
    """`default(w)` puts the root's Input nodes in sets, so it covers but doesn't partition; its computed part does."""
    text = TEXT["gemm"]
    whole = to_json(parse("default(32)"))
    computed = to_json(parse('instances("@2")'))

    def verdict(q, t):
        return _call(server, "annot.check", annot.instance(text, q, None, {"tiling": t}))[1]

    assert verdict(whole, "cover")["accept"]
    assert verdict(whole, "partition")["code"] == "input-held"
    assert verdict(computed, "partition") == {"accept": True, "count": 6}
    twice = to_json(parse('concat(instances("@2"), instances("//TDot_v1"))'))
    assert verdict(twice, "partition")["code"] == "overlap"
    assert verdict(twice, "cover") == {"accept": True, "count": 12}
