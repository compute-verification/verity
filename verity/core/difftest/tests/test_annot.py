"""The annotation check's port (`verity_difftest.annot`, `annot-difftest`) on small Programs built here:

* `annot.sets` gives, set for set, the gate intervals `verity.ml.circuits.query_ast` evaluates for the same query, on the
  terms both evaluate, `table`, `group` (every key part) and `label` (affine or by a table) among them, with the same
  tables, and `annot.term` writes `query_codec`'s document back as `query_codec` does;
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
from verity.ml.circuits.codec import canonical_json, descriptor_id, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_codec import from_json, to_json
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


def _specs(program: Program) -> dict:
    """Every Definition with a body the program reaches, by descriptor id."""
    out: dict = {}

    def go(spec) -> None:
        if descriptor_id(spec) not in out:
            out[descriptor_id(spec)] = spec
            for node in spec.body.nodes:
                if not isinstance(node.fn, PrimitiveDefinition):
                    go(node.fn)
    go(program.root)
    return out


def _row(n: int, mod: int, skip: int) -> list:
    """A dense row over n nodes: node j gets j mod `mod`, none when j mod `skip` is skip - 1, labels renumbered."""
    raw = [None if j % skip == skip - 1 else j % mod for j in range(n)]
    order = {v: i for i, v in enumerate(dict.fromkeys(v for v in raw if v is not None))}
    return [None if v is None else order[v] for v in raw]


def _rows(name: str, spec_name: str, mod: int, skip: int) -> dict:
    """Rows for every Definition whose id starts with `spec_name` (the root's when it is "")."""
    root = descriptor_id(PROGRAMS[name].root)
    return {k: _row(len(s.body.nodes), mod, skip) for k, s in _specs(PROGRAMS[name]).items()
            if (k.startswith(spec_name) if spec_name else k == root)}


def _group(base: str, key) -> dict:
    return {"op": "group", "base": to_json(parse(base)), "key": key}


def _label(base: str, t: str) -> dict:
    return {"op": "label", "base": to_json(parse(base)), "map": {"table": t}}


def _table(path: str, t: str, **of) -> dict:
    return {"op": "table", "path": path, "table": t, **of}


OPS = [
    *[("gemm", _group('members("//TGemmCoordinate_v1", "@*")', key), None)
      for key in ["definition", "name", "base", "static:K", "static:N", ["name", "static:K"], [], ["definition"]]],
    ("gemm", _group("default(16)", "definition"), None),
    ("gemm", _group('instances("//TDot16_v1")', ["base", "static:K"]), None),
    ("gemm", _group('members(".", "@*")', "name"), None),
    ("synth", _group('members(".", "@*")', "base"), None),
    ("synth", _group('members("//TScanStep_v1", "@*")', "definition"), None),
    ("synth", _group('concat(zip(".", ["@1", "@2"]), rest())', "definition"), None),
    ("twice", _group('members("//TGemm_v1", "@*", k=5)', ["static:K", "name"]), None),
    ("twice", _group('each("//TGemm_v1", concat(members(".", "@*", k=3), rest()))', "definition"), None),
    *[(name, _label(base, "u"), lambda name, m=m, s=s: {"u": _rows(name, "", m, s)})
      for name, base in [("gemm", 'members(".", "@*")'), ("gemm", 'nodes(".", "@*")'), ("gemm", "default(16)"),
                         ("synth", 'members(".", "@*")'), ("synth", 'zip(".", ["@1", "@2"])'),
                         ("synth", 'concat(zip(".", ["@1", "@2"]), rest())'), ("twice", 'instances("//TDot_v1")')]
      for m, s in [(2, 4), (3, 3), (1, 7)]],
    *[(name, _table(path, "t", **of), lambda name, spec=spec, m=m: {"t": _rows(name, spec, m, 3)})
      for name, path, spec in [("gemm", ".", ""), ("gemm", "//TGemmCoordinate_v1", "TGemmCoordinate_v1"),
                               ("gemm", "//TDot_v1", "TDot_v1"), ("synth", ".", ""), ("twice", "//TGemm_v1", "TGemm_v1")]
      for m in (1, 2) for of in ({}, {"of": "nodes"})],
    ("gemm", {"op": "where", "base": _table(".", "t"), "pred": {"form": "batch"}}, lambda name: {"t": _rows(name, "", 2, 3)}),
]


@pytest.mark.parametrize("name,doc,tables", OPS)
def test_table_group_label_are_query_ast_families(server, name: str, doc: dict, tables) -> None:
    """`table`, `group` (each key part, one or several or none) and `label(F, table=NAME)`: the gates each label holds
    are `query_ast`'s sets, and a term `query_ast` refuses is refused."""
    tables = tables(name) if tables else None
    q = from_json(doc)
    assert _call(server, "annot.term", doc) == ("output", to_json(q))
    try:
        want = [[list(iv) for iv in s.intervals()] for s in G.evaluate(PROGRAMS[name], q, tables)]
    except G.QueryError as e:
        want = e
    kind, got = _call(server, "annot.sets", annot.instance(TEXT[name], doc, tables))
    if isinstance(want, G.QueryError):
        assert kind == "reject" and got.startswith("plan:"), got
    else:
        assert kind == "output" and got["labels"] == want, got


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
    match draw(st.sampled_from(["none", "affine", "group", "table"])):
        case "affine":
            query = {"op": "label", "base": query, "map": {"affine": [draw(st.integers(0, 5)), draw(st.integers(0, 5)),
                                                                      draw(st.integers(1, 6))]}}
        case "group":
            parts = st.sampled_from(["definition", "name", "base", "static:K", "static:N"])
            query = {"op": "group", "base": query, "key": draw(st.one_of(parts, st.lists(parts, max_size=2)))}
        case "table":
            n = _root_nodes(name)
            labels = draw(st.integers(1, n))
            rows = draw(st.lists(st.one_of(st.none(), st.integers(0, labels - 1)), min_size=n, max_size=n))
            order = {v: i for i, v in enumerate(dict.fromkeys(v for v in rows if v is not None))}
            query = {"op": "label", "base": query, "map": {"table": "u"}}
            tables = (tables or {}) | {"u": {json.loads(TEXT[name])["root"]: [None if v is None else order[v] for v in rows]}}
    meaning = draw(st.fixed_dictionaries({}, optional={"tiling": st.sampled_from(["partition", "cover"]),
                                                       "max_count": st.integers(0, 300), "max_gates": st.integers(0, 300)}))
    return annot.instance(TEXT[name], query, tables, meaning)


@settings(max_examples=150, derandomize=True, database=None, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(_instances())
def test_check_is_the_reference(server, inst) -> None:
    """The check's verdict is the reference's, but where the check bounds a group's or a label table's gates by the sum of
    its sets' sizes: there it may refuse what the reference accepts, never the converse."""
    check = _call(server, "annot.check", inst)
    ref = _call(server, "annot.reference", inst)
    q = inst["annotation"]["query"]
    summed = q["op"] == "group" or (q["op"] == "label" and "table" in q["map"])
    if summed and "max_gates" in inst["meaning"] and inst["meaning"].get("tiling") != "partition":
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
        (gemm, {"op": "group", "base": to_json(parse('zip("//TGemm_v1", ["@2"])')), "key": "definition"}, None, "plan:"),
        (gemm, {"op": "label", "base": {"op": "default", "w": 8}, "map": {"affine": [1, 0, 0]}}, None, "term:"),
        (synth, {"op": "table", "path": ".", "table": "t", "of": "gates"}, {"t": {key: [None] * 7}}, "term:"),
        (synth, {"op": "table", "path": ".", "table": "t", "of": "items"}, None, "term:"),
        (gemm, {"op": "label", "base": {"op": "default", "w": 8}, "map": {"affine": [1, 0, 2], "table": "u"}}, None, "term:"),
        (gemm, {"op": "label", "base": {"op": "default", "w": 8}, "map": {"table": "u"}}, None, "table:"),
        (gemm, {"op": "group", "base": {"op": "default", "w": 8}, "key": "static:"}, None, "term:"),
        (gemm, {"op": "group", "base": {"op": "default", "w": 8}, "key": [1]}, None, "term:"),
        (gemm, {"op": "label", "base": {"op": "root"}, "map": {"affine": [1, 0, 2]}}, None, "plan:"),
        (synth, {"op": "label", "base": to_json(parse('zip(".", ["@1", "@2"])')), "map": {"table": "u"}},
         {"u": {key: [None, 0, 1, None, None]}}, "plan:"),
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
