"""`table(path, name, of)` and `group(F, key)` in the structural query language (`query_ast`): codec and string syntax
round-trips, the sets they denote on a small Program checked against gates listed by hand, and counts that never build a
set."""

from __future__ import annotations

import json

import pytest

from verity.ml.circuits import query_ast as G
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.parts import Part, n_members
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_ast import QueryError
from verity.ml.circuits.query_codec import from_json, query_id, to_json
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V32 = Value(32)


def prim(name: str, params, fn) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), V32, fn, register=False)


Add32 = prim("TgAdd32", [("a", V32), ("b", V32)], lambda a, b: (a + b) & 0xFFFFFFFF)
Mul32 = prim("TgMul32", [("a", V32), ("b", V32)], lambda a, b: (a * b) & 0xFFFFFFFF)
Zero32 = prim("TgZero32", [], lambda: 0)
Sq = CompositeDefinition("TgSq", 1, (), lambda S: ((("a", V32),), V32), lambda B, S, a: B.call(Mul32, a, a), register=False)
Pair = CompositeDefinition("TgPair", 1, ("N",), lambda S: ((("x", Array(S.N, V32)),), Array(S.N, V32)),
                           lambda B, S, x: B.batch(Sq, B.batch(Add32, x, x, axes=(0, 0)), axes=(0,)), register=False)
Step = CompositeDefinition("TgStep", 1, (), lambda S: ((("c", V32), ("x", V32)), Tuple(V32, V32)),
                           lambda B, S, c, x: (lambda s: tuple_of(s, B.call(Mul32, s, x)))(B.call(Add32, c, x)), register=False)


def _mix(B, S, x):
    a = B.call(bind(Pair, N=2), x[0:2])
    b = B.call(bind(Pair, N=2), x[2:4])
    c = B.call(bind(Pair, N=4), x)
    d = B.batch(Mul32, x, x, axes=(0, 0))
    s = B.scan(Step, B.call(Zero32), xs=[d])
    return tuple_of(a, b, c, s[0])


Mix = CompositeDefinition("TgMix", 1, (), lambda S: ((("x", Array(4, V32)),), Tuple(Array(2, V32), Array(2, V32), Array(4, V32), V32)),
                          _mix, register=False)
Scaled = CompositeDefinition("TgScaled", 1, ("EPS",), lambda S: ((("x", V32),), V32),
                             lambda B, S, x: B.call(Mul32, B.call(Add32, x, x), x), register=False)
Holder = CompositeDefinition("TgHolder", 1, ("EPS",), lambda S: ((("x", V32),), V32),
                             lambda B, S, x: B.call(bind(Scaled, EPS=S.EPS), x), register=False)
PAIR2, PAIR4 = bind(Pair, N=2), bind(Pair, N=4)


@pytest.fixture(scope="module")
def mix() -> Program:
    """Root: Input32 x (4) | call TgPair{N=2} | call TgPair{N=2} | call TgPair{N=4} | batch TgMul32 (4) | TgZero32 |
    scan TgStep (4).  TgPair{N}: batch TgAdd32 (N), batch TgSq (N, one TgMul32 each)."""
    P = Program(bind(Mix))
    assert [n.form for n in P.root.body.nodes] == ["batch", "call", "call", "call", "batch", "primitive", "scan"]
    assert P.gates == 4 + 4 + 4 + 8 + 4 + 1 + 8
    return P


def gates(P: Program, j: int, m: int | None = None) -> set[int]:
    """Root gates of root node j (member m, or all of it)."""
    node = P.root.body.nodes[j]
    ms = range(n_members(node)) if m is None else (m,)
    return set().union(*(range(*iv) for mm in ms for iv in Part(P.circuit.root_scope, frozenset({(j, mm)})).intervals()))


def gate_set(s: G.GateSet) -> set[int]:
    return {g for lo, hi in s.intervals() for g in range(lo, hi)}


TEXTS = ['table(".", "owners")', 'table("//TgPair_v1{N=2}", "t", of="gates")', 'table("@call:TgPair_v1", "t", "gates")',
         'each(instances("//TgPair_v1{*}"), table(".", "t", of="gates"))', 'group(nodes(".", "@*"), "base")',
         'group(nodes(".", "@*"), ["base", "static:N"])', 'group(members("//TgPair_v1{*}", "@*"), [])',
         'each("//TgPair_v1{*}", group(members(".", "@*"), "definition"))', 'concat(table(".", "owners"), rest())',
         'group(nodes(".", "@*"), "name")']


def test_table_and_group_round_trip_through_the_codec_and_the_string_syntax():
    seen = set()
    for t in TEXTS:
        q = parse(t)
        doc = to_json(q)
        assert from_json(json.loads(json.dumps(doc))) == q
        qid = query_id(q)
        assert qid == query_id(from_json(doc)) and qid not in seen
        seen.add(qid)
    # a nodes table and a one-part key are `Flock.Annot`'s documents; `of="gates"` and a list key are the mirror's own
    assert to_json(parse(TEXTS[0])) == {"op": "table", "path": ".", "table": "owners"}
    assert to_json(parse(TEXTS[1])) == {"op": "table", "path": "//TgPair_v1{N=2}", "table": "t", "of": "gates"}
    nodes = {"op": "nodes", "path": ".", "sel": "@*"}
    assert to_json(parse(TEXTS[9])) == {"op": "group", "base": nodes, "key": "name"}
    assert to_json(parse(TEXTS[5])) == {"op": "group", "base": nodes, "key": ["base", "static:N"]}
    assert parse(TEXTS[4]) == G.group(G.nodes(".", "@*"), ["base"]) and parse(TEXTS[0]) == G.table(".", "owners")
    assert from_json({"op": "group", "base": nodes, "key": ["name"]}) == parse(TEXTS[9])
    assert from_json({"op": "table", "path": ".", "table": "owners", "of": "nodes"}) == parse(TEXTS[0])
    assert query_id(parse(TEXTS[1])) == query_id(parse(TEXTS[2].replace("@call:TgPair_v1", "//TgPair_v1{N=2}")))


@pytest.mark.parametrize("bad", ['table(".")', 'table(".", 3)', 'table(".", "t", of="leaves")', 'table(".", "")',
                                 'group(nodes(".", "@*"))', 'group(nodes(".", "@*"), "module")', 'group(nodes(".", "@*"), 3)',
                                 'group(nodes(".", "@*"), ["static:"])'])
def test_malformed_terms_are_refused(bad):
    with pytest.raises(QueryError):
        parse(bad)


@pytest.mark.parametrize("doc", [{"op": "table", "path": "."}, {"op": "table", "path": ".", "table": "t", "of": "x"},
                                 {"op": "group", "base": {"op": "root"}, "key": "def"},
                                 {"op": "group", "base": {"op": "root"}, "key": ["bogus"]}, {"op": "group", "key": []},
                                 {"op": "group", "base": {"op": "root"}, "key": 3}])
def test_malformed_documents_are_refused(doc):
    with pytest.raises(QueryError):
        from_json(doc)


def test_a_nodes_table_at_the_root_labels_whole_root_nodes(mix):
    owners = {descriptor_id(mix.root): [None, 0, 0, 1, 2, 2, 1]}
    fam = G.evaluate(mix, 'table(".", "owners")', tables={"owners": owners})
    assert fam.count() == 3 and [fam.label_of(i) for i in range(3)] == [0, 1, 2]
    want = [gates(mix, 1) | gates(mix, 2), gates(mix, 3) | gates(mix, 6), gates(mix, 4) | gates(mix, 5)]
    assert [gate_set(s) for s in fam] == want == [gate_set(fam.by_index(i)) for i in range(3)]
    assert [fam.index_of(s) for s in fam] == [0, 1, 2]
    assert fam.tiling() is None and sum(c.total for c in fam.classes()) == 3
    whole = G.evaluate(mix, 'concat(table(".", "owners"), rest())', tables={"owners": owners})
    assert whole.count() == 3 + 4 and set().union(*map(gate_set, whole)) == set(range(mix.gates))
    assert sum(s.size() for s in whole) == mix.gates


def test_a_gates_table_labels_gates_inside_nested_activations(mix):
    rows = {"t": {descriptor_id(PAIR2): [0, 1, 1, 0]}}  # TgPair{N=2}: TgAdd32 m0, m1, then TgSq m0's and m1's TgMul32
    fam = G.evaluate(mix, 'each(instances("//TgPair_v1{N=2}"), table(".", "t", of="gates"))', tables=rows)
    assert fam.count() == 4 and [fam.label_of(i) for i in range(4)] == [0, 1, 0, 1]
    want = []
    for j in (1, 2):
        lo = min(gates(mix, j))
        want += [{lo, lo + 3}, {lo + 1, lo + 2}]
    assert [gate_set(s) for s in fam] == want
    s0 = fam.by_index(0)
    assert sorted(len(p.items) for p in s0.parts) == [1, 1] and {p.spec.id for p in s0.parts} == {PAIR2.id, "TgSq_v1"}
    assert [fam.index_of(s) for s in fam] == [0, 1, 2, 3]
    assert [gate_set(s) for s in G.evaluate(mix, 'table("//TgPair_v1{N=2}", "t", of="gates")', tables=rows)] == want
    deep = G.evaluate(mix, 'table("//TgPair_v1{N=2}", "t", of="gates")', tables={"t": {descriptor_id(PAIR2): [None, None, 0, 0]}})
    assert deep.count() == 2 and [len(s.parts) for s in deep] == [2, 2]  # TgSq's TgMul32s only, in two activations
    assert [deep.index_of(s) for s in deep] == [0, 1] and [fam.index_of(s) for s in deep] == [None, None]


def test_tables_are_refused_when_missing_or_malformed(mix):
    q = 'table(".", "owners")'
    n = len(mix.root.body.nodes)
    root = descriptor_id(mix.root)
    for tables in (None, {}, {"owners": [0] * n}, {"owners": {"Other_v1": [0] * n}}, {"owners": {root: [0] * (n - 1)}},
                   {"owners": {root: [0, -1] + [0] * (n - 2)}}, {"owners": {root: [True] * n}},
                   {"owners": {root: [0, 2] + [None] * (n - 2)}}, {"owners": {root: [1] * n}}):
        with pytest.raises(QueryError):
            G.evaluate(mix, q, tables=tables)
    with pytest.raises(QueryError):  # a gates row has one label per gate of the activation
        G.evaluate(mix, 'table("//TgPair_v1{N=2}", "t", of="gates")', tables={"t": {descriptor_id(PAIR2): [0, 0]}})
    assert G.evaluate(mix, q, tables={"owners": {root: [None] * n}}).count() == 0


def test_rows_and_keys_read_the_descriptor_id():
    """A float static's descriptor id writes its canonical JSON, not `fn.id`'s rendering: tables are keyed by it, and the
    `definition` and `static:K` keys read it, as `Flock.Annot` and the integration's views do."""
    P = Program(bind(Holder, EPS=0.01))
    (scaled,) = {n.fn for n in P.root.body.nodes if n.form == "call"}
    did, eps = descriptor_id(scaled), '{"f64":"%s"}' % (0.01).hex()
    assert did == "TgScaled_v1{EPS=" + eps + "}" != scaled.id
    q = 'table("//TgScaled_v1", "t")'
    with pytest.raises(QueryError):
        G.evaluate(P, q, tables={"t": {scaled.id: [0, 1]}})
    assert G.evaluate(P, q, tables={"t": {did: [0, 1]}}).count() == 2
    fam = G.evaluate(P, 'group(nodes(".", "@call:TgScaled_v1"), ["definition", "static:EPS", "name"])')
    assert fam.count() == 1 and fam.label_of(0) == (did, eps, "TgScaled") and fam.by_index(0).size() == 2


def test_table_and_group_sets_are_not_refined_or_filtered(mix):
    rows = {"t": {descriptor_id(mix.root): [None, 0, 0, 0, 1, 1, 1]}, "g": {descriptor_id(PAIR2): [0, 0, 0, 0]}}
    for q in ['each(table(".", "t"), members(".", "@*"))', 'where(group(nodes(".", "@*"), "base"), form="call")',
              'concat(table("//TgPair_v1{N=2}", "g", of="gates"), rest())', 'group(zip(".", ["@1", "@2"]), "base")',
              'concat(group(nodes(".", "@*"), "base"), rest())']:
        with pytest.raises(QueryError):
            G.evaluate(mix, q, tables=rows)


def reference_groups(fam, key) -> list[tuple[tuple[str, ...], set[int]]]:
    """vLLM's `partition_by` rule over a family's sets: class id = the key of the set's node's Definition, classes in
    first-seen order."""
    out: dict[tuple[str, ...], set[int]] = {}
    for s in fam:
        (part,) = s.parts
        (j,) = {j for j, _ in part.items}
        out.setdefault(G._Evaluator._set_key(part.spec.body.nodes[j].fn, key), set()).update(gate_set(s))
    return list(out.items())


@pytest.mark.parametrize("base", ['nodes(".", "@*")', 'members(".", "@*")', 'members("//TgPair_v1{*}", "@*")',
                                  'each("//TgPair_v1{*}", members(".", "@*"))', 'default(32)'])
@pytest.mark.parametrize("key", [["base"], ["definition"], ["name"], ["static:N"], ["base", "static:N"], []])
def test_group_is_partition_by_over_its_base(mix, base, key):
    fam = G.evaluate(mix, G.group(parse(base), key))
    want = reference_groups(G.evaluate(mix, base), tuple(key))
    assert [(fam.label_of(i), gate_set(s)) for i, s in enumerate(fam)] == want
    assert [fam.index_of(s) for s in fam] == list(range(fam.count()))
    assert fam.tiling() is None and sum(c.total for c in fam.classes()) == fam.count()


def test_group_keys_at_the_root(mix):
    fam = G.evaluate(mix, 'group(nodes(".", "@*"), ["base", "static:N"])')
    labels = [fam.label_of(i) for i in range(fam.count())]
    assert labels == [("Input32_v1", ""), ("TgPair_v1", "2"), ("TgPair_v1", "4"), ("TgMul32_v1", ""), ("TgZero32_v1", ""),
                      ("TgStep_v1", "")]
    assert gate_set(fam.by_index(1)) == gates(mix, 1) | gates(mix, 2)
    by_name = G.evaluate(mix, 'group(nodes(".", "@*"), "name")')
    assert [by_name.label_of(i) for i in range(5)] == [("Input32",), ("TgPair",), ("TgMul32",), ("TgZero32",), ("TgStep",)]
    assert gate_set(by_name.by_index(1)) == gates(mix, 1) | gates(mix, 2) | gates(mix, 3)
    by_definition = G.evaluate(mix, 'group(nodes(".", "@*"), "definition")')
    assert [by_definition.label_of(i) for i in (1, 2)] == [("TgPair_v1{N=2}",), ("TgPair_v1{N=4}",)]
    (one,) = G.evaluate(mix, 'group(nodes(".", "@*"), [])')
    assert gate_set(one) == set(range(mix.gates))


def test_group_inside_each_groups_per_activation(mix):
    fam = G.evaluate(mix, 'each("//TgPair_v1{*}", group(members(".", "@*"), "base"))')
    assert fam.count() == 6 and [fam.label_of(i) for i in range(6)] == [("TgAdd32_v1",), ("TgSq_v1",)] * 3
    lo = min(gates(mix, 3))
    assert gate_set(fam.by_index(4)) == set(range(lo, lo + 4)) and gate_set(fam.by_index(5)) == set(range(lo + 4, lo + 8))


def test_counts_and_labels_build_no_set(mix, monkeypatch):
    rows = {"t": {descriptor_id(PAIR2): [0, 1, 1, 0]}}
    tabled = G.evaluate(mix, 'each(instances("//TgPair_v1{N=2}"), table(".", "t", of="gates"))', tables=rows)
    grouped = G.evaluate(mix, 'each("//TgPair_v1{*}", group(members(".", "@*"), "base"))')

    def built(*_a, **_k):
        raise AssertionError("a set was built")

    monkeypatch.setattr(G, "_set_of", built)
    monkeypatch.setattr(G, "_union_parts", built)
    assert tabled.count() == 4 and tabled.label_of(3) == 1
    assert grouped.count() == 6 and grouped.label_of(5) == ("TgSq_v1",)
    big = Program(bind(Pair, N=4096))
    fam = G.evaluate(big, 'group(members(".", "@*"), "base")')
    assert fam.count() == 3 and fam.label_of(2) == ("TgSq_v1",)
    rows = {"t": {descriptor_id(big.root): [0, 1, None]}}
    assert G.evaluate(big, 'table(".", "t")', tables=rows).count() == 2
