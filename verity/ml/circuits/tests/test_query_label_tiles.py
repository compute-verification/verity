"""`members(path, sel, k=(r, c))` and `label(F, affine=(a, b, n) | table=name)` in the structural query language
(`query_ast`): codec and string syntax round-trips, rectangles checked against PoUW's tile rule written out by hand (r A
rows by c B rows, row blocks outermost), label maps against the union of the sets they send to each label, and counts that
never build a set."""

from __future__ import annotations

import json

import pytest

from verity.ml.circuits import query_ast as G
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_ast import GateSet, QueryError
from verity.ml.circuits.query_codec import from_json, query_id, to_json
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V32 = Value(32)


def prim(name: str, params, fn) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), V32, fn, register=False)


Add32 = prim("TlAdd32", [("a", V32), ("b", V32)], lambda a, b: (a + b) & 0xFFFFFFFF)
Mul32 = prim("TlMul32", [("a", V32), ("b", V32)], lambda a, b: (a * b) & 0xFFFFFFFF)
Coord = CompositeDefinition("TlCoord", 1, (), lambda S: ((("x", V32), ("w", V32)), V32),
                            lambda B, S, x, w: B.call(Add32, B.call(Mul32, x, w), x), register=False)
Row = CompositeDefinition("TlRow", 1, ("N",), lambda S: ((("x", V32), ("w", Array(S.N, V32))), Array(S.N, V32)),
                          lambda B, S, x, w: B.batch(bind(Coord), B.call(Add32, x, x), w, axes=(None, 0)), register=False)
Two = CompositeDefinition("TlTwo", 1, ("N",), lambda S: ((("w", Array(S.N, V32)),), Array(S.N, V32)),
                          lambda B, S, w: B.batch(Mul32, B.batch(Add32, w, w, axes=(0, 0)), w, axes=(0, 0)), register=False)


def _layer(B, S, x, w):
    r = B.batch(bind(Row, N=S.N), x, w, axes=(0, None))
    return tuple_of(r, B.batch(Mul32, x, x, axes=(0, 0)), B.call(bind(Two, N=S.N), w))


Layer = CompositeDefinition("TlLayer", 1, ("M", "N"), lambda S: ((("x", Array(S.M, V32)), ("w", Array(S.N, V32))),
                            Tuple(Array(S.M, Array(S.N, V32)), Array(S.M, V32), Array(S.N, V32))), _layer, register=False)


def _top(B, S, x, w):
    a, b = (B.call(bind(Layer, M=7, N=5), x, w) for _ in range(2))
    return tuple_of(a[0], b[0])


Top = CompositeDefinition("TlTop", 1, (), lambda S: ((("x", Array(7, V32)), ("w", Array(5, V32))),
                          Tuple(Array(7, Array(5, V32)), Array(7, Array(5, V32)))), _top, register=False)
ROWS = '@batch:TlRow'


@pytest.fixture(scope="module")
def layer() -> Program:
    """Root: Input32 x (7) | Input32 x (5) | batch TlRow{N=5} (7) | batch TlMul32 (7) | call TlTwo{N=5}.
    TlRow{N}: call TlAdd32 | batch TlCoord (N, two gates each).  TlTwo{N}: two batches."""
    P = Program(bind(Layer, M=7, N=5))
    assert [n.form for n in P.root.body.nodes] == ["batch", "batch", "batch", "batch", "call"]
    return P


def gate_set(s: GateSet) -> set[int]:
    return {g for lo, hi in s.intervals() for g in range(lo, hi)}


def tile(P: Program, r: int, c: int, u: int, v: int, j: int = 2) -> set[int]:
    """PoUW's tile (u, v) of the grid site at root node j, from the layout: A rows [u*r, u*r + r) by B rows [v*c, v*c + c)."""
    body, node = P.root.body, P.root.body.nodes[j]
    row = node.fn
    coords = row.body.nodes[1]
    out: set[int] = set()
    for m in range(u * r, min(u * r + r, node.n)):
        for x in range(v * c, min(v * c + c, coords.n)):
            lo = body.offsets[j] + m * row.gates + row.body.offsets[1] + x * coords.fn.gates
            out |= set(range(lo, lo + coords.fn.gates))
    return out


TEXTS = [f'members(".", "{ROWS}", k=(3, 2))', f'members("//TlLayer_v1{{*}}", "{ROWS}", k=[2, 5])', f'members(".", "{ROWS}", k=(1, 1))',
         f'members(".", "{ROWS}", k=1)', 'label(members(".", "@*"), affine=(3, 1, 4))', 'label(members(".", "@*"), affine=[0, 2, 3])',
         'label(members(".", "@*"), table="dev")', f'each("//TlLayer_v1{{*}}", label(members(".", "{ROWS}"), affine=(1, 0, 2)))',
         f'concat(members(".", "{ROWS}", k=(3, 2)), rest())', 'label(group(nodes(".", "@*"), "name"), affine=(1, 0, 2))']


def test_rectangles_and_label_maps_round_trip_through_the_codec_and_the_string_syntax():
    seen = set()
    for t in TEXTS:
        q = parse(t)
        doc = to_json(q)
        assert from_json(json.loads(json.dumps(doc))) == q
        qid = query_id(q)
        assert qid == query_id(from_json(doc)) and qid not in seen
        seen.add(qid)
    # `Flock.Annot`'s documents: it reads the affine map and refuses a two-axis k and a table map with its `term` code
    assert to_json(parse(TEXTS[0])) == {"op": "members", "path": ".", "sel": ROWS, "k": [3, 2]}
    members = {"op": "members", "path": ".", "sel": "@*", "k": 1}
    assert to_json(parse(TEXTS[4])) == {"op": "label", "base": members, "map": {"affine": [3, 1, 4]}}
    assert to_json(parse(TEXTS[6])) == {"op": "label", "base": members, "map": {"table": "dev"}}
    assert parse(TEXTS[0]) == G.members(".", ROWS, k=[3, 2]) and parse(TEXTS[4]) == G.label(G.members(".", "@*"), affine=[3, 1, 4])


@pytest.mark.parametrize("bad", ['members(".", "@*", k=(0, 2))', 'members(".", "@*", k=(2,))', 'members(".", "@*", k=(2, 2, 2))',
                                 'members(".", "@*", k=[True, 1])', 'members(".", "@*", k="2")', 'label(members(".", "@*"))',
                                 'label(members(".", "@*"), affine=[1, 2])', 'label(members(".", "@*"), affine=[1, 0, 0])',
                                 'label(members(".", "@*"), affine=[-1, 0, 2])', 'label(members(".", "@*"), affine=[1, 0, 2], table="t")',
                                 'label(members(".", "@*"), table="")', 'label(members(".", "@*"), 3)',
                                 'label(members(".", "@*"), map=[1, 0, 2])'])
def test_malformed_terms_are_refused(bad):
    with pytest.raises(QueryError):
        parse(bad)


@pytest.mark.parametrize("doc", [{"op": "label", "base": {"op": "root"}, "map": {}},
                                 {"op": "label", "base": {"op": "root"}, "map": {"affine": [1, 0, 0]}},
                                 {"op": "label", "base": {"op": "root"}, "map": {"affine": [1, 0, 2], "table": "t"}},
                                 {"op": "label", "base": {"op": "root"}, "map": "affine"},
                                 {"op": "label", "base": {"op": "root"}, "map": {"affine": "1,0,2"}},
                                 {"op": "label", "base": {"op": "root"}, "map": {"table": None}},
                                 {"op": "members", "path": ".", "sel": "@*", "k": [2]},
                                 {"op": "members", "path": ".", "sel": "@*", "k": 0}])
def test_malformed_documents_are_refused(doc):
    with pytest.raises(QueryError):
        from_json(doc)


@pytest.mark.parametrize("r,c", [(3, 2), (1, 1), (7, 5), (2, 5), (8, 6), (1, 3), (4, 1)])
def test_rectangles_are_row_blocks_by_column_chunks(layer, r, c):
    fam = G.evaluate(layer, G.members(".", ROWS, k=(r, c)))
    rows, cols = -(-7 // r), -(-5 // c)
    assert fam.count() == rows * cols
    want = [tile(layer, r, c, u, v) for u in range(rows) for v in range(cols)]
    assert [gate_set(s) for s in fam] == want == [gate_set(fam.by_index(i)) for i in range(fam.count())]
    assert len(fam.by_index(0).parts) == min(r, 7)
    assert [fam.index_of(s) for s in fam] == list(range(fam.count()))
    assert [fam.label_of(i) for i in range(fam.count())] == [None] * fam.count()
    classes = fam.classes()
    assert fam.tiling() is None and sum(cl.total for cl in classes) == fam.count() and len(classes) <= 4
    assert {(len(cl.representative.parts), cl.representative.size()) for cl in classes} == {
        (len(s.parts), s.size()) for s in fam}


def test_index_of_reads_a_rectangle_however_it_is_spelled(layer):
    fam = G.evaluate(layer, G.members(".", ROWS, k=(3, 2)))
    s = fam.by_index(4)
    assert fam.index_of(GateSet(tuple(reversed(s.parts)))) == 4
    assert fam.index_of(s.minus(fam.by_index(0))) == 4
    one_gate = GateSet((G.Part(s.parts[0].scope, frozenset({next(iter(s.parts[0].items))})),))
    assert fam.index_of(one_gate) is None and fam.index_of(s.union(fam.by_index(5))) is None
    assert fam.index_of(G.evaluate(layer, "root()").by_index(0)) is None


def test_rest_beside_rectangles_is_the_rest_of_the_circuit(layer):
    whole = G.evaluate(layer, f'concat(members(".", "{ROWS}", k=(3, 2)), rest())')
    assert sum(s.size() for s in whole) == layer.gates and set().union(*map(gate_set, whole)) == set(range(layer.gates))
    rest = [gate_set(s) for s in list(whole)[9:]]
    row = layer.root.body.nodes[2].fn
    adds = [layer.root.body.offsets[2] + m * row.gates for m in range(7)]  # each TlRow member's own TlAdd32
    assert [{g} for g in adds] == [s for s in rest if len(s) == 1 and next(iter(s)) in adds]


def test_rectangles_inside_each_are_the_path_form():
    P = Program(bind(Top))
    inner = G.evaluate(P, f'each("//TlLayer_v1{{*}}", members(".", "{ROWS}", k=(3, 2)))')
    path = G.evaluate(P, f'members("//TlLayer_v1{{*}}", "{ROWS}", k=(3, 2))')
    assert inner.count() == path.count() == 18
    assert [gate_set(s) for s in inner] == [gate_set(s) for s in path]
    assert [inner.index_of(s) for s in inner] == list(range(18))
    assert sum(cl.total for cl in inner.classes()) == 18


def test_rectangles_are_refused_without_two_axes(layer):
    for q in ['members(".", "@batch:TlMul32", k=(2, 2))', 'members(".", "@call:TlTwo", k=(2, 2))',
              f'each(members(".", "{ROWS}", k=(3, 2)), members(".", "@*"))', f'where(members(".", "{ROWS}", k=(3, 2)), form="batch")']:
        with pytest.raises(QueryError):
            G.evaluate(layer, q)
    (grouped,) = G.evaluate(layer, f'group(members(".", "{ROWS}", k=(3, 2)), "base")')
    assert gate_set(grouped) == set().union(*(tile(layer, 3, 2, u, v) for u in range(3) for v in range(3)))


def by_residue(fam, rule) -> list[set[int]]:
    a, b, n = rule
    out: list[set[int]] = [set() for _ in range(n)]
    for i, s in enumerate(fam):
        out[(a * i + b) % n] |= gate_set(s)
    return out


@pytest.mark.parametrize("rule", [(3, 1, 4), (2, 1, 4), (0, 2, 3), (1, 0, 1), (5, 9, 4), (1, 0, 40), (6, 3, 9)])
def test_an_affine_map_joins_the_sets_of_each_residue(layer, rule):
    base = G.members(".", "@*")
    fam = G.evaluate(layer, G.label(base, affine=rule))
    want = by_residue(G.evaluate(layer, base), rule)
    assert fam.count() == rule[2] and [gate_set(s) for s in fam] == want
    assert [fam.label_of(i) for i in range(rule[2])] == list(range(rule[2]))
    first = {}
    for i, s in enumerate(want):
        first.setdefault(frozenset(s), i)
    assert [fam.index_of(s) for s in fam if gate_set(s)] == [first[frozenset(s)] for s in want if s]
    assert fam.tiling() is None and sum(cl.total for cl in fam.classes()) == rule[2]


def test_an_affine_map_inside_each_counts_within_the_activation():
    P = Program(bind(Top))
    fam = G.evaluate(P, f'each("//TlLayer_v1{{*}}", label(members(".", "{ROWS}"), affine=(1, 0, 2)))')
    rows = G.evaluate(P, f'members("//TlLayer_v1{{*}}", "{ROWS}")')
    assert fam.count() == 4 and [fam.label_of(i) for i in range(4)] == [0, 1, 0, 1]
    assert [gate_set(s) for s in fam] == [set().union(*(gate_set(rows.by_index(7 * a + m)) for m in range(p, 7, 2)))
                                          for a in range(2) for p in range(2)]


def test_a_table_map_labels_a_set_by_its_node(layer):
    dev = {"dev": {descriptor_id(layer.root): [None, None, 0, 1, 0]}}
    fam = G.evaluate(layer, 'label(members(".", "@*"), table="dev")', tables=dev)
    nodes = G.evaluate(layer, 'nodes(".", "@*")')
    whole = [gate_set(nodes.by_index(j)) for j in range(5)]
    assert fam.count() == 2 and [fam.label_of(i) for i in range(2)] == [0, 1]
    assert [gate_set(s) for s in fam] == [whole[2] | whole[4], whole[3]]
    assert [fam.index_of(s) for s in fam] == [0, 1]
    nested = G.evaluate(layer, 'label(members("//TlRow", "@*"), table="dev")', tables=dev)
    assert [gate_set(s) for s in nested] == [whole[2], set()]
    same = {"dev": {descriptor_id(layer.root): [None, None, 0, 0, 1]}}
    zipped = G.evaluate(layer, f'label(zip(".", ["@batch:TlMul32", "{ROWS}"]), table="dev")', tables=same)
    assert [gate_set(s) for s in zipped] == [whole[2] | whole[3], set()]
    with pytest.raises(QueryError):  # a zip set spans the Row and Mul nodes, which the row puts on two labels
        G.evaluate(layer, f'label(zip(".", ["@batch:TlMul32", "{ROWS}"]), table="dev")', tables=dev)
    for tables in (None, {"dev": {"Other_v1": [0] * 5}}, {"dev": {descriptor_id(layer.root): [0, 2, None, None, None]}}):
        with pytest.raises(QueryError):
            G.evaluate(layer, 'label(members(".", "@*"), table="dev")', tables=tables)
    with pytest.raises(QueryError):
        G.evaluate(layer, 'concat(label(members(".", "@*"), table="dev"), rest())', tables=dev)


def test_counts_and_labels_build_no_set(monkeypatch):
    big = Program(bind(Layer, M=4096, N=4096))
    root = descriptor_id(big.root)

    def built(*_a, **_k):
        raise AssertionError("a set was built")

    monkeypatch.setattr(G, "_set_of", built)
    monkeypatch.setattr(G, "_union_parts", built)
    tiles = G.evaluate(big, G.members(".", ROWS, k=(64, 96)))
    assert tiles.count() == 64 * 43 and tiles.label_of(64 * 43 - 1) is None
    affine = G.evaluate(big, G.label(G.members(".", "@*"), affine=(3, 1, 8)))
    assert affine.count() == 8 and affine.label_of(5) == 5
    tabled = G.evaluate(big, 'label(members(".", "@*"), table="dev")', tables={"dev": {root: [0, 0, 1, 2, 1]}})
    assert tabled.count() == 3 and tabled.label_of(2) == 2
