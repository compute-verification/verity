"""`query_ast.refines(a, b)`: refinement of annotations decided from the two terms, never from a Program.  Every pair of a
pool of terms it accepts is checked against the sets themselves (`oracle`: each set of `a` inside one set of `b`, by root gate
intervals) on three Programs (a layer of rows and tiles, a Program of two layers, and nested calls, batches and scans); the
rules it states hold by name, the coarsenings it refuses fail on the sets, and it reads no Program."""
from __future__ import annotations

import bisect

import pytest

from test_query_frontier import CORPUS, V32, _c, _cat
from test_query_label_tiles import Layer, Top
from verity.ml.circuits import query_ast as G
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.intervals import norm
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_ast import QueryError
from verity.ml.circuits.types import Array

LAYER = Program(bind(Layer, M=7, N=5))
TOP = Program(Top)
CALLS = Program(_c("RfTop", (("x", Array(2, V32)),), Array(16, V32), lambda B, S, x: _cat(Array(16, V32), *[B.call(fn, x)[0] for fn in CORPUS[:16]])))
PROGRAMS = {"layer": LAYER, "top": TOP, "calls": CALLS}
TABLES = {"dev": {descriptor_id(LAYER.root): [None, None, 0, 1, None], descriptor_id(bind(Layer, M=7, N=5)): [0, 1, None]},
          "part": {descriptor_id(LAYER.root): [None, None, None, 0, None], descriptor_id(bind(Layer, M=7, N=5)): [None, 0, None]}}

L, ROWS, MULS = '"//TlLayer"', '"@batch:TlRow"', '"@batch:TlMul32"'
TERMS = [
    'root()', 'instances("@*")', f'instances({L})', f'nodes({L}, {ROWS})', f'nodes({L}, {MULS})',
    *(f'members({L}, {ROWS}, k={k})' for k in ("1", "2", "3", "4", "(2, 2)", "(2, 3)", "(4, 4)", "(4, 6)")),
    f'zip({L}, [{ROWS}, {MULS}])', f'table({L}, "dev")',
    *(f'each(instances({p}), {g}({w}))' for p in ('"@*"', L) for g in ("default", "frontier") for w in (32, 64, 4096)),
    'default(32)', 'default(64)', 'frontier(32)', 'frontier(64)', 'frontier(4096)', 'concat(frontier(32), rest())',
    f'each(members({L}, {ROWS}, k=2), default(32))', f'each(members({L}, {ROWS}), frontier(32))',
    'where(instances("@*"), form="call")', f'where(members({L}, {ROWS}), form="batch")',
    f'group(members({L}, {ROWS}), "definition")', 'group(instances("@*"), "name")',
    f'label(members({L}, {ROWS}), affine=(1, 0, 3))', f'label(members({L}, {ROWS}, k=2), affine=(2, 1, 4))',
    f'label(members({L}, {ROWS}), table="dev")',
    f'concat(nodes({L}, {ROWS}), nodes({L}, {MULS}))', f'minus(instances({L}), nodes({L}, {ROWS}))',
    f'inter(members({L}, {ROWS}, k=2), members({L}, {ROWS}, k=3))', f'union(members({L}, {ROWS}), root())',
]


def _sets(P: Program, q: str) -> list[tuple]:
    return [tuple(norm(s.intervals())) for s in G.evaluate(P, q, TABLES)]


def _inside(x: tuple, y: tuple) -> bool:
    starts = [lo for lo, _hi in y]
    for lo, hi in x:
        t = bisect.bisect_right(starts, lo) - 1
        if t < 0 or y[t][1] < hi:
            return False
    return True


def oracle(P: Program, a: str, b: str) -> bool | QueryError:
    """Every set of `a` inside one set of `b`, from the sets themselves; the refusal when either does not evaluate on `P`."""
    try:
        A, B = _sets(P, a), _sets(P, b)
    except QueryError as e:
        return e
    return all(any(_inside(x, y) for y in B) for x in A)


def _holds(got: bool | QueryError) -> bool:
    return got is True or isinstance(got, QueryError)


def test_every_pair_it_accepts_refines_on_the_sets():
    checked = 0
    for a in TERMS:
        for b in TERMS:
            if not G.refines(a, b):
                continue
            for name, P in PROGRAMS.items():
                got = oracle(P, a, b)
                assert _holds(got), (name, a, b)
                checked += got is True
    assert checked >= 600, checked


@pytest.mark.parametrize("a,b", [
    (f'each(instances({L}), frontier(32))', f'instances({L})'),
    (f'each(each(instances({L}), default(64)), default(32))', f'instances({L})'),
    (f'where(members({L}, {ROWS}), form="batch")', f'nodes({L}, {ROWS})'),
    (f'minus(instances({L}), nodes({L}, {ROWS}))', f'instances({L})'),
    (f'inter(members({L}, {ROWS}, k=2), members({L}, {ROWS}, k=3))', f'members({L}, {ROWS}, k=3)'),
    (f'members({L}, {ROWS})', f'group(members({L}, {ROWS}), "definition")'),
    (f'members({L}, {ROWS}, k=2)', f'label(members({L}, {ROWS}, k=2), affine=(2, 1, 4))'),
    (f'nodes({L}, {MULS})', f'concat(nodes({L}, {ROWS}), nodes({L}, {MULS}))'),
    (f'concat(nodes({L}, {ROWS}), nodes({L}, {MULS}))', f'instances({L})'),
    (f'members({L}, {ROWS}, k=(2, 3))', f'members({L}, {ROWS}, k=4)'),
    (f'members({L}, {ROWS}, k=(2, 3))', f'members({L}, {ROWS}, k=(4, 6))'),
    (f'members({L}, {ROWS}, k=2)', f'nodes({L}, {ROWS})'),
    (f'zip({L}, [{ROWS}, {MULS}])', f'instances({L})'),
    (f'table({L}, "dev")', f'instances({L})'),
    ('each(instances("@*"), frontier(32))', 'each(instances("@*"), frontier(64))'),
    ('each(instances("@*"), default(32))', 'each(instances("@*"), default(4096))'),
    ('frontier(32)', 'frontier(4096)'),
    ('concat(frontier(32), rest())', 'root()'),
])
def test_the_rules(a, b):
    assert G.refines(a, b) and G.refines(parse(a), parse(b))
    got = [oracle(P, a, b) for P in PROGRAMS.values()]
    assert all(_holds(g) for g in got) and any(g is True for g in got)


@pytest.mark.parametrize("a,b,on", [
    (f'instances({L})', f'each(instances({L}), default(32))', "layer"),
    (f'members({L}, {ROWS}, k=2)', f'members({L}, {ROWS}, k=3)', "layer"),
    (f'members({L}, {ROWS}, k=2)', f'members({L}, {ROWS}, k=(4, 4))', "layer"),
    (f'members({L}, {ROWS})', f'label(members({L}, {ROWS}), table="part")', "layer"),
    (f'nodes({L}, {MULS})', f'union(members({L}, {ROWS}), minus(root(), root()))', "layer"),
    ('each(instances("@*"), frontier(64))', 'each(instances("@*"), frontier(32))', "calls"),
    ('default(64)', 'default(32)', "calls"),
    (f'zip({L}, [{ROWS}, {MULS}])', f'nodes({L}, {ROWS})', "layer"),
])
def test_what_it_refuses_is_no_refinement_on_some_program(a, b, on):
    assert not G.refines(a, b)
    assert oracle(PROGRAMS[on], a, b) is False


def test_it_reads_no_program():
    """Terms whose paths match nothing anywhere are decided as well: the answer is the terms'."""
    a, b = 'each(members("//NoSuch", "@batch:None", k=(2, 3)), frontier(8))', 'group(nodes("//NoSuch", "@batch:None"), "base")'
    assert G.refines(a, b) and not G.refines(b, a)
    assert G.refines(parse(a), b) and G.refines(a, parse(b))


@pytest.mark.parametrize("bad", [42, None, "frontier(", "no_such_term(1)"])
def test_it_refuses_what_is_not_a_term(bad):
    with pytest.raises(QueryError):
        G.refines(bad, "root()")
