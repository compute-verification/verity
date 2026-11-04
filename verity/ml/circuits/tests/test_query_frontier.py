"""`frontier(X)`: Q_call v1's rule as a term of `query_ast`.  `cut.evaluate_call` reads its proof blocks from the term, so
`qcall_vectors.json` pins the term byte for byte; here the term is held to the rule decided from the gates alone (`rule`, the
layout's last readers in `cut.CallGates`), on the pinned programs and on a seeded corpus of nested calls, batches and scans with
pass-throughs, constants, dead gates and values returned twice, at several X; `each(instances("@*"), frontier(X))` over a
program is `CallPartition`'s proof blocks in order; its codec and string syntax round-trip; and its counts build no set."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from test_qcall_vectors import BOOLEAN_CALLS, WORD_CALLS, _c
from verity.ml.circuits import cut, partition_object as PO, query_ast as G
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import I32Add
from verity.ml.circuits.program import Program
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_ast import QueryError
from verity.ml.circuits.query_codec import from_json, to_json
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V32 = Value(32)
QCALL = 'each(instances("@*"), frontier(%d))'


def rule(Gs: cut.CallGates, X: int) -> list:
    """Q_call's step 4 from the gates: from the Call down, depth first in layout order, a piece without computing gates is no
    proof block, one whose computing gates read at or after its end (or returned) sum to at most X bits is one, else its
    children: an activation's body nodes, a batch or scan node's members or iterations."""
    units: list = []

    def piece(lo: int, hi: int, path: tuple, how: tuple) -> None:
        if not Gs.any_computing(lo, hi):
            return
        if Gs.w_out(lo, hi) <= X:
            units.append((lo, hi, path))
            return
        kind, of, off = how
        assert kind != "gate", "a gate wider than X: the query is inapplicable"
        if kind == "activation":
            body = of.body
            for k, node in enumerate(body.nodes):
                a, b = off + body.offsets[k], off + body.offsets[k + 1]
                if node.form == "primitive":
                    piece(a, b, path + ((k,),), ("gate", node.fn, a))
                elif node.form == "call":
                    piece(a, b, path + ((k,),), ("activation", node.fn, a))
                else:
                    piece(a, b, path + ((k,),), ("members", node, a))
            return
        step, g = path[-1][0], of.fn.gates
        for m in range(of.n):
            a = off + m * g
            piece(a, a + g, path[:-1] + ((step, m),), ("activation" if hasattr(of.fn, "body") else "gate", of.fn, a))

    piece(0, Gs.n, (), ("activation", Gs.fn, 0) if hasattr(Gs.fn, "body") else ("gate", Gs.fn, 0))
    return units


def _cat(t, *cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


StepSum = _c("FStepSum", (("c", V32), ("x", V32)), Tuple(V32, V32), lambda B, S, c, x: (lambda s: tuple_of(s, s))(B.call(I32Add, c, x)))
StepKeep = _c("FStepKeep", (("c", V32), ("x", V32)), Tuple(V32, V32), lambda B, S, c, x: tuple_of(c, B.call(I32Add, c, x)))
StepPass = _c("FStepPass", (("c", V32), ("x", V32)), Tuple(V32, V32), lambda B, S, c, x: tuple_of(B.call(I32Add, c, x), x))
STEPS = (StepSum, StepKeep, StepPass)


def _pool_step(rng: random.Random, name: str, pool: list):
    """A scan step (c, x) -> (carry, y) through an earlier Definition: carry and y are two of its returns or the parameters."""
    d, rd = rng.choice(pool)
    picks = [rng.randrange(rd + 2) for _ in range(2)]

    def body(B, S, c, x):
        out = B.call(d, _cat(Array(2, V32), c, x))
        vals = [c, x] + [out[i] for i in range(rd)]
        return tuple_of(vals[picks[0]], vals[picks[1]])

    return _c(name, (("c", V32), ("x", V32)), Tuple(V32, V32), body)


def _random_def(rng: random.Random, name: str, pool: list) -> tuple:
    """A Definition (x: 2 x 32 bits) -> R x 32 bits whose body is a few random steps over a pool of values: a sum, a constant,
    a call or a batch of an earlier Definition, a scan (its step a sum or a call of an earlier Definition); it returns R pool
    values, repeats and its parameters allowed."""
    steps: list = []
    for i in range(rng.randint(1, 5)):
        r = rng.random()
        ix = lambda n: [rng.randrange(64) for _ in range(n)]  # noqa: E731
        if r < 0.25 or not pool:
            steps.append(("add", *ix(2)) if rng.random() < 0.8 else ("const", rng.randrange(100)))
        elif r < 0.45:
            d, rd = rng.choice(pool)
            steps.append(("call", d, rd, ix(2), rng.sample(range(rd), rd if rng.random() < 0.6 else rng.randint(1, rd))))
        elif r < 0.65:
            d, rd = rng.choice(pool)
            k = rng.randint(1, 3)
            steps.append(("batch", d, rd, k, ix(2 * k), [(rng.randrange(k), rng.randrange(rd)) for _ in range(rng.randint(1, 3))]))
        elif r < 0.75:
            k = rng.randint(1, 3)
            steps.append(("prims", k, ix(2 * k), [rng.randrange(k) for _ in range(rng.randint(1, 2))]))
        else:
            n = rng.randint(1, 3)
            step = rng.choice(STEPS) if rng.random() < 0.4 else _pool_step(rng, f"{name}s{i}", pool)
            steps.append(("scan", step, n, ix(n), ix(1)[0], [rng.randrange(n) for _ in range(rng.randint(0, n))]))
    R = rng.randint(1, 3)
    ret = [rng.randrange(64) if rng.random() < 0.4 else -1 - rng.randrange(3) for _ in range(R)]  # mostly the latest values

    def body(B, S, x):
        vals = [x[0], x[1]]

        def pick(i):
            return vals[i % len(vals)]

        for s in steps:
            if s[0] == "add":
                vals.append(B.call(I32Add, pick(s[1]), pick(s[2])))
            elif s[0] == "const":
                vals.append(B.call(const(32, s[1])))
            elif s[0] == "call":
                out = B.call(s[1], _cat(Array(2, V32), *map(pick, s[3])))
                vals.extend(out[i] for i in s[4])
            elif s[0] == "batch":
                out = B.batch(s[1], _cat(Array(s[3], Array(2, V32)), *map(pick, s[4])), axes=(0,))
                vals.extend(out[m][i] for m, i in s[5])
            elif s[0] == "prims":
                a = _cat(Array(s[1], V32), *map(pick, s[2][:s[1]]))
                b = _cat(Array(s[1], V32), *map(pick, s[2][s[1]:]))
                out = B.batch(I32Add, a, b, axes=(0, 0))
                vals.extend(out[m] for m in s[3])
            else:
                out = B.scan(s[1], pick(s[4]), xs=[_cat(Array(s[2], V32), *map(pick, s[3]))])
                vals.append(out[0])
                vals.extend(out[1][t] for t in s[5])
        return _cat(Array(R, V32), *map(pick, ret))

    return _c(name, (("x", Array(2, V32)),), Array(R, V32), body), R


def corpus(seed: int, n: int) -> list:
    rng = random.Random(seed)
    pool: list = []
    for i in range(n):
        pool.append(_random_def(rng, f"F{seed}x{i}", pool))
    return [d for d, _r in pool]


CORPUS = [d for seed in range(20261028, 20261028 + 40) for d in corpus(seed, 10)]
XS = (32, 33, 64, 96, 128, 4096)


def _widest(fn) -> int:
    Gs = cut.CallGates(fn)
    return max((Gs.width[j] for j in Gs.keep), default=0)


@pytest.mark.parametrize("fn", WORD_CALLS + BOOLEAN_CALLS, ids=lambda f: f.id)
def test_the_term_is_the_rule_on_the_pinned_calls(fn):
    memo: dict = {}
    for X in (1, 16, 32, 64, 96, 1 << 16):
        if X >= _widest(fn):
            assert G.frontier_pieces(fn, X, memo) == rule(cut.CallGates(fn), X), X


def _steps(fn, path: tuple):
    """The kind of each step of a proof block's path: a node whole (its form) or a member (form + "-member")."""
    for st in path:
        node = fn.body.nodes[st[0]]
        yield node.form if len(st) == 1 else node.form + "-member"
        fn = node.fn


def test_the_term_is_the_rule_on_a_seeded_corpus():
    memo, seen = {}, Counter()
    for fn in CORPUS:
        Gs = cut.CallGates(fn)
        for X in XS:
            want = rule(Gs, X)
            assert G.frontier_pieces(fn, X, memo) == want, (fn.id, X)
            seen.update(k for _lo, _hi, path in want for k in _steps(fn, path))
            seen.update("depth %d" % len(path) for _lo, _hi, path in want if len(path) >= 2)
    for kind in ("primitive", "call", "batch", "batch-member", "scan", "scan-member", "depth 2", "depth 3"):
        assert seen[kind] >= 5, (kind, seen)


def test_each_instances_frontier_is_call_partition_in_order():
    P = Program(_c("FTop", (("x", Array(2, V32)),), Array(len(CORPUS[:24]), V32),
                   lambda B, S, x: _cat(Array(24, V32), *[B.call(fn, x)[0] for fn in CORPUS[:24]])))
    for X in (32, 64, 128):
        part = PO.CallPartition(P, PO.call_query(X))
        fam = G.evaluate(P, QCALL % X)
        assert fam.count() == part.population
        assert [s.intervals() for s in fam] == [[part.gates(i)] for i in range(part.population)]


def _small() -> Program:
    """x (two Input gates), a sum, a constant and a Row: six root gates, 0-1 the Input, 3 the constant."""
    return Program(_c("FTopSmall", (("x", Array(2, V32)),), Tuple(V32, V32, V32),
                      lambda B, S, x: tuple_of(B.call(I32Add, x[0], x[1]), B.call(const(32, 3)), B.call(Row, x))))


def test_a_call_that_fits_is_its_member_and_a_gate_that_computes_nothing_no_set():
    P = _small()
    fam = G.evaluate(P, QCALL % 32)
    nodes = P.root.body.nodes
    assert [(e.kind, nodes[e.j].fn.id) for e in fam.plan.entries] == [("atomic", I32Add.id), ("atomic", Row.id)]
    assert fam.count() == 2 and fam.tiling() is not None


def test_outside_each_frontier_cuts_the_enclosing_activation():
    """The root whole when it fits: one set spanning its nodes, Input gates and the constant included, so no tiling; else its
    pieces, and `rest()` beside them holds what no set took (the Input members and the constant)."""
    P = _small()
    whole = G.evaluate(P, "frontier(4096)")
    assert [s.intervals() for s in whole] == [[(0, P.root.gates)]] and whole.tiling() is None
    assert G.evaluate(P, "concat(frontier(4096), rest())").count() == 1
    assert [s.intervals() for s in G.evaluate(P, "frontier(32)")] == [[(2, 3)], [(4, 6)]]
    assert [s.intervals() for s in G.evaluate(P, "concat(frontier(32), rest())")] == [[(2, 3)], [(4, 6)], [(0, 1)], [(1, 2)], [(3, 4)]]


@pytest.mark.parametrize("text", ["frontier(32)", QCALL % 32, 'override(frontier(64), {"//Gemm_v1{*}": frontier(16)})',
                                  'concat(frontier(0), rest())'])
def test_frontier_round_trips_through_the_codec_and_the_string_syntax(text):
    q = parse(text)
    assert from_json(to_json(q)) == q and parse(text.replace(" ", "")) == q


def test_the_document_is_flock_annots():
    assert to_json(G.frontier(32)) == {"op": "frontier", "w": 32}
    assert to_json(parse(QCALL % 32)) == {"op": "each", "base": {"op": "instances", "path": "@*"}, "inner": {"op": "frontier", "w": 32}}


@pytest.mark.parametrize("bad", ["frontier()", "frontier(-1)", "frontier(True)", 'frontier("32")', "frontier(1.5)"])
def test_malformed_terms_are_refused(bad):
    with pytest.raises(QueryError):
        parse(bad)


@pytest.mark.parametrize("doc", [{"op": "frontier"}, {"op": "frontier", "w": "32"}, {"op": "frontier", "w": -1},
                                 {"op": "frontier", "w": True}, {"op": "frontier", "w": 1.0}])
def test_malformed_documents_are_refused(doc):
    with pytest.raises(QueryError):
        from_json(doc)


Row = _c("FRow", (("x", Array(2, V32)),), V32, lambda B, S, x: B.call(I32Add, B.call(I32Add, x[0], x[1]), x[1]))


def _wide(n: int):
    return _c(f"FWide{n}", (("rows", Array(n, Array(2, V32))),), Array(n, V32), lambda B, S, rows: B.batch(Row, rows, axes=(0,)))


def test_counts_are_structural(monkeypatch):
    """A batch of 2^17 rows of two sums, each row 32 bits out, counted with every set builder raising: at X = 32 the rows are
    the sets, at 32 * 2^17 the Call is one, and at 31 every sum is (a gate is one set; Q_call refuses that X)."""
    n = 1 << 17
    P = Program(_c("FTopWide", (("rows", Array(n, Array(2, V32))),), Array(n, V32), lambda B, S, rows: B.call(_wide(n), rows)))

    def boom(*a, **k):
        raise AssertionError("a set was built")

    monkeypatch.setattr(G, "_set_of", boom)
    monkeypatch.setattr(G, "_union_parts", boom)
    assert G.evaluate(P, QCALL % 32).count() == n
    assert G.evaluate(P, QCALL % (32 * n)).count() == 1
    assert G.evaluate(P, QCALL % 31).count() == 2 * n
