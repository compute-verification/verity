"""`order_violation` names the same first violation as the reference walk (a FIFO with a seen set checked when a Definition is taken, and
`_out_of_order` reading `Concat` parts front to back) on random programs: call graphs with shared callees, cycles and bodiless
primitives; references nested in `Concat`s, including `PartLog` cuts whose log grew after the cut; and several violations per program,
so a walk that visits bodies or parts in another order names another one."""
from __future__ import annotations

import random
from types import SimpleNamespace as NS

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.refs import Affine, Concat, Explicit, PartLog, Strided


def _reference_order_violation(program):
    seen: set[int] = set()
    todo = [program.root]
    while todo:
        fn = todo.pop(0)
        if id(fn) in seen or not hasattr(fn, "body"):
            continue
        seen.add(id(fn))
        body = fn.body
        for k, node in enumerate(body.nodes):
            for i, a in enumerate(node.args):
                bad = _reference_out_of_order(a.refs, len(body.params), k)
                if bad is not None:
                    return {"definition": fn.id, "node": k, "arg": i, "reference": bad, "why": "not topological"}
            todo.append(node.fn)
        bad = _reference_out_of_order(body.ret, len(body.params), len(body.nodes))
        if bad is not None:
            return {"definition": fn.id, "node": "ret", "reference": bad, "why": "not topological"}
    return None


def _reference_out_of_order(refs, params, nodes):
    parts = [refs]
    while parts:
        r = parts.pop(0)
        if isinstance(r, Concat):
            parts[:0] = r.parts
            continue
        names = [(x[0], x[1]) for x in r.items] if isinstance(r, Explicit) else ([(r.space, r.idx)] if r.count else [])
        for space, idx in names:
            if not (PO._is_int(idx) and 0 <= idx < (params if space == "p" else nodes if space == "n" else 0)):
                return [space, idx]
    return None


def _name(rng, params, k, bad):
    """A (space, idx) a node `k` of a body with `params` parameters may name, or with probability `bad` one it may not."""
    if rng.random() < bad:
        return rng.choice([("n", k), ("n", k + rng.randrange(1, 4)), ("p", params), ("p", -1), ("n", -1), ("x", 0), ("n", True), ("p", 0.0)])
    if params and (k == 0 or rng.random() < 0.5):
        return "p", rng.randrange(params)
    return ("n", rng.randrange(k)) if k else ("x", 0)


def _refs(rng, params, k, bad, depth=0):
    leaf = rng.randrange(8) if depth < 3 else rng.randrange(4)
    if leaf == 0:
        s, i = _name(rng, params, k, bad)
        return Affine(s, i, rng.randrange(4), rng.choice([0, 1, 3]))
    if leaf == 1:
        return Explicit(tuple((*_name(rng, params, k, bad), rng.randrange(4)) for _ in range(rng.randrange(4))))
    if leaf in (2, 3):
        s, i = _name(rng, params, k, bad)
        return Strided(s, i, 0, 2, 3, 1, rng.choice([0, 2, 5]))
    if leaf in (4, 5):
        return Concat([_refs(rng, params, k, bad, depth + 1) for _ in range(rng.randrange(1, 4))])
    log = PartLog()
    for _ in range(rng.randrange(1, 5)):
        log.append(_refs(rng, params, k, bad, depth + 1))
    j = rng.randrange(1, len(log) + 1)
    cut = log.raw_cut(j, [_refs(rng, params, k, bad, depth + 1) for _ in range(rng.randrange(3))])
    for _ in range(rng.randrange(3)):
        log.append(_refs(rng, params, k, bad, depth + 1))
    return cut


def _program(rng, bad):
    """Definitions d0..d{n-1} (d0 the root) whose nodes call any Definition, itself included, or a primitive without a body."""
    defs = [NS(id=f"d{j}") for j in range(rng.randrange(1, 7))]
    prims = [NS(id=f"prim{j}") for j in range(2)]
    for fn in defs:
        params = [None] * rng.randrange(4)
        nodes = []
        for k in range(rng.randrange(6)):
            args = [NS(refs=_refs(rng, len(params), k, bad)) for _ in range(rng.randrange(3))]
            nodes.append(NS(fn=rng.choice(defs + prims + prims), args=args))
        fn.body = NS(params=params, nodes=nodes, ret=_refs(rng, len(params), len(nodes), bad / 2))
    return NS(root=defs[0])


def _violations(program):
    """Every (definition, node, arg) with an out-of-order reference, over the bodies the root reaches."""
    out, seen, todo = [], set(), [program.root]
    while todo:
        fn = todo.pop()
        if id(fn) in seen or not hasattr(fn, "body"):
            continue
        seen.add(id(fn))
        n = len(fn.body.params)
        for k, node in enumerate(fn.body.nodes):
            out += [(fn.id, k, i) for i, a in enumerate(node.args) if _reference_out_of_order(a.refs, n, k) is not None]
            todo.append(node.fn)
        if _reference_out_of_order(fn.body.ret, n, len(fn.body.nodes)) is not None:
            out.append((fn.id, "ret", None))
    return out


def test_order_violation_names_the_reference_walks_first_violation_on_random_programs():
    rng = random.Random(20261027)
    none = several = 0
    for _ in range(4000):
        program = _program(rng, rng.choice([0.0, 0.02, 0.1, 0.3]))
        want = _reference_order_violation(program)
        assert PO.order_violation(program) == want
        none += want is None
        several += len(_violations(program)) >= 2
    assert none >= 500 and several >= 500, (none, several)


def test_out_of_order_names_the_reference_walks_first_reference_on_nested_concats():
    rng = random.Random(5188)
    found = 0
    for _ in range(4000):
        params, k = rng.randrange(4), rng.randrange(6)
        refs = _refs(rng, params, k, 0.2)
        want = _reference_out_of_order(refs, params, k)
        assert PO._out_of_order(refs, params, k) == want
        found += want is not None
    assert found >= 1000, found


def test_the_first_violation_is_the_root_first_breadth_first_one():
    """d0 calls d1 then d2, d1 calls d3, and d2 and d3 call back up. With d1, d2 and d3 all bad, d1's first bad argument is named
    before its ret; with d1 fixed, d2's (taken before d3, breadth first) is named, and within it the first leaf of nested `Concat`s."""
    ok, late = Affine("p", 0, 0, 1), Affine("n", 5, 0, 1)
    d0, d1, d2, d3 = (NS(id=f"d{j}") for j in range(4))
    d3.body = NS(params=[None], nodes=[NS(fn=d1, args=[NS(refs=late)])], ret=ok)
    d2.body = NS(params=[None], nodes=[NS(fn=d0, args=[NS(refs=Concat([ok, Concat([late, Affine("n", 7, 0, 1)])]))])], ret=ok)
    d1.body = NS(params=[None], nodes=[NS(fn=d3, args=[NS(refs=ok), NS(refs=Concat([Affine("n", 0, 0, 1), late]))])],
                 ret=Affine("n", 9, 0, 1))
    d0.body = NS(params=[None], nodes=[NS(fn=d1, args=[NS(refs=ok)]), NS(fn=d2, args=[NS(refs=Affine("n", 0, 0, 1))])], ret=ok)
    program = NS(root=d0)
    want = {"definition": "d1", "node": 0, "arg": 1, "reference": ["n", 0], "why": "not topological"}
    assert _reference_order_violation(program) == want and PO.order_violation(program) == want
    d1.body.nodes[0].args[1].refs = ok
    want = {"definition": "d1", "node": "ret", "reference": ["n", 9], "why": "not topological"}
    assert _reference_order_violation(program) == want and PO.order_violation(program) == want
    d1.body.ret = ok
    want = {"definition": "d2", "node": 0, "arg": 0, "reference": ["n", 5], "why": "not topological"}
    assert _reference_order_violation(program) == want and PO.order_violation(program) == want
