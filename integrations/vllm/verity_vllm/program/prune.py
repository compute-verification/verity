"""A flat body's dead units, and the body without them.

A unit is one node of a body: a Call, a primitive, a batch or a scan. It is live when the body's return reads one of its outputs,
or a live unit's arguments do; a unit no live unit or return reads is dead, whatever its gates compute. This is the unit level of
circuit-check's dead-gates property (`verity.ml.circuits.liveness`, transitive), which also counts the gates no output reads inside
a live unit. `without_dead_units` drops the dead units and renumbers the references to the rest, each reference kept in its form
(an `Affine` stays an `Affine`, a `Concat` keeps its parts), so that equal bodies stay equal: the Build's derive and
`predict.build` drop the request wrapper's dead units alike (the last layer's rows before a prefill's last).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from verity.ml.circuits.defs import Body
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Refs, Strided


def _read(r: Refs, out: set[int]) -> None:
    if isinstance(r, (Affine, Strided)):
        if r.space == "n" and r.count:
            out.add(r.idx)
    elif isinstance(r, Concat):
        for p in r.parts:
            _read(p, out)
    elif isinstance(r, Explicit):
        out.update(i for s, i, _ in r.items if s == "n")
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")


def live_units(body: Body) -> list[bool]:
    """Per node of `body`, whether the return reads it through live nodes (a node reads only earlier nodes)."""
    live, read = [False] * len(body.nodes), set()
    _read(body.ret, read)
    for k in range(len(body.nodes) - 1, -1, -1):
        if k in read:
            live[k] = True
            for a in body.nodes[k].args:
                _read(a.refs, read)
    return live


def dead_units(body: Body) -> list[int]:
    """The indices of `body`'s dead units, in body order."""
    return [k for k, v in enumerate(live_units(body)) if not v]


def _renumbered(r: Refs, new: list[int], memo: dict[int, Refs]) -> Refs:
    if (hit := memo.get(id(r))) is not None:
        return hit
    if isinstance(r, Affine):
        out = Affine("n", new[r.idx], r.base, r.count) if r.space == "n" and r.count and new[r.idx] != r.idx else r
    elif isinstance(r, Strided):
        out = (Strided("n", new[r.idx], r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
               if r.space == "n" and r.count and new[r.idx] != r.idx else r)
    elif isinstance(r, Concat):
        parts = [_renumbered(p, new, memo) for p in r.parts]
        out = r if all(p is q for p, q in zip(parts, r.parts)) else Concat(parts)
    elif isinstance(r, Explicit):
        items = tuple((s, new[i] if s == "n" else i, j) for s, i, j in r.items)
        out = r if items == r.items else Explicit(items)
    else:
        raise TypeError(f"a reference sequence of an unknown form, {type(r).__name__}")
    memo[id(r)] = out
    return out


def without_dead_units(body: Body) -> tuple[Body, list[int]]:
    """`body` without its dead units, and the index in `body` of each node kept, in order."""
    kept = [k for k, v in enumerate(live_units(body)) if v]
    if len(kept) == len(body.nodes):
        return body, kept
    new = list(range(len(body.nodes)))
    for i, k in enumerate(kept):
        new[k] = i
    memo: dict[int, Refs] = {}

    def coll(a: Coll) -> Coll:
        r = _renumbered(a.refs, new, memo)
        return a if r is a.refs else Coll(a.type, r)

    nodes, offsets = [], [0]
    for k in kept:
        n = body.nodes[k]
        nodes.append(replace(n, args=tuple(coll(a) for a in n.args)))
        offsets.append(offsets[-1] + n.gates)
    return Body(body.params, body.ret_type, nodes, _renumbered(body.ret, new, memo), offsets), kept


def kept_records(body: Body, calls: list, hints: dict[int, Any], *, drop: bool) -> tuple[Body, list, dict[int, Any]]:
    """A derivation's flat `body` with its runtime correspondence's `calls` (one per node) and its node `hints` (by node index),
    without the dead units when `drop`."""
    if not drop:
        return body, calls, dict(hints)
    pruned, kept = without_dead_units(body)
    return pruned, [calls[k] for k in kept], {i: hints[k] for i, k in enumerate(kept) if k in hints}
