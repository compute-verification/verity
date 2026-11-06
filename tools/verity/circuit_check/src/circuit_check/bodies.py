"""A composite's evaluation and its non-constant gates, body by body.

The reference (`verity.ml.circuits.evaluation.evaluate`) walks a composite's standalone call gate by gate
(`verity.ml.circuits.evaluation.evaluate_region`), and finds each gate and each of its operands from the root
(`layout.Circuit.gate`, `layout.resolve_gate`); that search is most of what checking a large Definition costs. `evaluate`
here compiles each specialized Definition's body once into local wires (its parameters' leaves, then each node's output
leaves) and runs it per activation: the same gates, in the same canonical order, on the same operands, through the same
primitive evaluators, except that And2, Xor2 and Not on `int` operands take the bit their evaluators return through Python's
own ``&`` and ``^`` (`_and2`, `_xor2`, `_BITS`), and their evaluators on anything else. A body of only gates and constant
bits (`_lanes`) on arguments that are each an `int` 0 or 1 runs `LANES` vectors at once (`evaluate_rows`, and `evaluate`
as one), bit i of each wire vector i's value: ``&``, ``^`` and ``^`` with every lane set are each lane's And, Xor and Not,
which never mix lanes. So it returns what the reference returns and raises the first exception the reference raises, and
its `MissingValue` messages name the same gate indices. A Definition whose body it doesn't compile (`Unsupported`), or whose
activations would copy far more leaves than it has gates (`MAX_TRAFFIC`), is evaluated by the reference.

Within one `evaluate`, an activation of a body of at least `MEMO_GATES_PER_LEAF` gates a leaf it copies, on parameter values
an earlier activation of that body returned on, takes that one's return values (`_Memo`): a primitive's value is a function of
its operands' alone, so they are the values it would compute, and an activation that raised returned none, so the first one
to raise is the reference's. An edge vector sets every leaf alike, so a batch's members mostly repeat one activation.

`nonconstant` is the same traversal for `checks._pieces_used`: the primitives of a Definition's gates that aren't constant
(a gate is constant when its family is ``Const``, its primitive takes no parameters, or every operand is constant), counted
once per body and pattern of constant parameter leaves.
"""
from __future__ import annotations

from collections import Counter
from functools import partial, reduce
from itertools import repeat
from operator import and_, xor
from typing import Any, Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.evaluation import MissingValue
from verity.ml.circuits.evaluation.reference import RegisteredTable, resolve, standalone_call
from verity.ml.circuits.evaluation.reference import evaluate as reference
from verity.ml.circuits.layout import WIDE_FAN_IN
from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import axis_runs, remove_axis

#: a body of more nodes than this is left to the reference: its compiled steps would hold several hundred bytes a gate
MAX_NODES = 1 << 22
#: precomputed operand wires of one batch of primitives, at most (a larger batch gathers its operands member by member)
MAX_MEMBER_WIRES = 1 << 22
#: a Definition whose activations copy more leaves (parameters in, returns out, a primitive's operands) than this many a
#: gate, plus `TRAFFIC_SLACK`, is left to the reference, which resolves only the leaves a gate reads
MAX_TRAFFIC = 4
TRAFFIC_SLACK = 1 << 22
#: an activation is looked up (`_Memo`) when its body has at least this many gates per parameter and return leaf, so a miss
#: costs a small part of running it
MEMO_GATES_PER_LEAF = 2
#: the parameter and return leaves one `evaluate` holds for activations it may repeat, at most
MEMO_LEAVES = 1 << 22


class Unsupported(Exception):
    """A body `compile_body` leaves to the reference."""


_PRIM1, _PRIM2, _PRIMN, _PRIMS, _GATHER, _LAZY, _INPUT, _CALL, _BATCH, _SCAN, _COLS = range(11)
_ALL, _AX0, _PER = range(3)


def _and2(a: Any, b: Any) -> Any:
    return a & b & 1 if type(a) is int and type(b) is int else BL.AND.evaluate(a, b)


def _xor2(a: Any, b: Any) -> Any:
    return (a ^ b) & 1 if type(a) is int and type(b) is int else BL.XOR.evaluate(a, b)


def _and_cols(a: list, b: list) -> Any:
    return map(and_, map(and_, a, b), repeat(1))


def _xor_cols(a: list, b: list) -> Any:
    return map(and_, map(xor, a, b), repeat(1))


def _not_cols(a: list) -> Any:
    return map(and_, map(xor, a, repeat(1)), repeat(1))


#: a two-operand gate's evaluator on any operands, the same as the primitive's
_PAIRS = {BL.AND: _and2, BL.XOR: _xor2}
#: a batch's values from its operands' columns where every operand is an `int` (`_COLS`): on those, each is the bit the
#: primitive's evaluator returns
_BITS = {BL.AND: _and_cols, BL.XOR: _xor_cols, BL.NOT: _not_cols}


class Body:
    """One specialized Definition's body over local wires: ``P`` parameter leaves first, ``size`` wires in all. ``steps``
    evaluate its nodes in order, ``plans`` describe them for `nonconstant`, ``ret`` are its return's wires, ``traffic`` the
    leaves one activation copies, ``lanes`` its steps on `LANES` vectors at once (`_lanes`: None until asked, False where
    it has none)."""

    __slots__ = ("spec", "P", "size", "pad", "steps", "plans", "ret", "traffic", "consts", "memo", "lanes")

    def __init__(self, spec: SpecializedDefinition) -> None:
        self.spec = spec
        self.consts: dict[tuple, tuple[Counter, tuple]] = {}
        self.memo = False
        self.lanes: list | bool | None = None


_BODIES: dict[Any, Body | Unsupported] = {}


def compile_body(spec: SpecializedDefinition) -> Body:
    hit = _BODIES.get(spec)
    if hit is None:
        try:
            hit = _compile(spec)
        except Unsupported as e:
            hit = e
        _BODIES[spec] = hit
    if isinstance(hit, Unsupported):
        raise hit
    return hit


def _member(spec_args: Sequence[tuple], m: int) -> list[int]:
    """Member ``m``'s operand (or parameter) wires of a batch, from its arguments' specs (`_compile`'s ``argspec``)."""
    out: list[int] = []
    for s in spec_args:
        if s[0] == _ALL:
            out.extend(s[1])
        elif s[0] == _AX0:
            L = s[2]
            out.extend(s[1][m * L:(m + 1) * L])
        else:
            out.extend(s[1][m])
    return out


def _compile(spec: SpecializedDefinition) -> Body:
    b = spec.body
    nodes = b.nodes
    if len(nodes) > MAX_NODES:
        raise Unsupported(f"{spec.id}: {len(nodes)} nodes")
    pbase = [0]
    for _, pt in b.params:
        pbase.append(pbase[-1] + pt.leaves)
    P = pbase[-1]
    nbase, size = [], P
    for node in nodes:
        nbase.append(size)
        size += node.out.leaves

    def wires(refs: Any, below: int) -> list[int]:
        out: list[int] = []
        for space, idx, lo, hi in runs(refs):
            if space == "p":
                if not (0 <= idx < len(b.params) and 0 <= lo <= hi <= pbase[idx + 1] - pbase[idx]):
                    raise Unsupported(f"{spec.id}: a reference to parameter {idx} leaves [{lo}:{hi}]")
                base = pbase[idx]
            else:
                if not (0 <= idx < below and 0 <= lo <= hi <= nodes[idx].out.leaves):
                    raise Unsupported(f"{spec.id}: node {below} reads node {idx} leaves [{lo}:{hi}]")
                base = nbase[idx]
            out.extend(range(base + lo, base + hi))
        return out

    def whole(colls: Sequence[Any], k: int) -> list[int]:
        return [w for a in colls for w in wires(a.refs, k)]

    def argspec(node: Any, k: int) -> list[tuple]:
        """Per argument of a batch, as the reference binds member m's parameters (`layout.resolve_gate`): (_ALL, wires) for
        a whole argument, (_AX0, wires, L) for member m's run of L leaves, (_PER, [member m's wires]) along a later axis."""
        out = []
        for a, ax in zip(node.args, node.axes):
            if ax is None:
                out.append((_ALL, wires(a.refs, k)))
            elif ax == 0:
                out.append((_AX0, wires(a.refs, k), a.type.elem.leaves))
            elif node.n and remove_axis(a.type, ax)[0] == node.n:
                # member m's leaves: the run of L at m in each of `outer` blocks of n * L leaves (`axis_runs`)
                outer, n, L = axis_runs(a.type, ax)
                ws, s = wires(a.refs, k), n * L
                out.append((_PER, [[w for o in range(m * L, outer * s, s) for w in ws[o:o + L]] for m in range(n)]))
            else:
                out.append((_PER, [wires(a.take(ax, m).refs, k) for m in range(node.n)]))
        return out

    body = Body(spec)
    steps: list[tuple] = []
    plans: list[tuple] = []
    traffic = 0
    for k, node in enumerate(nodes):
        fn, out, goff, form = node.fn, nbase[k], b.offsets[k], node.form
        if form in ("primitive", "batch") and isinstance(fn, PrimitiveDefinition):
            n = 1 if form == "primitive" else node.n
            if node.out.leaves != n:
                raise Unsupported(f"{spec.id}: node {k} has {node.out.leaves} output leaves for {n} gates")
            if n == 0:
                continue
            sa = [(_ALL, wires(a.refs, k)) for a in node.args] if form == "primitive" else argspec(node, k)
            wide = sum(p.leaves for _, p in fn.params) > WIDE_FAN_IN
            fanin = len(_member(sa, 0))
            cols = None if wide or n * fanin > MAX_MEMBER_WIRES else [list(c) for c in zip(*(_member(sa, m) for m in range(n)))]
            plans.append(("prim", fn, n, sa, cols, wide, out))
            lazy = getattr(fn, "evaluate_lazy", None)
            traffic += n if wide and lazy is not None else n * fanin
            if getattr(fn, "family", None) == "Input":
                steps.append((_INPUT, goff))
            elif lazy is not None:
                steps.append((_LAZY, lazy, fn.id, n, sa, cols, wide, out, goff))
            elif cols is None:
                steps.append((_GATHER, fn.evaluate, n, sa, out))
            elif n == 1 and fanin == 1:
                steps.append((_PRIM1, fn.evaluate, cols[0][0], out))
            elif n == 1 and fanin == 2:
                steps.append((_PRIM2, _PAIRS.get(fn, fn.evaluate), cols[0][0], cols[1][0], out))
            elif n == 1:
                steps.append((_PRIMN, fn.evaluate, [c[0] for c in cols], out))
            elif fn in _BITS:
                steps.append((_COLS, _BITS[fn], fn.evaluate, cols, n, out))
            else:
                steps.append((_PRIMS, fn.evaluate, cols, n, out))
            continue
        if not isinstance(fn, SpecializedDefinition):
            raise Unsupported(f"{spec.id}: node {k} is a {form} of {fn!r}")
        sub = compile_body(fn)
        R, G = len(sub.ret), fn.gates
        if form == "call":
            ws = whole(node.args, k)
            if len(ws) != sub.P or node.out.leaves != R:
                raise Unsupported(f"{spec.id}: node {k} calls {fn.id} with {len(ws)} of its {sub.P} parameter leaves")
            steps.append((_CALL, sub, ws, out, goff))
            plans.append(("call", sub, ws, out))
            traffic += sub.P + R + sub.traffic
        elif form == "batch":
            n = node.n
            sa = argspec(node, k)
            if (n and len(_member(sa, 0)) != sub.P) or node.out.leaves != n * R:
                raise Unsupported(f"{spec.id}: node {k} batches {fn.id} over mismatched leaves")
            if n:
                steps.append((_BATCH, sub, sa, n, out, goff, G))
                plans.append(("batch", sub, sa, n, out))
            traffic += n * (sub.P + R + sub.traffic)
        elif form == "scan":
            n, nx = node.n, node.n_xs
            if n == 0:
                raise Unsupported(f"{spec.id}: node {k} scans {fn.id} zero times, an output the reference refuses to read")
            init = wires(node.args[0].refs, k)
            xs = [(wires(a.refs, k), a.type.elem.leaves) for a in node.args[1:1 + nx]]
            shared = whole(node.args[1 + nx:], k)
            c = len(init)
            y = R - c
            if (c + sum(L for _, L in xs) + len(shared) != sub.P or y < 0 or node.out.leaves != c + n * y
                    or fn.body.params[0][1].leaves != c):
                raise Unsupported(f"{spec.id}: node {k} scans {fn.id} over mismatched leaves")
            steps.append((_SCAN, sub, init, xs, shared, n, c, y, out, goff, G))
            plans.append(("scan", sub, init, xs, shared, n, c, y, out))
            traffic += n * (sub.P + R + sub.traffic)
        else:
            raise Unsupported(f"{spec.id}: node {k} has form {form!r}")
    body.ret = wires(b.ret, len(nodes))
    body.P, body.size, body.pad = P, size, [None] * (size - P)
    body.steps, body.plans, body.traffic = steps, plans, traffic
    body.memo = spec.gates >= MEMO_GATES_PER_LEAF * (P + len(body.ret))
    return body


class _Wide:
    """Member ``m``'s operand values of a wide-fan-in gate, read on demand as `layout.LazyOperands` reads its gates."""

    __slots__ = ("V", "segs", "n")

    def __init__(self, V: list, spec_args: Sequence[tuple], m: int) -> None:
        self.V, self.segs, n = V, [], 0
        for s in spec_args:
            if s[0] == _ALL:
                seq, lo, cnt = s[1], 0, len(s[1])
            elif s[0] == _AX0:
                seq, lo, cnt = s[1], m * s[2], s[2]
            else:
                seq, lo, cnt = s[1][m], 0, len(s[1][m])
            self.segs.append((n, cnt, seq, lo))
            n += cnt
        self.n = n

    def __getitem__(self, j: int) -> Any:
        if j < 0:
            j += self.n
        if not 0 <= j < self.n:
            raise IndexError(j)
        for start, cnt, seq, lo in self.segs:
            if j < start + cnt:
                return self.V[seq[lo + j - start]]
        raise IndexError(j)

    def all(self) -> bool:
        return all(self.V[w] for _start, cnt, seq, lo in self.segs for w in seq[lo:lo + cnt])


_INT = {int}

#: the vectors a body of only gates and constant bits runs at once (`_lanes`)
LANES = 64
_ONES = (1 << LANES) - 1


def _and_all(*x: int) -> int:
    return reduce(and_, x)


def _xor_all(*x: int) -> int:
    return reduce(xor, x)


def _lane_ev(fn: PrimitiveDefinition) -> Callable | None:
    """``fn``'s evaluator on `LANES` bits a value, where ``fn`` is a gate or a constant bit (else None): each lane the bit
    ``fn``'s evaluator returns on that lane's operands."""
    if fn is BL.NOT:
        return partial(xor, _ONES)
    if fn is BL.CONSTANTS[0]:
        return int
    if fn is BL.CONSTANTS[1]:
        return partial(int, _ONES)
    kind = BL.gate_kind(fn)
    if kind is None:
        return None
    if kind[0] == "and":
        return and_ if kind[1] == 2 else _and_all
    return xor if kind[1] == 2 else _xor_all


def _lanes(body: Body) -> list | None:
    """``body``'s steps on `LANES` vectors at once where every primitive it reaches is a gate or a constant bit (`_lane_ev`),
    else None: each primitive's steps with its lanes' evaluator, a batch of And2, Xor2 or Not over columns a map of it.
    `_compile` makes one step and one plan of each node it keeps, in order, so a step's plan names its primitive."""
    if body.lanes is None:
        body.lanes = False
        out: list[tuple] = []
        for st, plan in zip(body.steps, body.plans):
            op = st[0]
            if plan[0] != "prim":
                if _lanes(st[1]) is None:
                    return None
                out.append(st)
                continue
            ev = _lane_ev(plan[1])
            if ev is None:
                return None
            if op == _COLS:
                out.append((_PRIMS, ev, *st[3:]))
            elif op in (_PRIM1, _PRIM2, _PRIMN, _PRIMS, _GATHER):
                out.append((op, ev, *st[2:]))
            else:
                return None
        body.lanes = out
    return None if body.lanes is False else body.lanes


def _bits(V: list) -> bool:
    return {*map(type, V)} <= _INT and {*V} <= {0, 1}


class _Memo:
    """One `evaluate`'s activations that returned, by body and parameter values (each an `int`), and the leaves it may still
    hold (`MEMO_LEAVES`)."""

    __slots__ = ("seen", "room")

    def __init__(self) -> None:
        self.seen: dict[Body, dict[tuple, list]] = {}
        self.room = MEMO_LEAVES


def _activated(sub: Body, cv: list, off: int, memo: _Memo, lanes: bool) -> list:
    """``sub``'s return values on its parameters' values ``cv``, which it extends with room for its nodes' outputs: an earlier
    activation's on the same values where ``sub`` is looked up (`Body.memo`), else run. ``lanes``: `_run`'s."""
    key = tuple(cv) if sub.memo and (lanes or {*map(type, cv)} <= _INT) else None
    if key is not None:
        seen = memo.seen.setdefault(sub, {})
        if (hit := seen.get(key)) is not None:
            return hit
    cv.extend(sub.pad)
    _run(sub, cv, off, memo, lanes)
    ret = list(map(cv.__getitem__, sub.ret))
    if key is not None and memo.room >= len(key) + len(ret) and (lanes or {*map(type, ret)} <= _INT):
        seen[key] = ret
        memo.room -= len(key) + len(ret)
    return ret


def _run(body: Body, V: list, off: int, memo: _Memo, lanes: bool) -> None:
    """``body``'s steps on ``V``, its parameters' values followed by room for every node's outputs; ``off`` is the
    activation's root-scoped gate offset, which names a gate in an error as the reference names it. ``lanes``: its steps on
    `LANES` vectors at once (`_lanes`), every value in ``V`` `LANES` bits."""
    val = V.__getitem__
    for st in (body.lanes if lanes else body.steps):
        op = st[0]
        if op == _PRIM2:
            V[st[4]] = st[1](V[st[2]], V[st[3]])
        elif op == _PRIM1:
            V[st[3]] = st[1](V[st[2]])
        elif op == _PRIMS:
            _, ev, cols, n, out = st
            V[out:out + n] = list(map(ev, *[map(val, c) for c in cols])) if cols else [ev() for _ in range(n)]
        elif op == _COLS:
            _, bits, ev, cols, n, out = st
            xs = [list(map(val, c)) for c in cols]
            V[out:out + n] = list(bits(*xs) if all({*map(type, x)} <= _INT for x in xs) else map(ev, *xs))
        elif op == _PRIMN:
            V[st[3]] = st[1](*map(val, st[2]))
        elif op == _CALL:
            _, sub, ws, out, goff = st
            V[out:out + len(sub.ret)] = _activated(sub, list(map(val, ws)), off + goff, memo, lanes)
        elif op == _BATCH:
            _, sub, sa, n, out, goff, G = st
            R = len(sub.ret)
            shared = [list(map(val, s[1])) if s[0] == _ALL else None for s in sa]
            for m in range(n):
                cv: list = []
                for s, vals in zip(sa, shared):
                    if vals is not None:
                        cv.extend(vals)
                    elif s[0] == _AX0:
                        L = s[2]
                        cv.extend(map(val, s[1][m * L:(m + 1) * L]))
                    else:
                        cv.extend(map(val, s[1][m]))
                V[out + m * R:out + (m + 1) * R] = _activated(sub, cv, off + goff + m * G, memo, lanes)
        elif op == _SCAN:
            _, sub, init, xs, shared, n, c, y, out, goff, G = st
            carry = list(map(val, init))
            svals = list(map(val, shared))
            for m in range(n):
                cv = carry
                for ws, L in xs:
                    cv.extend(map(val, ws[m * L:(m + 1) * L]))
                cv.extend(svals)
                r = _activated(sub, cv, off + goff + m * G, memo, lanes)
                carry = r[:c]
                V[out + c + m * y:out + c + (m + 1) * y] = r[c:]
            V[out:out + c] = carry
        elif op == _LAZY:
            _, lazy, pid, n, sa, cols, wide, out, goff = st
            for m in range(n):
                if wide:
                    get = _Wide(V, sa, m).__getitem__
                else:
                    ws = tuple(c[m] for c in cols) if cols is not None else tuple(_member(sa, m))
                    def get(j, _ws=ws):
                        return V[_ws[j]]
                try:
                    V[out + m] = lazy(get)
                except KeyError as e:
                    raise MissingValue(f"gate {off + goff + m} ({pid}) operand {e.args[0]} is not in the transcript") from None
        elif op == _GATHER:
            _, ev, n, sa, out = st
            for m in range(n):
                V[out + m] = ev(*map(val, _member(sa, m)))
        elif op == _INPUT:
            raise MissingValue(f"Input gate {off + st[1]} has no prescribed value in the transcript")
        else:  # pragma: no cover
            raise AssertionError(op)


def _flat(args: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for a in args:
        if isinstance(a, int):
            out.append(a)
        else:
            out.extend(int(v) for v in a)
    return out


def compiled(t: Any) -> Body | None:
    """``t``'s compiled body, or None where the reference evaluates it: a primitive or a `ScalarReference`, a body
    `compile_body` leaves to it, or activations that copy more leaves than `MAX_TRAFFIC` allows."""
    if not isinstance(t, SpecializedDefinition):
        return None
    try:
        body = compile_body(t)
    except Unsupported:
        return None
    return body if body.traffic <= MAX_TRAFFIC * t.gates + TRAFFIC_SLACK else None


def evaluate(target: Any, *args: Any) -> tuple[int, ...]:
    """`verity.ml.circuits.evaluation.evaluate`, a composite run body by body: the same output leaves, or the same first
    exception."""
    t = resolve(target)
    body = None if any(isinstance(a, RegisteredTable) for a in args) else compiled(t)
    _prog, call = standalone_call(t) if body is not None else (None, None)
    if body is None or len(call.args) != body.P:
        return reference(t, *args)
    V = _flat(args)
    if len(V) != body.P:
        raise ValueError(f"{call!r}: expected {body.P} argument encodings, got {len(V)}")
    if _bits(V) and _lanes(body) is not None:
        return _lanes_run(body, call.lo, [V])[0]
    V.extend(body.pad)
    _run(body, V, call.lo, _Memo(), False)
    return tuple(V[w] for w in body.ret)


def _lanes_run(body: Body, lo: int, Vs: list[list[int]]) -> list[tuple[int, ...]]:
    """The return leaves on each of ``Vs`` (parameter leaves, each an `int` 0 or 1) by ``body``'s steps on lanes (`_lanes`),
    `LANES` of them a run, bit i of each wire the i-th's value."""
    out: list[tuple[int, ...]] = []
    for c in range(0, len(Vs), LANES):
        chunk = Vs[c:c + LANES]
        V = [sum(v << i for i, v in enumerate(col)) for col in zip(*chunk)]
        V.extend(body.pad)
        _run(body, V, lo, _Memo(), True)
        ret = [V[w] for w in body.ret]
        out.extend(tuple((r >> i) & 1 for r in ret) for i in range(len(chunk)))
    return out


def _bit_leaves(row: Sequence[Any]) -> list[int] | None:
    """``row``'s argument leaves where each argument is an `int` or a list of them and every leaf is 0 or 1, else None."""
    V: list[int] = []
    for a in row:
        if type(a) is int:
            V.append(a)
        elif type(a) is list:
            V.extend(a)
        else:
            return None
    return V if _bits(V) else None


def _lanes_ready(target: Any) -> tuple[Body, int] | None:
    """``target``'s body and its standalone call's gate offset where `evaluate` runs it and it has steps on lanes, else None;
    raising what `evaluate` raises before it runs arguments that hold no `RegisteredTable`."""
    t = resolve(target)
    body = compiled(t)
    if body is None or _lanes(body) is None:
        return None
    _prog, call = standalone_call(t)
    return (body, call.lo) if len(call.args) == body.P else None


def evaluate_rows(target: Any, rows: Sequence[Sequence[Any]]) -> list[tuple[int, ...] | Exception]:
    """`evaluate` on each of ``rows`` (each one vector's arguments): its output leaves, or what it raised.  Where every row's
    argument leaves are each an `int` 0 or 1 and the body has steps on lanes, they run `LANES` at once (`_lanes_run`)."""
    bits = [V for V in map(_bit_leaves, rows) if V is not None]
    if rows and len(bits) == len(rows):
        try:
            ready = _lanes_ready(target)
        except Exception as e:  # noqa: BLE001  (what `evaluate` raises on each row before it runs it)
            return [e] * len(rows)
        if ready is not None and all(len(V) == ready[0].P for V in bits):
            return list(_lanes_run(ready[0], ready[1], bits))
    out: list[tuple[int, ...] | Exception] = []
    for r in rows:
        try:
            out.append(evaluate(target, *r))
        except Exception as e:  # noqa: BLE001  (what it raised is the row's outcome)
            out.append(e)
    return out


def _consts(body: Body, pattern: tuple) -> tuple[Counter, tuple]:
    """(primitive id -> gates of it that aren't constant, which return leaves are constant) of an activation of ``body``
    whose parameter leaves are constant where ``pattern`` says; the counter's keys in the order the gates first appear."""
    hit = body.consts.get(pattern)
    if hit is not None:
        return hit
    C = list(pattern)
    C.extend([False] * (body.size - body.P))
    count: Counter = Counter()
    for plan in body.plans:
        kind = plan[0]
        if kind == "prim":
            _, prim, n, sa, cols, wide, out = plan
            fam = getattr(prim, "family", None)
            if fam == "Input":
                continue
            always = fam == "Const" or not prim.params
            live = 0
            for m in range(n):
                if always:
                    c = True
                elif cols is not None:
                    c = bool(cols) and all(C[col[m]] for col in cols)
                else:
                    ops = _Wide(C, sa, m)
                    c = ops.n > 0 and ops.all()
                if c:
                    C[out + m] = True
                else:
                    live += 1
            if live:
                count[prim.id] += live
        elif kind == "call":
            _, sub, ws, out = plan
            got, ret = _consts(sub, tuple(C[w] for w in ws))
            count.update(got)
            C[out:out + len(ret)] = ret
        elif kind == "batch":
            _, sub, sa, n, out = plan
            R = len(sub.ret)
            for m in range(n):
                got, ret = _consts(sub, tuple(C[w] for w in _member(sa, m)))
                count.update(got)
                C[out + m * R:out + (m + 1) * R] = ret
        else:
            _, sub, init, xs, shared, n, c, y, out = plan
            carry = [C[w] for w in init]
            sp = [C[w] for w in shared]
            for m in range(n):
                got, ret = _consts(sub, tuple(carry + [C[w] for ws, L in xs for w in ws[m * L:(m + 1) * L]] + sp))
                count.update(got)
                carry = list(ret[:c])
                C[out + c + m * y:out + c + (m + 1) * y] = ret[c:]
            C[out:out + c] = carry
    hit = body.consts[pattern] = (count, tuple(C[w] for w in body.ret))
    return hit


def nonconstant(t: Any) -> Counter | None:
    """Primitive id -> ``t``'s gates of it that aren't constant (its parameters are not), keyed in the order the gates first
    appear; None where ``t`` has no compiled body (`compiled`). The counter is shared: don't change it."""
    body = compiled(t)
    if body is None:
        return None
    return _consts(body, (False,) * body.P)[0]
