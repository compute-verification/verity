"""A composite's evaluation and its non-constant gates, body by body.

The reference (`verity.ml.circuits.evaluation.evaluate`) walks a composite's standalone call gate by gate
(`verity.ml.circuits.evaluation.evaluate_region`), and finds each gate and each of its operands from the root
(`layout.Circuit.gate`, `layout.resolve_gate`); that search is most of what checking a large Definition costs. `evaluate`
here compiles each specialized Definition's body once into local wires (its parameters' leaves, then each node's output
leaves) and runs it per activation: the same gates, in the same canonical order, on the same operands, through the same
primitive evaluators. So it returns what the reference returns and raises the first exception the reference raises, and
its `MissingValue` messages name the same gate indices. A Definition whose body it doesn't compile (`Unsupported`), or whose
activations would copy far more leaves than it has gates (`MAX_TRAFFIC`), is evaluated by the reference.

`nonconstant` is the same traversal for `checks._pieces_used`: the primitives of a Definition's gates that aren't constant
(a gate is constant when its family is ``Const``, its primitive takes no parameters, or every operand is constant), counted
once per body and pattern of constant parameter leaves.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.evaluation import MissingValue
from verity.ml.circuits.evaluation.reference import RegisteredTable, resolve, standalone_call
from verity.ml.circuits.evaluation.reference import evaluate as reference
from verity.ml.circuits.layout import WIDE_FAN_IN
from verity.ml.circuits.refs import runs

#: a body of more nodes than this is left to the reference: its compiled steps would hold several hundred bytes a gate
MAX_NODES = 1 << 22
#: precomputed operand wires of one batch of primitives, at most (a larger batch gathers its operands member by member)
MAX_MEMBER_WIRES = 1 << 22
#: a Definition whose activations copy more leaves (parameters in, returns out, a primitive's operands) than this many a
#: gate, plus `TRAFFIC_SLACK`, is left to the reference, which resolves only the leaves a gate reads
MAX_TRAFFIC = 4
TRAFFIC_SLACK = 1 << 22


class Unsupported(Exception):
    """A body `compile_body` leaves to the reference."""


_PRIM1, _PRIM2, _PRIMN, _PRIMS, _GATHER, _LAZY, _INPUT, _CALL, _BATCH, _SCAN = range(10)
_ALL, _AX0, _PER = range(3)


class Body:
    """One specialized Definition's body over local wires: ``P`` parameter leaves first, ``size`` wires in all. ``steps``
    evaluate its nodes in order, ``plans`` describe them for `nonconstant`, ``ret`` are its return's wires, ``traffic`` the
    leaves one activation copies."""

    __slots__ = ("spec", "P", "size", "pad", "steps", "plans", "ret", "traffic", "consts")

    def __init__(self, spec: SpecializedDefinition) -> None:
        self.spec = spec
        self.consts: dict[tuple, tuple[Counter, tuple]] = {}


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
                steps.append((_PRIM2, fn.evaluate, cols[0][0], cols[1][0], out))
            elif n == 1:
                steps.append((_PRIMN, fn.evaluate, [c[0] for c in cols], out))
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


def _run(body: Body, V: list, off: int) -> None:
    """``body``'s steps on ``V``, its parameters' values followed by room for every node's outputs; ``off`` is the
    activation's root-scoped gate offset, which names a gate in an error as the reference names it."""
    for st in body.steps:
        op = st[0]
        if op == _PRIM2:
            V[st[4]] = st[1](V[st[2]], V[st[3]])
        elif op == _PRIM1:
            V[st[3]] = st[1](V[st[2]])
        elif op == _PRIMS:
            _, ev, cols, n, out = st
            get = V.__getitem__
            V[out:out + n] = list(map(ev, *[map(get, c) for c in cols])) if cols else [ev() for _ in range(n)]
        elif op == _PRIMN:
            V[st[3]] = st[1](*[V[w] for w in st[2]])
        elif op == _CALL:
            _, sub, ws, out, goff = st
            cv = [V[w] for w in ws]
            cv.extend(sub.pad)
            _run(sub, cv, off + goff)
            V[out:out + len(sub.ret)] = [cv[w] for w in sub.ret]
        elif op == _BATCH:
            _, sub, sa, n, out, goff, G = st
            R = len(sub.ret)
            shared = [[V[w] for w in s[1]] if s[0] == _ALL else None for s in sa]
            for m in range(n):
                cv: list = []
                for s, vals in zip(sa, shared):
                    if vals is not None:
                        cv.extend(vals)
                    elif s[0] == _AX0:
                        L = s[2]
                        cv.extend([V[w] for w in s[1][m * L:(m + 1) * L]])
                    else:
                        cv.extend([V[w] for w in s[1][m]])
                cv.extend(sub.pad)
                _run(sub, cv, off + goff + m * G)
                V[out + m * R:out + (m + 1) * R] = [cv[w] for w in sub.ret]
        elif op == _SCAN:
            _, sub, init, xs, shared, n, c, y, out, goff, G = st
            carry = [V[w] for w in init]
            svals = [V[w] for w in shared]
            for m in range(n):
                cv = carry
                for ws, L in xs:
                    cv.extend([V[w] for w in ws[m * L:(m + 1) * L]])
                cv.extend(svals)
                cv.extend(sub.pad)
                _run(sub, cv, off + goff + m * G)
                r = [cv[w] for w in sub.ret]
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
                    get = (lambda j, _ws=ws: V[_ws[j]])
                try:
                    V[out + m] = lazy(get)
                except KeyError as e:
                    raise MissingValue(f"gate {off + goff + m} ({pid}) operand {e.args[0]} is not in the transcript") from None
        elif op == _GATHER:
            _, ev, n, sa, out = st
            for m in range(n):
                V[out + m] = ev(*[V[w] for w in _member(sa, m)])
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
    V.extend(body.pad)
    _run(body, V, call.lo)
    return tuple(V[w] for w in body.ret)


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
