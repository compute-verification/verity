"""Circuit types (`circuit_types`) of a traced lowering: every call of a named function (an IR primitive's piece, a helper the
Boolean export names, a table read) is one type, built in a `gf2` context of its own, so gates are shared only inside a type
and a type is its circuit wherever it's called.

A call's inputs are the free bits of its arguments. A constant argument bit is folded into the callee, which is then its own
type (``fold``; constant-api-public §1.3: constant sub-circuits are folded before the partition), or bound at the call
(``fold=False``, the callee generic). A callee output that is linear in its inputs (wiring, a constant) is the caller's form
directly; only the outputs its own ANDs, calls or reads make are the call's wires.

    types, root = trace_types(build, label)      # build() makes a gf2.Circuit; root is its type's digest
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from verity.ml.circuits import library

from . import circuit_types as CT
from . import gf2
from .gf2 import L, ONE, ZERO

_S: dict = {"on": False, "frames": [], "types": {}, "memo": {}, "fold": True, "names": {}, "labels": []}


class _Frame:
    __slots__ = ("C", "calls")

    def __init__(self, C):
        self.C, self.calls = C, []


def _leaves(x, out):
    if isinstance(x, L):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _leaves(y, out)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            _leaves(getattr(x, f.name), out)


def _rebuild(x, it):
    """``x`` with each L leaf replaced by the next of ``it``."""
    if isinstance(x, L):
        return next(it)
    if isinstance(x, list):
        return [_rebuild(y, it) for y in x]
    if isinstance(x, tuple):
        return tuple(_rebuild(y, it) for y in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: _rebuild(getattr(x, f.name), it) for f in dataclasses.fields(x)})
    return x


def _shape(x, fold):
    """A call's key: its arguments with each bit free (``b``) or, folding, its constant."""
    if isinstance(x, L):
        return ("c", x.c) if fold and x.is_const() else "b"
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(_shape(y, fold) for y in x))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, tuple((f.name, _shape(getattr(x, f.name), fold)) for f in dataclasses.fields(x)))
    if isinstance(x, gf2.Circuit):
        return "C"
    return ("v", repr(x))


_SHA512: dict[str, str] = {}


def table_sha512(name: str) -> str:
    """A table's SHA-512 over its raw little-endian u32 words: the library's pin, or computed (the program tables)."""
    if name not in _SHA512:
        try:
            _SHA512[name] = library.table(name).sha512
        except KeyError:
            import hashlib

            import numpy as np

            _SHA512[name] = hashlib.sha512(np.ascontiguousarray(np.asarray(gf2.table(name)), dtype="<u4").tobytes()).hexdigest()
    return _SHA512[name]


def table_name(sha512: str) -> str:
    """The name of a table this module has named (the inverse of `table_sha512`)."""
    for n, d in _SHA512.items():
        if d == sha512:
            return n
    raise KeyError(f"no table named {sha512[:16]}...")


def _table(name: str) -> tuple[str, int, int]:
    from .tail_pieces import TABLE_BITS, TABLE_VALUE_BITS, VALUE_BITS

    return table_sha512(name), TABLE_BITS[name], TABLE_VALUE_BITS.get(name, VALUE_BITS)


def to_type(fr: _Frame, inputs: list[L], outputs: list[L], ins, outs, label: str) -> CT.CircuitType:
    """A type from its frame: the items its outputs reach, in the order the circuit made them (`circuit_types.from_gf2`,
    with calls)."""
    C = fr.C
    wire: dict[int, int] = {}
    for i, f in enumerate(inputs):
        (v,) = tuple(f.s)
        wire[v] = i
    groups: dict[int, list[int]] = {}
    for v, kind in enumerate(C.kind):
        if kind == "lookup":
            groups.setdefault(C.data[v][0], []).append(v)
    live, stack = set(), [v for f in outputs for v in f.s]
    while stack:
        v = stack.pop()
        if v in live or v in wire:
            continue
        kind = C.kind[v]
        if kind == "and":
            live.add(v)
            a, b = C.data[v]
            stack.extend(a.s)
            stack.extend(b.s)
        elif kind == "lookup":
            g = C.data[v][0]
            live.update(groups[g])
            for f in C.lookups[g][1]:
                stack.extend(f.s)
        elif kind == "call":
            k = C.data[v][0]
            _d, args, v0, n = fr.calls[k]
            live.update(range(v0, v0 + n))
            for f in args:
                stack.extend(f.s)
        else:
            raise CT.InvalidType(f"a {kind} bit ({C.data[v]}) is read but is not one of the type's inputs")

    def form(f):
        return (tuple(sorted(wire[v] for v in f.s)), f.c)

    n, items, calls, tables, done = len(inputs), [], [], [], set()
    for v in sorted(live):
        kind = C.kind[v]
        if kind == "and":
            a, b = C.data[v]
            items.append((CT.AND, form(a), form(b)))
            wire[v] = n
            n += 1
        elif kind == "lookup" and C.data[v][0] not in done:
            g = C.data[v][0]
            done.add(("t", g))
            done.add(g)
            name, idx, width = C.lookups[g]
            entry = _table(name)
            if width != entry[2]:
                raise CT.InvalidType(f"table {name}: a read of {width} bits, not {entry[2]}")
            if entry not in tables:
                tables.append(entry)
            items.append((CT.READ, tables.index(entry), tuple(form(f) for f in idx)))
            for u in groups[g]:
                wire[u] = n + C.data[u][1]
            n += entry[2]
        elif kind == "call" and ("c", C.data[v][0]) not in done:
            k = C.data[v][0]
            done.add(("c", k))
            d, args, v0, cnt = fr.calls[k]
            if d not in calls:
                calls.append(d)
            items.append((CT.CALL, calls.index(d), tuple(form(f) for f in args)))
            for j in range(cnt):
                wire[v0 + j] = n + j
            n += cnt
    return CT.CircuitType(label, tuple(ins), tuple(outs), tuple(calls), tuple(tables), tuple(items),
                          tuple(form(f) for f in outputs))


def _ports(args, fold) -> list[tuple[int, int]]:
    ps = []
    for x in args:
        bits: list = []
        _leaves(x, bits)
        n = sum(1 for b in bits if not (fold and b.is_const()))
        if n:
            ps.append((1, n))
    return ps


def _made(o, t: CT.CircuitType) -> bool:
    """Whether output form ``o`` of ``t`` is made inside it (an AND, call or read reaches it, or it XORs several wires), and so is
    the call's wire; a constant or one input bit (wiring) is the caller's form directly, which keeps constants propagating."""
    return len(o[0]) > 1 or any(w >= t.in_bits for w in o[0])


def call(name: str, fn: Callable, a: tuple, k: dict):
    """A traced call inside a type: the callee's type (built once per argument shape), and the caller's view of its outputs."""
    fr = _S["frames"][-1]
    fold = _S["fold"]
    parent = _S["labels"][-1]
    label = library.LIBRARY if parent == library.LIBRARY or library.definition_label(name) == library.LIBRARY else library.PROGRAM
    key = (name, label, _shape(a, fold), _shape(k, fold))
    got = _S["memo"].get(key)
    if got is None:
        C2 = gf2.Circuit(cse=True)
        fresh: list[L] = []

        def sub(x):
            if fold and x.is_const():
                return x
            (b,) = C2.input(f"i{len(fresh)}", 1)
            fresh.append(b)
            return b

        def swap(x):
            if isinstance(x, gf2.Circuit):
                return C2
            bits: list = []
            _leaves(x, bits)
            return _rebuild(x, iter([sub(b) for b in bits])) if bits else x

        a2, k2 = tuple(swap(x) for x in a), {n_: swap(x) for n_, x in k.items()}
        fr2 = _Frame(C2)
        _S["frames"].append(fr2)
        _S["labels"].append(label)
        try:
            r2 = fn(*a2, **k2)
        finally:
            _S["frames"].pop()
            _S["labels"].pop()
        outs: list = []
        _leaves(r2, outs)
        ins = _ports([x for x in a if not isinstance(x, gf2.Circuit)] + list(k.values()), fold)
        t = to_type(fr2, fresh, outs, ins, [(1, len(outs))] if outs else [], label)
        d = t.digest()
        _S["types"].setdefault(d, t)
        _S["names"].setdefault(d, name)
        got = _S["memo"][key] = (d, r2, t)
    d, r2, t = got
    bits: list = []
    for x in list(a) + list(k.values()):
        if not isinstance(x, gf2.Circuit):
            _leaves(x, bits)
    args = [b for b in bits if not (fold and b.is_const())]
    live = [o for o in t.outputs if _made(o, t)]
    C = fr.C
    v0 = len(C.kind)
    idx = len(fr.calls)
    if live:
        for j in range(len(t.outputs)):
            C._new("call", (idx, j))
        fr.calls.append((d, args, v0, len(t.outputs)))

    def view(j, o):
        ws, c = o
        if _made(o, t):
            return L(frozenset((v0 + j,)), 0)
        f = ONE if c else ZERO
        for w in ws:
            f = f ^ args[w]
        return f

    return _rebuild(r2, iter([view(j, o) for j, o in enumerate(t.outputs)]))


def trace_types(build: Callable[[], gf2.Circuit], label: str = library.PROGRAM, fold: bool = True):
    """Run ``build`` (it makes a Circuit, its inputs and outputs) with every named call a type: ``(types, root digest, names)``.
    The root is the built circuit, its inputs every ``C.input`` bit in order, its outputs every output bit."""
    from . import boolean_export as BX

    BX._patch()
    saved = dict(_S)
    _S.update(on=True, frames=[], types={}, memo={}, fold=fold, names={}, labels=[label])
    real = gf2.Circuit.__init__
    first: list = []

    def init(self, cse=False):
        real(self, cse=True)
        if not first:
            first.append(self)
            _S["frames"].append(_Frame(self))

    gf2.Circuit.__init__ = init
    try:
        C = build()
    finally:
        gf2.Circuit.__init__ = real
    try:
        fr = _S["frames"][0]
        ins = [b for bits in C.inputs.values() for b in bits]
        outs = [b for bits in C.outputs.values() for b in bits]
        t = to_type(fr, ins, outs, [(1, len(bits)) for bits in C.inputs.values()], [(1, len(bits)) for bits in C.outputs.values()], label)
        d = t.digest()
        _S["types"].setdefault(d, t)
        return dict(_S["types"]), d, dict(_S["names"])
    finally:
        _S.clear()
        _S.update(saved)


def own_counts(t: CT.CircuitType) -> dict:
    """A type's own gates: its ANDs, and one XOR chain per distinct form it uses (|wires| - 1 XORs, and a NOT for the constant
    1), outside its calls and reads."""
    forms = set(t.outputs)
    ands = 0
    for it in t.items:
        if it[0] == CT.AND:
            ands += 1
            forms.update((it[1], it[2]))
        else:
            forms.update(it[2])
    return {"and": ands, "xor": sum(len(ws) - 1 for ws, _ in forms if ws), "not": sum(k for ws, k in forms if ws)}


def counts(types: dict, d: str, read_cost: Callable[[tuple], dict], memo: dict | None = None) -> dict:
    """AND / XOR / NOT of type ``d`` expanded: its ANDs, one XOR chain per distinct form it uses (|wires| - 1 XORs, and a NOT
    for the constant 1), its calls' and its table reads' (``read_cost(table entry)``)."""
    memo = {} if memo is None else memo
    if d in memo:
        return memo[d]
    t = types[d]
    c = own_counts(t)
    for it in t.items:
        if it[0] != CT.AND:
            sub = counts(types, t.calls[it[1]], read_cost, memo) if it[0] == CT.CALL else read_cost(t.tables[it[1]])
            for x in c:
                c[x] += sub[x]
    memo[d] = c
    return c


__all__ = ["call", "counts", "own_counts", "table_name", "table_sha512", "to_type", "trace_types"]
