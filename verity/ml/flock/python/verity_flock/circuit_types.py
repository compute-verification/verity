"""Circuit types: each distinct sub-circuit once, as its own ANDs, its table reads and calls into its parts (Project doc
`constant-api-public.md` §2.1–§2.2). The Boolean export's modules, the docs site's lazy expansion and the prover's layouts
share this one object.

**A type is its circuit, independent of where it's called.**
- Gates are shared (hash-consed) only inside a type, never across a call.
- A caller reads only a callee's outputs, never a bit inside it.
- A type holds no dead item: dead code is removed only against its own outputs.
- The prover lays a type out as rows at a placement. That's its layout, with a digest of its own; the type holds none of it.

**Content**, hashed as canonical JSON (RFC 8785, `verity.ml.circuits.codec.canonical_json`)::

    {"format": "verity/circuit-type/v1", "label": "library" | "program",
     "in": [[count, width], ...], "out": [[count, width], ...],       ports: arrays of elements, in order
     "calls": [type digest, ...], "tables": [[sha512, index_bits, value_bits], ...],
     "items": [...], "outputs": [form, ...]}

Wires 0 .. in_bits - 1 are the input bits: ports in order, elements in order, bits LSB first. Each item appends wires:

    [0, A, B]                AND of two forms                     one wire
    [3, k, [F_1 .. F_n]]     a call of calls[k] on n input forms  the callee's out_bits wires
    [4, t, [F_1 .. F_n]]     a read of tables[t] at n index forms its value_bits wires

A form is ``[[w, ...], c]``: the XOR of the ascending, distinct wires ``w`` and the constant ``c`` (0 or 1), so XOR and NOT
are free, as in `gf2`. Every constant inside a type is bound at registration (`verity.ml.circuits.constants`). ``calls`` and
``tables`` list each entry once, in order of first use. The op codes are the export's module codes; its XOR, NOT and tap
items don't occur.

**Digest:** ``SHA-512(b"verity/circuit-type/v1\\0" || canon(content))``. Names (a function's, a port's, a table's) are
metadata and never enter it. A table is named by the SHA-512 of its raw little-endian words, and its label is
`verity.ml.circuits.library.table_label` of that digest.

**Labels:** a type is ``library`` when it's the circuit of a library object (`verity.ml.circuits.library`), or reached only from
one. A library type calls only library types and reads only library tables.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits import library

from .gf2 import L

FORMAT = "verity/circuit-type/v1"
AND, CALL, READ = 0, 3, 4

Form = tuple[tuple[int, ...], int]
_SHA512 = re.compile(r"[0-9a-f]{128}")


@dataclass(frozen=True)
class CircuitType:
    label: str
    ins: tuple[tuple[int, int], ...]
    outs: tuple[tuple[int, int], ...]
    calls: tuple[str, ...]
    tables: tuple[tuple[str, int, int], ...]
    items: tuple
    outputs: tuple[Form, ...]

    @property
    def in_bits(self) -> int:
        return sum(n * w for n, w in self.ins)

    @property
    def out_bits(self) -> int:
        return sum(n * w for n, w in self.outs)

    def content(self) -> dict:
        def form(f):
            return [list(f[0]), f[1]]

        items = []
        for it in self.items:
            if it[0] == AND:
                items.append([AND, form(it[1]), form(it[2])])
            else:
                items.append([it[0], it[1], [form(f) for f in it[2]]])
        return {"format": FORMAT, "label": self.label, "in": [list(p) for p in self.ins], "out": [list(p) for p in self.outs],
                "calls": list(self.calls), "tables": [list(t) for t in self.tables], "items": items,
                "outputs": [form(f) for f in self.outputs]}

    def canonical(self) -> bytes:
        return canonical_json(self.content())

    def digest(self) -> str:
        return hashlib.sha512(FORMAT.encode() + b"\0" + self.canonical()).hexdigest()


def from_content(d: Mapping) -> CircuitType:
    """The type a content dict encodes: the inverse of `CircuitType.content`."""
    if d.get("format") != FORMAT:
        raise InvalidType(f"not a {FORMAT} type")

    def form(f):
        return (tuple(f[0]), f[1])

    items = []
    for it in d["items"]:
        items.append((AND, form(it[1]), form(it[2])) if it[0] == AND else (it[0], it[1], tuple(form(f) for f in it[2])))
    return CircuitType(d["label"], tuple(tuple(p) for p in d["in"]), tuple(tuple(p) for p in d["out"]), tuple(d["calls"]),
                       tuple(tuple(t) for t in d["tables"]), tuple(items), tuple(form(f) for f in d["outputs"]))


class InvalidType(ValueError):
    """A type that breaks the format's rules."""


def _item_forms(it) -> tuple:
    return (it[1], it[2]) if it[0] == AND else it[2]


def validate(t: CircuitType, callee: Callable[[str], CircuitType]) -> None:
    """Refuse a type that breaks the rules above. ``callee(digest)`` gives a called type, already validated."""

    def bad(why):
        raise InvalidType(why)

    if t.label not in library.LABELS:
        bad(f"label {t.label!r} is neither library nor program")
    for n, w in (*t.ins, *t.outs):
        if not (isinstance(n, int) and isinstance(w, int) and n >= 0 and w >= 1):
            bad(f"a port is not [count >= 0, width >= 1]: {[n, w]}")
    for d in t.calls:
        if not _SHA512.fullmatch(d):
            bad(f"a call names {d!r}, not a type digest")
    if len(set(t.calls)) != len(t.calls) or len({x[0] for x in t.tables}) != len(t.tables):
        bad("calls and tables list each entry once")
    for sha, ib, vb in t.tables:
        if not _SHA512.fullmatch(sha) or not (1 <= ib <= 32 and 1 <= vb <= 32):
            bad(f"a table is not [sha512, index_bits, value_bits]: {[sha, ib, vb]}")
        if library.table_label(sha) == library.LIBRARY:
            lt = library.table(sha)
            if (lt.index_bits, lt.value_bits) != (ib, vb):
                bad(f"library table {lt.name} has {lt.index_bits} index and {lt.value_bits} value bits")
        elif t.label == library.LIBRARY:
            bad("a library type reads a program table")

    def check_form(f, n):
        ws, c = f
        if c not in (0, 1) or list(ws) != sorted(set(ws)) or any(not 0 <= w < n for w in ws):
            bad(f"a form is not [ascending distinct earlier wires, 0 or 1]: {[list(ws), c]}")

    n, first_call, first_table, spans = t.in_bits, [], [], []
    for it in t.items:
        op = it[0]
        if op == AND:
            width = 1
        elif op == CALL:
            if not 0 <= it[1] < len(t.calls):
                bad(f"a call of calls[{it[1]}]")
            u = callee(t.calls[it[1]])
            if t.label == library.LIBRARY and u.label != library.LIBRARY:
                bad("a library type calls a program type")
            if len(it[2]) != u.in_bits:
                bad(f"a call passes {len(it[2])} bits to a type of {u.in_bits}")
            if it[1] not in first_call:
                first_call.append(it[1])
            width = u.out_bits
        elif op == READ:
            if not 0 <= it[1] < len(t.tables):
                bad(f"a read of tables[{it[1]}]")
            _, ib, vb = t.tables[it[1]]
            if len(it[2]) != ib:
                bad(f"a read passes {len(it[2])} index bits to a table of {ib}")
            if it[1] not in first_table:
                first_table.append(it[1])
            width = vb
        else:
            bad(f"item op {op}: only AND (0), call (3) and read (4)")
        for f in _item_forms(it):
            check_form(f, n)
        spans.append((n, width, it))
        n += width
    if first_call != list(range(len(t.calls))) or first_table != list(range(len(t.tables))):
        bad("calls and tables are not listed in order of first use")
    if len(t.outputs) != t.out_bits:
        bad(f"{len(t.outputs)} output forms for {t.out_bits} output bits")
    for f in t.outputs:
        check_form(f, n)
    used = {w for f in t.outputs for w in f[0]}
    for lo, width, it in reversed(spans):
        if not any(w in used for w in range(lo, lo + width)):
            bad(f"a dead item at wire {lo}: nothing it makes is read")
        used.update(w for f in _item_forms(it) for w in f[0])


def evaluate(types: Mapping[str, CircuitType], digest: str, inputs: Sequence[int], T: int,
             word: Callable[[str, int], int]) -> list[int]:
    """The output bits of type ``digest`` on bit-sliced inputs: each value a T-bit int, one bit per vector. ``word(sha512, i)``
    is a table's word at index i."""
    mask = (1 << T) - 1

    def run(t: CircuitType, ins: Sequence[int]) -> list[int]:
        val = list(ins)

        def ev(f):
            x = mask if f[1] else 0
            for w in f[0]:
                x ^= val[w]
            return x

        for it in t.items:
            if it[0] == AND:
                val.append(ev(it[1]) & ev(it[2]))
            elif it[0] == CALL:
                val.extend(run(types[t.calls[it[1]]], [ev(f) for f in it[2]]))
            else:
                sha, _, vb = t.tables[it[1]]
                idx = [ev(f) for f in it[2]]
                out = [0] * vb
                for v in range(T):
                    wv = word(sha, sum(((x >> v) & 1) << b for b, x in enumerate(idx)))
                    for j in range(vb):
                        out[j] |= ((wv >> j) & 1) << v
                val.extend(out)
        return [ev(f) for f in t.outputs]

    return run(types[digest], [x & mask for x in inputs])


def from_gf2(C, inputs: Sequence[L], outputs: Sequence[L], ins: Sequence[tuple[int, int]], outs: Sequence[tuple[int, int]],
             label: str, tables: Mapping[str, tuple[str, int, int]]) -> CircuitType:
    """A flat type (no calls) from a `gf2.Circuit`. ``inputs`` are its input bits, single committed bits with ports in order,
    and ``outputs`` its output forms. Only the items the outputs reach are kept, in the order the circuit made them; a read is
    kept whole when any of its bits is. ``tables`` maps a table name `C.lookup` reads to ``(sha512, index_bits, value_bits)``."""
    wire: dict[int, int] = {}
    for i, f in enumerate(inputs):
        if f.c or len(f.s) != 1:
            raise InvalidType("an input is not a single committed bit")
        (v,) = tuple(f.s)
        wire[v] = i
    group_bits: dict[int, list[int]] = {}
    for v, kind in enumerate(C.kind):
        if kind == "lookup":
            group_bits.setdefault(C.data[v][0], []).append(v)
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
            live.update(group_bits[g])
            for f in C.lookups[g][1]:
                stack.extend(f.s)
        elif kind == "in":
            raise InvalidType(f"input bit {v} ({C.data[v]}) is read but is not one of the type's inputs")
        else:
            raise InvalidType(f"a {kind} bit: types have no hints")

    def form(f):
        return (tuple(sorted(wire[v] for v in f.s)), f.c)

    n, items, table_list, done = len(inputs), [], [], set()
    for v in sorted(live):
        kind = C.kind[v]
        if kind == "and":
            a, b = C.data[v]
            items.append((AND, form(a), form(b)))
            wire[v] = n
            n += 1
        elif C.data[v][0] not in done:
            g = C.data[v][0]
            done.add(g)
            name, idx, width = C.lookups[g]
            entry = tuple(tables[name])
            if width != entry[2]:
                raise InvalidType(f"table {name}: a read of {width} bits, not {entry[2]}")
            if entry not in table_list:
                table_list.append(entry)
            items.append((READ, table_list.index(entry), tuple(form(f) for f in idx)))
            for u in group_bits[g]:
                wire[u] = n + C.data[u][1]
            n += width
    return CircuitType(label, tuple(tuple(p) for p in ins), tuple(tuple(p) for p in outs), (), tuple(table_list), tuple(items),
                       tuple(form(f) for f in outputs))


def expand(types: Mapping[str, CircuitType], digest: str) -> CircuitType:
    """Type ``digest`` flattened: every call replaced by its callee's items, recursively. The reference for lazy expansion."""
    t = types[digest]
    items, table_list, n = [], [], t.in_bits

    def inline(u: CircuitType, args: list[Form]) -> list[Form]:
        nonlocal n
        local: list[Form] = list(args)

        def form(f):
            ws, c = set(), f[1]
            for w in f[0]:
                ws ^= set(local[w][0])
                c ^= local[w][1]
            return (tuple(sorted(ws)), c)

        for it in u.items:
            if it[0] == AND:
                items.append((AND, form(it[1]), form(it[2])))
                local.append(((n,), 0))
                n += 1
            elif it[0] == CALL:
                local.extend(inline(types[u.calls[it[1]]], [form(f) for f in it[2]]))
            else:
                entry = u.tables[it[1]]
                if entry not in table_list:
                    table_list.append(entry)
                items.append((READ, table_list.index(entry), tuple(form(f) for f in it[2])))
                local.extend(((n + j,), 0) for j in range(entry[2]))
                n += entry[2]
        return [form(f) for f in u.outputs]

    outs = inline(t, [((w,), 0) for w in range(t.in_bits)])
    return CircuitType(t.label, t.ins, t.outs, (), tuple(table_list), tuple(items), tuple(outs))


__all__ = ["AND", "CALL", "FORMAT", "READ", "CircuitType", "InvalidType", "evaluate", "expand", "from_content", "from_gf2",
           "validate"]
