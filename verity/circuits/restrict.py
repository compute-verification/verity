"""C_S on bits, compacted: Core's `Circuit.restrict` of C to a proof block's units, written in the block's own numbering.

**C's index.** C's gates are numbered by PROTOCOL.md §6's layout of its root: bodies laid out in node order, recursively; a
primitive node is one gate, a call node its callee's gates, a batch's members (a scan's iterations) consecutive in member
order, each its callee's gates, a batch of a primitive one gate a member. An input node (`Input<w>_v1`, or
`RegisteredInput<w>_v1` for a registered value) is a primitive node of the root: one gate, at its node's place.

**C on bits** numbers the bits of C's gates, gate by gate in C's index order: gate g's bit gates come after every bit gate of
the gates before g (`Bits.base`), and are, in place:
- an input gate of w bits, of either binding: w one-bit input gates of that binding, bit j at offset `split(w)[j]`, which is
  j (least significant bit first);
- a gate of the Boolean profile (`And<n>_v1`, `Xor<n>_v1`, `Not_v1`, `Const1[0x0]_v1`, `Const1[0x1]_v1`): itself;
- any other primitive's gate: the gates of its Boolean body (the descriptor's `boolean` map, §4.9) in the body's own §6
  layout, the body's parameters taking no gate: its parameter bits are the bits of the gate's operands, in argument order
  and each operand's bits least significant first, and bit j of the gate's value is the body's j-th returned bit (one of its
  gates, or a parameter bit it passes through).
Every gate of C on bits reads only earlier gates.

**The rule.** For a block S, the gates of C on bits inside its units (each unit the instance the partition's template query
numbers, `TemplatePartition.gates`): the file's gates are S's gates in increasing index of C on bits; its input ports are
in(S), the gates outside S that a gate of S reads, in increasing index; its output ports are out(S), the gates of S that a gate
outside S reads or that C returns, in increasing index. The file numbers the input ports 0..I-1 and S's gates I..I+K-1 in
that order, and an output port names its gate's file number. `Block.inputs` and `Block.gates` are the file's numbering into C
on bits (ports, then gates), injective and increasing within each part.

**The text** (`Block.text`), ASCII, one line each, `\\n`-terminated: `verity/block-circuit/v1`; `I K O`; per input port its
binding (`run`, `registration`, or `wire` for a computed gate of C); per gate its kind and its operands' file numbers, spaces
between; per output port `out` and its file number.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from . import boolean as BL
from . import partition_object as PO
from .codec import DecodedProgram, decode_program, lookup_primitive
from .defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
from .layout import Circuit
from .program import REGISTRATION, RUN, input_id, input_prim
from .refs import Coll
from .types import Value, type_from_json

FORMAT = "verity/block-circuit/v1"
WIRE = "wire"


class Refused(ValueError):
    """What `restrict` does not cover, by code."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


def split(w: int) -> tuple[int, ...]:
    """Where bit j of a w-bit input gate lies among its w one-bit gates: in place, least significant bit first."""
    return tuple(range(w))


@dataclass(frozen=True)
class Block:
    units: tuple[tuple[int, int], ...]
    inputs: tuple[int, ...]
    gates: tuple[tuple[int, int], ...]
    outputs: tuple[int, ...]
    text: bytes

    def gate_indices(self) -> list[int]:
        return [g for lo, hi in self.gates for g in range(lo, hi)]


def _unevaluated(*_a: int) -> int:
    raise ValueError("decoded from C for its structure only")


def _declared(desc: Mapping[str, Any]) -> DefinitionTable:
    reg = DefinitionTable()
    for fid, d in desc["definitions"].items():
        if d.get("form") != "primitive" or input_id(fid) is not None:
            continue
        try:
            lookup_primitive(DefinitionTable(), fid)
            continue
        except KeyError:
            pass
        name, _, version = fid.rpartition("_v")
        reg.add(PrimitiveDefinition(name, int(version), tuple((n, type_from_json(t)) for n, t in d["params"]),
                                    type_from_json(d["ret"]), _unevaluated, register=False))
    return reg


def program(c: bytes) -> DecodedProgram:
    """C decoded from its bytes alone: every primitive as C declares it."""
    try:
        desc = json.loads(c)
        return decode_program(desc, _declared(desc))
    except (ValueError, KeyError, TypeError) as e:
        raise Refused("E_DESCRIPTOR", str(e)) from e


@dataclass(frozen=True)
class _Body:
    params: int
    gates: tuple[tuple[str, tuple[int, ...]], ...]
    ret: tuple[int, ...]
    reads: frozenset[int]


def _kind(fn: Any) -> str:
    if not isinstance(fn, PrimitiveDefinition) or not (BL.is_gate(fn) or BL.is_constant_bit(fn)):
        raise Refused("E_KIND", f"{getattr(fn, 'id', fn)!r} is no gate of the Boolean profile")
    return fn.id


@lru_cache(maxsize=None)
def _body(fn: Any) -> _Body:
    """``fn``'s §6 layout under one-bit input gates for its parameter bits: its gates' kinds and operands, an operand or a
    returned bit as a gate of the body (>= 0) or parameter bit p (-1 - p), and the parameter bits its gates read."""
    params, ret = fn.signature()
    bit = input_prim(1)

    def body(B, S):
        return B.call(fn, *[Coll(t, B.batch(bit, axes=(), n=t.leaves).refs) for _, t in params])

    wrap = bind(CompositeDefinition("RestrictBody", 1, (), lambda S: ((), ret), body, register=False))
    P = sum(t.leaves for _, t in params)
    enc = (lambda o: o - P if o >= P else -1 - o)
    C = Circuit(wrap)
    gates = tuple((_kind(g.prim), tuple(enc(o) for o in g.operands)) for g in C.gates(P, C.size))
    return _Body(P, gates, tuple(enc(o) for o in C.root_outputs()),
                 frozenset(-1 - o for _, ops in gates for o in ops if o < 0))


def _width(fn: PrimitiveDefinition) -> int:
    _, ret = fn.signature()
    if not isinstance(ret, Value):
        raise Refused("E_LEAVES", f"{fn.id} returns {ret!r}, not one value")
    return ret.w


class Bits:
    """C on bits (module docstring), gate by gate: each gate of C's first bit (`base`), its count of bit gates, and the bit
    gates its value's bits are (`out`)."""

    def __init__(self, D: DecodedProgram) -> None:
        self.D, self.C = D, D.circuit
        self.word = list(self.C.gates())
        self.base: list[int] = []
        self.out: list[tuple[int, ...]] = []
        self.kind: list[tuple[str, Any]] = []
        at = 0
        for g in self.word:
            fn, w = g.prim, _width(g.prim)
            self.base.append(at)
            if (inp := input_id(fn.id)) is not None:
                self.kind.append(("input", inp[0]))
                self.out.append(tuple(at + j for j in split(w)))
                at += w
            elif w == 1 and (BL.is_gate(fn) or BL.is_constant_bit(fn)):
                self.kind.append(("gate", fn.id))
                self.out.append((at,))
                at += 1
            else:
                bfn = D.boolean.get(fn.id)
                if bfn is None:
                    raise Refused("E_NO_BODY", f"C names no Boolean body for {fn.id}")
                b = _body(bfn)
                ops = self.operand_bits(g.index)
                if len(ops) != b.params or len(b.ret) != w:
                    raise Refused("E_BODY", f"{fn.id}'s body has {b.params} parameter bits and {len(b.ret)} return bits")
                self.kind.append(("body", b))
                self.out.append(tuple(at + r if r >= 0 else ops[-1 - r] for r in b.ret))
                at += len(b.gates)
        self.size = at

    def operand_bits(self, g: int) -> list[int]:
        return [b for o in self.word[g].operands for b in self.out[o]]

    def read_bits(self, g: int) -> list[int]:
        """The bit gates outside gate g's own that its bit gates read."""
        k, v = self.kind[g]
        if k == "input":
            return []
        ops = self.operand_bits(g)
        return ops if k == "gate" else [ops[p] for p in sorted(v.reads)]

    def lines(self, g: int) -> list[tuple[str, list[int]]]:
        """Gate g's bit gates in order: kind and operands as indices of C on bits."""
        k, v = self.kind[g]
        if k == "input":
            raise Refused("E_INPUT_IN_BLOCK", f"gate {g} of C is an input gate inside the block")
        ops = self.operand_bits(g)
        if k == "gate":
            return [(v, ops)]
        base = self.base[g]
        return [(kind, [base + o if o >= 0 else ops[-1 - o] for o in os]) for kind, os in v.gates]

    def gate_of(self, b: int) -> int:
        return bisect.bisect_right(self.base, b) - 1


def unit_gates(D: DecodedProgram, partition: Mapping[str, Any], units: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The gates of C (§6) of ``units``, ascending ``[lo, hi)`` ranges of the partition's template query's units, as gate ranges."""
    q = partition.get("query") if isinstance(partition, Mapping) else None
    if not isinstance(q, Mapping) or (q.get("name"), q.get("version")) not in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        raise Refused("E_QUERY", f"restrict covers the template queries, not {q!r}")
    try:
        tp = PO.TemplatePartition(D, q)
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    us = [tuple(u) for u in units]
    if not us or any(not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi) for lo, hi in us) or \
            any(a[1] > b[0] for a, b in zip(us, us[1:])) or us[-1][1] > tp.population:
        raise Refused("E_UNITS", f"units {us!r}: not ascending disjoint nonempty ranges within 0..{tp.population}")
    out: list[tuple[int, int]] = []
    for lo, hi in us:
        for u in range(lo, hi):
            a, b = tp.gates(u)
            if out and out[-1][1] == a:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]], bits: Bits | None = None) -> Block:
    """`compact (restrict C S)` for S the gates of ``units`` (module docstring: the rule and the text)."""
    bits = Bits(program(c)) if bits is None else bits
    words = unit_gates(bits.D, partition, units)
    inside = [False] * len(bits.word)
    for lo, hi in words:
        for g in range(lo, hi):
            inside[g] = True
    kept = tuple((bits.base[lo], bits.base[hi] if hi < len(bits.base) else bits.size) for lo, hi in words)

    def in_s(b: int) -> bool:
        return inside[bits.gate_of(b)]

    body: list[tuple[str, list[int]]] = []
    for lo, hi in words:
        for g in range(lo, hi):
            body += bits.lines(g)
    ins = sorted({o for _, ops in body for o in ops if not in_s(o)})
    read = {b for g in range(len(bits.word)) if not inside[g] for b in bits.read_bits(g)}
    read |= {b for g in bits.C.root_outputs() for b in bits.out[g]}
    outs = sorted(b for b in read if in_s(b))
    num = {b: i for i, b in enumerate(ins)}
    at = len(ins)
    for lo, hi in kept:
        for b in range(lo, hi):
            num[b] = at
            at += 1
    lines = [FORMAT, f"{len(ins)} {len(body)} {len(outs)}"]
    for b in ins:
        k, v = bits.kind[bits.gate_of(b)]
        lines.append(v if k == "input" else WIRE)
    lines += [" ".join([kind, *(str(num[o]) for o in ops)]) for kind, ops in body]
    lines += [f"out {num[b]}" for b in outs]
    return Block(tuple(tuple(u) for u in units), tuple(ins), kept, tuple(outs), ("\n".join(lines) + "\n").encode("ascii"))


def evaluate(text: bytes, inputs: Sequence[int]) -> list[int]:
    """The output ports' bits of the block circuit ``text`` on its input ports' bits ``inputs``."""
    it = iter(text.decode("ascii").split("\n"))
    if next(it) != FORMAT:
        raise Refused("E_FORMAT", "not a block circuit")
    I, K, O = map(int, next(it).split())
    if len(inputs) != I:
        raise Refused("E_FORMAT", f"{len(inputs)} input bits for {I} ports")
    v = [b & 1 for b in inputs]
    for _ in range(I):
        next(it)
    for _ in range(K):
        kind, *ops = next(it).split()
        x = [v[int(o)] for o in ops]
        if kind.startswith("And"):
            v.append(int(all(x)))
        elif kind.startswith("Xor"):
            v.append(sum(x) & 1)
        elif kind == "Not_v1":
            v.append(1 - x[0])
        elif kind.startswith("Const1["):
            v.append(int(kind == "Const1[0x1]_v1"))
        else:
            raise Refused("E_KIND", kind)
    return [v[int(next(it).split()[1])] for _ in range(O)]


__all__ = ["Bits", "Block", "FORMAT", "REGISTRATION", "RUN", "Refused", "WIRE", "evaluate", "program", "restrict", "split",
           "unit_gates"]
