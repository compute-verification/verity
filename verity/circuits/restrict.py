"""C_S compacted: Core's `Circuit.restrict` of C to a proof block's units, as a Program of the block's own (`restrict`).

**C's index, the one that names things.** C's gates are its descriptor's word-level gates, numbered by PROTOCOL.md §6's
layout of its root: bodies laid out in node order, recursively; a primitive node is one gate, a call node its callee's gates,
a batch's members (a scan's iterations) consecutive in member order, each its callee's gates, a batch of a primitive one gate
a member. An input node (`Input<w>_v1`, or `RegisteredInput<w>_v1` for a registered value) is a primitive node of the root,
one gate a member, at its node's place. A gate has its primitive's width. This index, and only it, names a block (`words`),
a boundary (in(S) and out(S): `inputs`, `outputs`) and a leaf; `Block.to_c_gate` maps the block's descriptor gates into it.
Units, and a partition's blocks, are counted in the population's own order (`TemplatePartition`, `fit.FitPartition`), not in
gates; under `Q_fit` a block may also be a piece of one activation (`fit`), named by its path and its nodes or members.

**C on bits, for a statement and a count only.** It never names a block, a boundary or a leaf. It numbers the bits of C's
gates, gate by gate in C's index order: gate g's bit gates come after every bit gate of
the gates before g (`Bits.base`), and are, in place:
- an input gate of w bits, of either binding: w one-bit input gates of that binding, bit j at offset `split(w)[j]`, which is
  j (least significant bit first);
- a gate of the Boolean profile (`And<n>_v1`, `Xor<n>_v1`, `Not_v1`, `Const1[0x0]_v1`, `Const1[0x1]_v1`): itself;
- any other primitive's gate: the gates of its Boolean body (the descriptor's `boolean` map, §4.9) in the body's own §6
  layout, the body's parameters taking no gate: its parameter bits are the bits of the gate's operands, in argument order
  and each operand's bits least significant first, and bit j of the gate's value is the body's j-th returned bit (one of its
  gates, or a parameter bit it passes through).
Every gate of C on bits reads only earlier gates.

**The block.** S is the gates of C inside the block's units, each unit the instance the partition's query numbers
(`TemplatePartition.gates`, `fit.FitPartition.gates`), or inside its piece. in(S) is the gates outside S that a gate of S
reads, out(S) the gates of S that a gate outside S reads or that C returns, each in increasing index of C (`ports`, from the
descriptor by `boundary`, never by a pass over C).

**The object** (`restrict`): the Program whose root
- has a parameter for each maximal run of in(S) of one width, one binding and, under `Q_fit` with the partition's blocks,
  one writer row (`fit`'s rows), in order, `p<i>` of type `Array<n, Value<w>>`, bound at registration when its gates are
  `RegisteredInput` gates of C and per run otherwise;
- then has the block's nodes of its activation (the root's for units) in C's order, each as C has it when the block holds
  all of its members, and one call node a member, in member order, when it holds some (a unit is one member); each argument
  leaf names the parameter leaf
  of its gate of in(S), or the earliest leaf of the block's nodes whose gate it is, the leaves spelled as the maximal runs
  of consecutive leaves of one node or parameter, each an `Affine`, concatenated;
- returns, for each maximal run of out(S) of one width, `Array<n, Value<w>>`, in a Tuple, each gate as the earliest leaf of
  the block's nodes whose gate it is;
and whose Definitions and Boolean map are C's (a word primitive's body is C's). Its bytes are the codec's
(`descriptor_chunks`). Laid out by §6 as C is, its gates are in(S), one input gate a leaf, then S's gates in order
(`Block.to_c_gate`). On bits, its gates are in(S)'s bits, each gate's least significant first, then S's bit gates in order
(`Block.to_c_bit`).
`listing` writes the block on bits gate by gate, a test's reference: `verity/block-circuit/v1`; `I K O`; per input bit its
binding (`run`, `registration`, or `wire` for a computed gate of C); per gate its kind and its operands; per output `out`
and its gate, all in the block's numbering, one ASCII line each.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Mapping, Sequence

from . import boolean as BL
from . import boundary as BD
from . import fit as FT
from . import partition_object as PO
from . import program as P
from .codec import DecodedProgram, decode_program, descriptor_chunks, lookup_primitive
from .defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
from .layout import Circuit, Scope, resolve
from .parts import Part, Unsupported
from .program import REGISTRATION, RUN, input_id, input_prim
from .refs import Affine, Coll, concat
from .types import Array, Tuple, Value, remove_axis, type_from_json

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
    n = sum(t.leaves for _, t in params)
    def enc(o: int) -> int:
        return o - n if o >= n else -1 - o

    C = Circuit(wrap)
    gates = tuple((_kind(g.prim), tuple(enc(o) for o in g.operands)) for g in C.gates(n, C.size))
    return _Body(n, gates, tuple(enc(o) for o in C.root_outputs()),
                 frozenset(-1 - o for _, ops in gates for o in ops if o < 0))


def _width(fn: PrimitiveDefinition) -> int:
    _, ret = fn.signature()
    if not isinstance(ret, Value):
        raise Refused("E_LEAVES", f"{fn.id} returns {ret!r}, not one value")
    return ret.w


class _Seq:
    """A read-only sequence of ``n`` items, each computed when read: `Bits`' per-gate tables without a pass over C."""

    def __init__(self, n: int, f) -> None:
        self.n, self.f = n, f

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.f(i)

    def __iter__(self):
        return (self.f(i) for i in range(self.n))


class Bits:
    """C on bits (module docstring), gate by gate, each entry computed when read from the descriptor's structure, never by a
    pass over C: each gate of C's first bit (`base`), the bit gates its value's bits are (`out`), its kind and width."""

    def __init__(self, D: DecodedProgram) -> None:
        self.D, self.C = D, D.circuit
        n = self.C.size
        self._of: dict[int, tuple[Any, int]] = {}
        self._prefix: dict[int, tuple[Any, list[int]]] = {}
        self._kinds: dict[int, tuple[str, Any]] = {}
        self._outs: dict[int, tuple[int, ...]] = {}
        self.word = _Seq(n, self.C.gate)
        self.base = _Seq(n + 1, self._base)
        self.out = _Seq(n, self._out)
        self.kind = _Seq(n, self._kind)
        self.width = _Seq(n, lambda g: _width(self.C.gate(g).prim))
        self.size = self._bits(self.C.root)

    def _bits(self, fn: Any) -> int:
        """The bit gates of one activation (or gate) of ``fn``."""
        hit = self._of.get(id(fn))
        if hit is not None and hit[0] is fn:
            return hit[1]
        if isinstance(fn, PrimitiveDefinition):
            w = _width(fn)
            if (inp := input_id(fn.id)) is not None:
                n = inp[1]
            elif w == 1 and (BL.is_gate(fn) or BL.is_constant_bit(fn)):
                n = 1
            else:
                bfn = self.D.boolean.get(fn.id)
                if bfn is None:
                    raise Refused("E_NO_BODY", f"C names no Boolean body for {fn.id}")
                n = len(_body(bfn).gates)
        else:
            n = self._prefixes(fn)[-1]
        self._of[id(fn)] = (fn, n)
        return n

    def _prefixes(self, fn: Any) -> list[int]:
        hit = self._prefix.get(id(fn))
        if hit is not None and hit[0] is fn:
            return hit[1]
        out, at = [0], 0
        for node in fn.body.nodes:
            at += (node.n if node.form in ("batch", "scan") else 1) * self._bits(node.fn)
            out.append(at)
        self._prefix[id(fn)] = (fn, out)
        return out

    def _base(self, g: int) -> int:
        if g == self.C.size:
            return self.size
        scope, acc = self.C.root_scope, 0
        while True:
            body = scope.spec.body
            k = body.node_at(g - scope.offset)
            node = body.nodes[k]
            node_off = scope.offset + body.offsets[k]
            m = 0 if node.form in ("primitive", "call") else (g - node_off) // node.fn.gates
            acc += self._prefixes(scope.spec)[k] + m * self._bits(node.fn)
            if isinstance(node.fn, PrimitiveDefinition):
                return acc
            scope = scope.child(k, m)

    def _kind(self, g: int) -> tuple[str, Any]:
        hit = self._kinds.get(g)
        if hit is None:
            fn = self.C.gate(g).prim
            if (inp := input_id(fn.id)) is not None:
                hit = ("input", inp[0])
            elif _width(fn) == 1 and (BL.is_gate(fn) or BL.is_constant_bit(fn)):
                hit = ("gate", fn.id)
            else:
                bfn = self.D.boolean.get(fn.id)
                if bfn is None:
                    raise Refused("E_NO_BODY", f"C names no Boolean body for {fn.id}")
                hit = ("body", _body(bfn))
            self._kinds[g] = hit
        return hit

    def _out(self, g: int) -> tuple[int, ...]:
        hit = self._outs.get(g)
        if hit is not None:
            return hit
        k, v = self._kind(g)
        at = self._base(g)
        if k == "input":
            hit = tuple(at + j for j in split(self.width[g]))
        elif k == "gate":
            hit = (at,)
        else:
            w = self.width[g]
            word = self.C.gate(g)
            if sum(self.width[o] for o in word.operands) != v.params or len(v.ret) != w:
                raise Refused("E_BODY", f"{word.prim.id}'s body has {v.params} parameter bits and {len(v.ret)} return bits")
            ops = self.operand_bits(g) if any(r < 0 for r in v.ret) else ()
            hit = tuple(at + r if r >= 0 else ops[-1 - r] for r in v.ret)
        self._outs[g] = hit
        return hit

    def operand_bits(self, g: int) -> list[int]:
        return [b for o in self.C.gate(g).operands for b in self.out[o]]

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
        """The gate of C whose bit gates hold bit gate ``b``."""
        if not 0 <= b < self.size:
            raise IndexError(b)
        scope, acc = self.C.root_scope, 0
        while True:
            body = scope.spec.body
            pre = self._prefixes(scope.spec)
            k = bisect.bisect_right(pre, b - acc) - 1
            node = body.nodes[k]
            per = self._bits(node.fn)
            m = (b - acc - pre[k]) // per
            acc += pre[k] + m * per
            if isinstance(node.fn, PrimitiveDefinition):
                return scope.offset + body.offsets[k] + m
            scope = scope.child(k, m)

    def bits_of(self, gates: Sequence[int]) -> list[int]:
        """Each gate's bits, least significant first, gate by gate; refused where two are one bit gate."""
        out = [b for g in gates for b in self.out[g]]
        if len(set(out)) != len(out):
            raise Refused("E_PORT_TWICE", "two port bits are one gate of C on bits")
        return out


@dataclass(frozen=True)
class _Where:
    """A block in C: the activation its gates lie in (`scope`), its nodes there, each ``(k, first member, end member)``
    (`kept`), and the same gates as a `Part`."""
    scope: Scope
    kept: tuple[tuple[int, int, int], ...]
    part: Part


def _partition(D: DecodedProgram, partition: Mapping[str, Any]) -> Any:
    q = partition.get("query") if isinstance(partition, Mapping) else None
    kind = (q.get("name"), q.get("version")) if isinstance(q, Mapping) else None
    if kind not in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES, FT.Q_FIT):
        raise Refused("E_QUERY", f"restrict covers the template queries and Q_fit, not {q!r}")
    try:
        return FT.FitPartition(D, q) if kind == FT.Q_FIT else PO.TemplatePartition(D, q)
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e


def _where(D: DecodedProgram, tp: Any, units: Any) -> _Where:
    """``units`` (ascending ``[lo, hi)`` ranges of the partition's units, or under `Q_fit` a piece, `fit`'s entry) in C."""
    if isinstance(units, Mapping):
        if not isinstance(tp, FT.FitPartition):
            raise Refused("E_QUERY", "a piece below a unit is a block of Q_fit only")
        if not FT.entry_form(dict(units)) or "path" not in units:
            raise Refused("E_UNITS", f"{units!r}: not a piece (`fit`'s module docstring)")
        try:
            part = FT.part_of(D, tp, units)
        except ValueError as e:
            raise Refused("E_UNITS", str(e)) from e
    else:
        us = [tuple(u) for u in units]
        if not us or any(not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi) for lo, hi in us) or \
                any(x[1] > y[0] for x, y in zip(us, us[1:])) or us[-1][1] > tp.population:
            raise Refused("E_UNITS", f"units {us!r}: not ascending disjoint nonempty ranges within 0..{tp.population}")
        part = Part(D.circuit.root_scope, frozenset(tp.locate(u) for lo, hi in us for u in range(lo, hi)))
    body, kept = part.scope.spec.body, []
    for k, m in sorted(part.items):
        node = body.nodes[k]
        root = part.scope.parent is None
        if (root and node.form not in ("call", "batch")) or node.form == "scan":
            raise Refused("E_FORM", f"node {k} of {part.scope.spec.id} is a {node.form}: a block holds calls, batches and "
                                    "primitives of one activation, and a unit is a call or a batch member")
        if kept and kept[-1][0] == k and kept[-1][2] == m:
            kept[-1] = (k, kept[-1][1], m + 1)
        else:
            kept.append((k, m, m + 1))
    return _Where(part.scope, tuple(kept), part)


def unit_gates(D: DecodedProgram, partition: Mapping[str, Any], units: Any) -> list[tuple[int, int]]:
    """The gates of C (§6) of ``units``, ascending ``[lo, hi)`` ranges of the partition's units (or, under `Q_fit`, a piece),
    as gate ranges."""
    return _where(D, _partition(D, partition), units).part.intervals()


def ports(part: Part) -> tuple[list[int], list[int]]:
    """in(S) and out(S) of S, the part's gates of C, each in increasing index, from the descriptor (`boundary`), never by a
    pass over C."""
    ins, outs = BD.in_gates(part), BD.out_gates(part)
    for x in (ins, outs):
        if isinstance(x, Unsupported):
            raise Refused("E_BOUNDARY", f"{x.reason}: {x.detail}")
    return sorted(ins), sorted(outs)


def _rows(D: DecodedProgram, tp: Any, partition: Mapping[str, Any], units: Any) -> Any:
    """Under `Q_fit` with the partition's ``blocks``: a gate of in(S) -> its writer row (`fit`'s rows), from the blocks before
    this one; None otherwise. Refused where ``units`` is no block of the partition."""
    entries = partition.get(PO.BLOCKS) if isinstance(tp, FT.FitPartition) else None
    if entries is None:
        return None
    me = dict(units) if isinstance(units, Mapping) else ([list(u) for u in units][0] if len(units) == 1 else None)
    if me not in entries:
        raise Refused("E_UNITS", f"{units!r} is no block of the partition's blocks")
    fit = FT.Fit(D)
    for e in entries[:entries.index(me)]:
        p = FT.part_of(D, tp, e)
        try:
            _ins, outs = fit._ports(p)
        except FT.Unfit as x:
            raise Refused("E_BOUNDARY", str(x)) from None
        fit.finish(p, outs, [BD.gate_width(D, g) for g in outs])

    def key(g: int) -> tuple:
        got = fit._input_row(g) or fit.writers.row(g)
        if got is None:
            raise Refused("E_ORDER", f"gate {g} is in no block before this one and no input")
        return got[0]
    return key


def _runs(gates: Sequence[int], key) -> list[list[int]]:
    out: list[list[int]] = []
    for g in gates:
        if out and key(out[-1][-1]) == key(g):
            out[-1].append(g)
        else:
            out.append([g])
    return out


def _spell(items: Sequence[tuple]) -> Any:
    parts: list[list] = []
    for s, i, leaf in items:
        if parts and parts[-1][:2] == [s, i] and parts[-1][2] + parts[-1][3] == leaf:
            parts[-1][3] += 1
        else:
            parts.append([s, i, leaf, 1])
    return concat([Affine(s, i, b, n) for s, i, b, n in parts])


def _binding(bits: Bits, g: int) -> str:
    k, v = bits.kind[g]
    return v if k == "input" else RUN


@dataclass(frozen=True)
class Block:
    """`restrict`'s value: the block's Program and its bytes, S as C's gate ranges (`words`), in(S) and out(S) (`inputs`,
    `outputs`), and the bits of C on bits behind them (`bits`). ``units`` is the block as given: unit ranges, or a piece."""
    units: Any
    words: tuple[tuple[int, int], ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    program: Any
    descriptor: bytes
    bits: Bits

    @cached_property
    def input_bits(self) -> list[int]:
        return self.bits.bits_of(self.inputs)

    @cached_property
    def output_bits(self) -> list[int]:
        return self.bits.bits_of(self.outputs)

    @cached_property
    def gate_ranges(self) -> list[tuple[int, int]]:
        return [(self.bits.base[lo], self.bits.base[hi]) for lo, hi in self.words]

    def to_c_gate(self, i: int) -> int:
        """Gate ``i`` of the block's descriptor (§6) as C's descriptor gate, the index that names a block, a boundary or a
        leaf: its input gates are in(S), then its gates are S's (`words`) in order."""
        return _to_c(i, self.inputs, self.words)

    def to_c_bit(self, i: int) -> int:
        """Gate ``i`` of the block on bits as a gate of C on bits, for the block's statement and counts only: its input bits
        are in(S)'s, then its gates are S's."""
        return _to_c(i, self.input_bits, self.gate_ranges)

    def listing(self) -> bytes:
        """The block on bits, gate by gate (module docstring), from C on bits: a test's reference for its layout."""
        bits, ins = self.bits, self.input_bits
        num = {b: i for i, b in enumerate(ins)}
        at = len(ins)
        for lo, hi in self.gate_ranges:
            for b in range(lo, hi):
                num[b] = at
                at += 1
        body = [ln for lo, hi in self.words for g in range(lo, hi) for ln in bits.lines(g)]
        return _text(bits, ins, body, self.output_bits, num)


def _to_c(i: int, ins: Sequence[int], ranges: Sequence[tuple[int, int]]) -> int:
    if i < 0:
        raise IndexError(i)
    if i < len(ins):
        return ins[i]
    i -= len(ins)
    for lo, hi in ranges:
        if i < hi - lo:
            return lo + i
        i -= hi - lo
    raise IndexError(i)


def _text(bits: Bits, ins: Sequence[int], body, outs: Sequence[int], num: Mapping[int, int]) -> bytes:
    lines = [FORMAT, f"{len(ins)} {len(body)} {len(outs)}"]
    for b in ins:
        k, v = bits.kind[bits.gate_of(b)]
        lines.append(v if k == "input" else WIRE)
    try:
        lines += [" ".join([kind, *(str(num[o]) for o in ops)]) for kind, ops in body]
    except KeyError as e:
        raise Refused("E_UNREAD", f"a gate reads bit {e.args[0]} of C, which no port carries") from None
    lines += [f"out {num[b]}" for b in outs]
    return ("\n".join(lines) + "\n").encode("ascii")


def listing_of(c: bytes) -> bytes:
    """A Program whose input nodes come first (a block's), laid out on bits by §6 as C is, gate by gate: its input bits, its
    gates, the bits it returns, in its own numbering (module docstring's `listing`)."""
    bits = Bits(program(c))
    ins = [b for g, (k, _) in enumerate(bits.kind) if k == "input" for b in bits.out[g]]
    if ins != list(range(len(ins))):
        raise Refused("E_FORMAT", "an input node after a computed gate")
    body = [ln for g, (k, _) in enumerate(bits.kind) if k != "input" for ln in bits.lines(g)]
    return _text(bits, ins, body, bits.bits_of(bits.C.root_outputs()), {b: b for b in range(bits.size)})


def _build(bits: Bits, where: _Where, ins: list[int], outs: list[int], row: Any = None) -> Any:
    D = bits.D
    scope = where.scope
    body = scope.spec.body
    in_runs = _runs(ins, lambda g: (bits.width[g], _binding(bits, g), row(g) if row else None))
    out_runs = _runs(outs, lambda g: bits.width[g])
    params = tuple((f"p{i}", Array(len(r), Value(bits.width[r[0]]))) for i, r in enumerate(in_runs))
    ret = Tuple(*[Array(len(r), Value(bits.width[r[0]])) for r in out_runs])
    kept = where.kept

    def build(B, S, *colls):
        ref: dict[int, tuple] = {}
        for run, coll in zip(in_runs, colls):
            for j, g in enumerate(run):
                ref[g] = coll.refs[j]

        def leaf(r) -> tuple:
            g = resolve(scope, r)
            if g not in ref:
                raise Refused("E_UNREAD", f"an argument leaf of the block is gate {g} of C, which no port or node carries")
            return ref[g]

        def args(pairs):
            return [Coll(t, _spell([leaf(rs[j]) for j in range(rs.count)])) for t, rs in pairs]

        for k, m0, m1 in kept:
            node = body.nodes[k]
            if node.form in ("call", "primitive") or (m0, m1) == (0, node.n):
                a = args((x.type, x.refs) for x in node.args)
                o = B.batch(node.fn, *a, axes=node.axes, n=node.n) if node.form == "batch" else B.call(node.fn, *a)
                for j in range(node.out.leaves):
                    ref.setdefault(resolve(scope, ("n", k, j)), o.refs[j])
                continue
            per = node.out.leaves // node.n
            for m in range(m0, m1):
                pairs = []
                for x, ax in zip(node.args, node.axes):
                    if ax is None:
                        pairs.append((x.type, x.refs))
                    elif ax == 0:
                        L = x.type.elem.leaves
                        pairs.append((x.type.elem, x.refs.slice(m * L, (m + 1) * L)))
                    else:
                        pairs.append((remove_axis(x.type, ax)[1], x.take(ax, m).refs))
                o = B.call(node.fn, *args(pairs))
                for j in range(per):
                    ref.setdefault(resolve(scope, ("n", k, m * per + j)), o.refs[j])
        return Coll(ret, concat([_spell([ref[g] for g in r]) for r in out_runs]))

    fn = CompositeDefinition("Block", 1, (), lambda S: (params, ret), build, register=False)
    registered = [n for (n, _), r in zip(params, in_runs) if _binding(bits, r[0]) == REGISTRATION]
    saved = list(P._BODY_LOOKUP)

    def lookup(p):
        try:
            return D.boolean[p.id]
        except KeyError:
            raise LookupError(f"C names no Boolean body for {p.id}") from None

    P.register_boolean_bodies(lookup)
    try:
        return P.Program(fn, registered=registered)
    finally:
        P._BODY_LOOKUP[:] = saved


def restrict(c: bytes, partition: Mapping[str, Any], units: Any, bits: Bits | None = None) -> Block:
    """`compact (restrict C S)` for S the gates of ``units``: ascending ranges of the partition's units, or under `Q_fit` a
    piece (`fit`) (module docstring: the index, the block and the object)."""
    bits = Bits(program(c)) if bits is None else bits
    tp = _partition(bits.D, partition)
    where = _where(bits.D, tp, units)
    ins, outs = ports(where.part)
    prog = _build(bits, where, ins, outs, _rows(bits.D, tp, partition, units))
    key = dict(units) if isinstance(units, Mapping) else tuple(tuple(u) for u in units)
    return Block(key, tuple(where.part.intervals()), tuple(ins), tuple(outs), prog, b"".join(descriptor_chunks(prog)), bits)


def evaluate(text: bytes, inputs: Sequence[int]) -> list[int]:
    """The output bits of the listing ``text`` on its input bits ``inputs``."""
    it = iter(text.decode("ascii").split("\n"))
    if next(it) != FORMAT:
        raise Refused("E_FORMAT", "not a block listing")
    I, K, O = map(int, next(it).split())
    if len(inputs) != I:
        raise Refused("E_FORMAT", f"{len(inputs)} input bits for {I}")
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


__all__ = ["Bits", "Block", "FORMAT", "REGISTRATION", "RUN", "Refused", "WIRE", "evaluate", "listing_of", "ports", "program", "restrict",
           "split", "unit_gates"]
