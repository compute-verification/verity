"""C_S compacted: Core's `Circuit.restrict` of C to a proof block's units, as a Program of the block's own (`restrict`).

**C's index, the one that names things.** C's gates are its descriptor's word-level gates, numbered by PROTOCOL.md §6's
layout of its root: bodies laid out in node order, recursively; a primitive node is one gate, a call node its callee's gates,
a batch's members (a scan's iterations) consecutive in member order, each its callee's gates, a batch of a primitive one gate
a member. An input node (`Input<w>_v1`, `RegisteredInput<w>_v1` for a registered value, `SuppliedInput<w>_v1` for a supplied
one) is a primitive node of the root,
one gate a member, at its node's place. A gate has its primitive's width. This index, and only it, names a block (`words`),
a boundary (in(S) and out(S): `inputs`, `outputs`) and a leaf; `Block.to_c_gate` maps the block's descriptor gates into it.
Units, and a partition's blocks, are counted in the population's own order (`TemplatePartition`), not in gates.

**C on bits, for a statement and a count only.** It never names a block, a boundary or a leaf. It numbers the bits of C's
gates, gate by gate in C's index order: gate g's bit gates come after every bit gate of
the gates before g (`Bits.base`), and are, in place:
- an input gate of w bits, of any binding: w one-bit input gates of that binding, bit j at offset `split(w)[j]`, which is
  j (least significant bit first);
- a gate of the Boolean profile (`And<n>_v1`, `Xor<n>_v1`, `Not_v1`, `Const1[0x0]_v1`, `Const1[0x1]_v1`): itself;
- any other primitive's gate: the gates of its Boolean body (the descriptor's `boolean` map, §4.9) in the body's own §6
  layout, the body's parameters taking no gate: its parameter bits are the bits of the gate's operands, in argument order
  and each operand's bits least significant first, and bit j of the gate's value is the body's j-th returned bit (one of its
  gates, or a parameter bit it passes through).
Every gate of C on bits reads only earlier gates.

**The block.** S is the gates of C inside the block's units, each unit the instance the partition's template query numbers
(`TemplatePartition.gates`). in(S) is the gates outside S that the units' parameter leaves name, out(S) the gates of S that C
returns or that a root node's argument outside S names, each once in increasing index of C (`front`), read from the
descriptor and never by a pass over C.

**The object** (`restrict`): the Program whose root
- has a parameter for each maximal run of in(S) of one width and one binding, in order, `p<i>` of type `Array<n, Value<w>>`,
  bound at registration when its gates are `RegisteredInput` gates of C, supplied when they are `SuppliedInput` gates (the
  kind only: the mark stays C's, under the input's name in C's description), and per run otherwise;
- then has the units' root nodes of C in C's order, each as C has it when the block holds all of its members, and one call
  node a member, in member order, when it holds some (a unit is one member); each argument leaf names the parameter leaf
  of its gate of in(S), or the earliest leaf of the block's nodes whose gate it is, the leaves spelled as the maximal runs
  of consecutive leaves of one node or parameter, each an `Affine`, concatenated;
- returns, for each maximal run of out(S) of one width, `Array<n, Value<w>>`, in a Tuple, each gate as the earliest leaf of
  the block's nodes whose gate it is;
and whose Definitions and Boolean map are C's (a word primitive's body is C's). Its bytes are the codec's
(`descriptor_chunks`). Laid out by §6 as C is, its gates are in(S), one input gate a leaf, then S's gates in order
(`Block.to_c_gate`). On bits, its gates are in(S)'s bits, each gate's least significant first, then S's bit gates in order
(`Block.to_c_bit`).
`listing` writes the block on bits gate by gate, a test's reference: `verity/block-circuit/v1`; `I K O`; per input bit its
binding (`run`, `registration`, `supplied`, or `wire` for a computed gate of C); per gate its kind and its operands; per output `out`
and its gate, all in the block's numbering, one ASCII line each.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import Any, Mapping, Sequence

from . import boolean as BL
from . import partition_object as PO
from . import ports as PT
from . import program as P
from .codec import DecodedProgram, decode_program, descriptor_chunks, lookup_primitive
from .defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
from .layout import Circuit, FunctionCall, resolve
from .program import REGISTRATION, RUN, SUPPLIED, input_id, input_prim
from .refs import Affine, Coll, Concat, Grid, Strided, concat
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
        except KeyError:
            name, _, version = fid.rpartition("_v")
            p = PrimitiveDefinition(name, int(version), tuple((n, type_from_json(t)) for n, t in d["params"]),
                                    type_from_json(d["ret"]), _unevaluated, register=False)
            if (pj := d.get("ports")) is not None:
                runs = {k: v for k, v in pj.items() if k != "transfer"}
                pt = PT.PortTypes.from_json(fid, runs, p.params, p.ret) if runs else PT.PortTypes(tuple(() for _ in p.params), ())
                p.port_types = replace(pt, transfer=_unevaluated) if pj.get("transfer") else pt
            reg.add(p)
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
    return fn.signature()[1].w


class Bits:
    """C on bits (module docstring), gate by gate: each gate of C's first bit (`base`), its count of bit gates, and the bit
    gates its value's bits are (`out`)."""

    def __init__(self, D: DecodedProgram) -> None:
        self.D, self.C = D, D.circuit
        self.word = list(self.C.gates())
        self.base: list[int] = []
        self.out: list[tuple[int, ...]] = []
        self.kind: list[tuple[str, Any]] = []
        self.width: list[int] = []
        at = 0
        for g in self.word:
            fn, w = g.prim, _width(g.prim)
            self.base.append(at)
            self.width.append(w)
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
        self.base.append(at)
        self.size = at

    def operand_bits(self, g: int) -> list[int]:
        return [b for o in self.word[g].operands for b in self.out[o]]

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

    def bits_of(self, gates: Sequence[int]) -> list[int]:
        """Each gate's bits, least significant first, gate by gate; refused where two are one bit gate."""
        out = [b for g in gates for b in self.out[g]]
        if len(set(out)) != len(out):
            raise Refused("E_PORT_TWICE", "two port bits are one gate of C on bits")
        return out


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


def front(D: DecodedProgram, words: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """in(S) and out(S) from the root's nodes, with no pass over C: in(S) the gates the units' parameter leaves name outside
    S (the door's `Spot.BlockFront.front`), out(S) the gates of S that C returns or that a root node outside S, or a member
    outside S of a batch S holds in part, takes as an argument, each once, in increasing index."""
    C = D.circuit

    def inside(g: int) -> bool:
        return any(lo <= g < hi for lo, hi in words)

    body, scope = D.root.body, C.root_scope
    kept = _nodes(D, words)
    held = {k: (m0, m1) for k, m0, m1 in kept}
    ins: set[int] = set()
    for k, m0, m1 in kept:
        for m in range(m0, m1):
            ins.update(g for g in FunctionCall(C, scope, k, m).args if not inside(g))

    leaves = {}
    for k, (m0, m1) in held.items():
        per = body.nodes[k].out.leaves // (body.nodes[k].n if body.nodes[k].form == "batch" else 1)
        leaves[k] = (m0 * per, m1 * per)

    reads = [body.ret]
    for k, node in enumerate(body.nodes):
        if k not in held:
            reads += [x.refs for x in node.args]
            continue
        m0, m1 = held[k]
        if node.form != "batch" or (m0, m1) == (0, node.n):
            continue
        for x, ax in zip(node.args, node.axes):
            if ax is None:
                reads.append(x.refs)
            elif ax == 0:
                L = x.type.elem.leaves
                reads += [x.refs.slice(0, m0 * L), x.refs.slice(m1 * L, node.n * L)]
            else:
                reads += [x.take(ax, m).refs for m in range(node.n) if not m0 <= m < m1]
    outs = {resolve(scope, r) for rs in reads for r in _into(rs, leaves)}
    return sorted(ins), sorted(outs)


def _into(refs: Any, leaves: Mapping[int, tuple[int, int]]) -> Any:
    """The references of ``refs`` to the leaves ``leaves`` names of root nodes, part by part, each distinct one once a
    part: a part that names no such node is not read leaf by leaf."""
    todo = [refs]
    while todo:
        p = todo.pop()
        if isinstance(p, Concat):
            todo.extend(p.parts)
        elif isinstance(p, Affine):
            if p.space == "n" and p.idx in leaves:
                lo, hi = leaves[p.idx]
                yield from (("n", p.idx, j) for j in range(max(lo, p.base), min(hi, p.base + p.count)))
        elif isinstance(p, Strided):
            if p.space == "n" and p.idx in leaves:
                lo, hi = leaves[p.idx]
                rows = 1 if p.outer_stride == 0 else -(-p.count // p.inner)
                for q in range(rows):
                    n = min(p.inner, p.count - q * p.inner)
                    for r in range(1 if p.inner_stride == 0 else n):
                        j = p.base + q * p.outer_stride + r * p.inner_stride
                        if lo <= j < hi:
                            yield ("n", p.idx, j)
        elif isinstance(p, Grid):
            if any(s == "n" and i in leaves for s, i, *_x in p.lanes):
                todo.extend(p.lane_parts())
        else:
            for r in p:
                if r[0] == "n" and r[1] in leaves and leaves[r[1]][0] <= r[2] < leaves[r[1]][1]:
                    yield r


class _Gates:
    """Each gate's width and binding, read from the root node that holds it: once a root input node, and
    down C's hierarchy for any other."""

    def __init__(self, D: DecodedProgram) -> None:
        self.D, self.body = D, D.root.body
        self.memo: dict[int, tuple[int, str]] = {}

    def of(self, g: int) -> tuple[int, str]:
        k = self.body.node_at(g)
        if k in self.memo:
            return self.memo[k]
        fn = self.body.nodes[k].fn
        if isinstance(fn, PrimitiveDefinition) and (inp := input_id(fn.id)) is not None:
            self.memo[k] = (_width(fn), inp[0])
            return self.memo[k]
        return _width(self.D.circuit.gate(g).prim), RUN

    def width(self, g: int) -> int:
        return self.of(g)[0]

    def binding(self, g: int) -> str:
        return self.of(g)[1]


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


def _nodes(D: DecodedProgram, words: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """The root nodes of C the gates ``words`` are, each ``(k, first member, end member)``."""
    body, out = D.root.body, []
    for lo, hi in words:
        for k, node in enumerate(body.nodes):
            a, b = body.offsets[k], body.offsets[k + 1]
            if b <= lo or a >= hi:
                continue
            if node.form not in ("call", "batch"):
                raise Refused("E_FORM", f"root node {k} is a {node.form}: a block holds calls and batches")
            per = node.fn.gates
            if per == 0:
                raise Refused("E_NO_GATE", f"root node {k} calls {node.fn.id}, which has no gate: no unit holds it")
            m0, m1 = (max(lo, a) - a) // per, (min(hi, b) - a) // per
            if a + m0 * per != max(lo, a) or a + m1 * per != min(hi, b):
                raise Refused("E_UNITS", f"the units cut root node {k} inside a member")
            out.append((k, m0, m1))
    return out


@dataclass(frozen=True)
class Block:
    """`restrict`'s value: the block's Program and its bytes, S as C's gate ranges (`words`), in(S) and out(S) (`inputs`,
    `outputs`), and the bits of C on bits behind them (`bits`)."""
    units: tuple[tuple[int, int], ...]
    words: tuple[tuple[int, int], ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    program: Any
    descriptor: bytes
    D: DecodedProgram
    given: Bits | None = field(default=None, repr=False, compare=False)

    @cached_property
    def bits(self) -> Bits:
        """C on bits, built here on first read (a pass over C): the block's statement and counts read it, its Program never."""
        return self.given if self.given is not None else Bits(self.D)

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


def _build(D: DecodedProgram, words: Sequence[tuple[int, int]], ins: list[int], outs: list[int]) -> Any:
    C, gates = D.circuit, _Gates(D)
    body, scope = D.root.body, C.root_scope
    in_runs = _runs(ins, gates.of)
    out_runs = _runs(outs, gates.width)
    params = tuple((f"p{i}", Array(len(r), Value(gates.width(r[0])))) for i, r in enumerate(in_runs))
    ret = Tuple(*[Array(len(r), Value(gates.width(r[0]))) for r in out_runs])
    kept = _nodes(D, words)

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
            if node.form == "call" or (m0, m1) == (0, node.n):
                a = args((x.type, x.refs) for x in node.args)
                o = B.call(node.fn, *a) if node.form == "call" else B.batch(node.fn, *a, axes=node.axes, n=node.n)
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
    registered = [n for (n, _), r in zip(params, in_runs) if gates.binding(r[0]) == REGISTRATION]
    supplied = [n for (n, _), r in zip(params, in_runs) if gates.binding(r[0]) == SUPPLIED]
    saved = list(P._BODY_LOOKUP)

    def lookup(p):
        try:
            return D.boolean[p.id]
        except KeyError:
            raise LookupError(f"C names no Boolean body for {p.id}") from None

    P.register_boolean_bodies(lookup)
    try:
        return P.Program(fn, registered=registered, supplied=supplied)
    finally:
        P._BODY_LOOKUP[:] = saved


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]], bits: Bits | None = None,
             D: DecodedProgram | None = None) -> Block:
    """`compact (restrict C S)` for S the gates of ``units`` (module docstring: the index, the block and the object);
    ``D``: C decoded, when the caller holds it."""
    D = D if D is not None else program(c) if bits is None else bits.D
    words = unit_gates(D, partition, units)
    ins, outs = front(D, words)
    prog = _build(D, words, ins, outs)
    return Block(tuple(tuple(u) for u in units), tuple(words), tuple(ins), tuple(outs), prog,
                 b"".join(descriptor_chunks(prog)), D, bits)


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


__all__ = ["Bits", "Block", "FORMAT", "REGISTRATION", "RUN", "Refused", "SUPPLIED", "WIRE", "evaluate", "listing_of", "program", "restrict",
           "split", "unit_gates"]
