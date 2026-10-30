"""The counted operations of a circuit, and each product's subcircuit inside one operation's Definition.

A designation names the counted operations of an arbitrary circuit in the `verity-ir/descriptor/v1` format
(`verity/ml/circuits/PROTOCOL.md`): a `Site` is a Definition (a regex over its id) whose activations compute counted
products, with the parameter holding the activation operands (`x`) and the one holding the weight operands (`w`). With
``lanes`` the two parameters are arrays of one length L, and lane k of an activation computes the product of x[k] and
w[k]; without, an activation is one product of the whole of both. ``within`` counts an activation only under an
ancestor Definition it matches (a GEMM's dot steps, not attention's).

The subcircuit of product k of one activation, m(p_k), is positional: the activation's gates that read some operand
of the op, read only lane k's activation and weight operands (nothing of another lane or of any other parameter), and
are not constants. `OpShape` computes it once per op Definition by flattening its body, and refuses an op whose lane
has no gate reading both of its operands (`NotSeparable`: a word primitive computing 16 products in one gate has no
per-product gates). These sets are disjoint, nonempty and computed, as `Nci.Cover.MulBlocks` requires of its products'
gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from verity.ml.circuits import cut
from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.program import is_input
from verity.ml.circuits.types import Array


@dataclass(frozen=True)
class Site:
    """Activations of a Definition matching ``op`` compute counted products of parameter ``x`` (activations) and ``w``
    (weights), lane by lane, counted only under an ancestor matching ``within`` when it is given."""

    op: str
    x: str
    w: str
    lanes: bool = True
    within: str | None = None

    @cached_property
    def op_rx(self) -> re.Pattern:
        return re.compile(self.op)

    @cached_property
    def within_rx(self) -> re.Pattern | None:
        return None if self.within is None else re.compile(self.within)


@dataclass(frozen=True)
class Designation:
    """The counted operations (``sites``) and the Definitions whose returned leaves are the outputs fan-in is measured
    at (``entries``, a regex; none: no fan-in)."""

    sites: tuple[Site, ...]
    entries: str | None = None

    @cached_property
    def entries_rx(self) -> re.Pattern | None:
        return None if self.entries is None else re.compile(self.entries)


class NotSeparable(ValueError):
    """A lane of a counted operation has no gate of its own reading both of its operands."""


def ids(fn) -> tuple[str, ...]:
    """The ids a regex matches against: a lowered root's are its program function's (`Root[...]_v1` is no name), and a
    decoded root's its `program_fn_id` (a string, `Name_vN{...}`)."""
    if isinstance(fn, str):
        return (fn.split("{", 1)[0], fn)
    fn = getattr(fn, "program_fn", fn)
    if isinstance(fn, SpecializedDefinition):
        return (fn.base_id, fn.id)
    return (fn.id,)


def matches(rx: re.Pattern | None, fn) -> bool:
    return rx is not None and any(rx.fullmatch(i) for i in ids(fn))


def is_constant(fn) -> bool:
    """A constant gate: a primitive with no parameters that is not an input (`Const1[0x0]_v1`)."""
    return isinstance(fn, PrimitiveDefinition) and not fn.params and not is_input(fn)


def param_bases(fn) -> list[int]:
    """Flat index of each parameter's first leaf, in parameter order, and the total last."""
    out = [0]
    for _n, t in fn.signature()[0]:
        out.append(out[-1] + t.leaves)
    return out


def member_refs(node, m: int, i: int):
    """The references member ``m`` of a batch or scan node passes as its parameter ``i``."""
    a = node.args[i]
    if node.form == "batch":
        return cut.member_refs(a, node.axes[i], m)
    if node.form == "scan" and 1 <= i <= node.n_xs:
        L = a.type.elem.leaves
        return a.refs.slice(m * L, (m + 1) * L)
    return a.refs


@dataclass
class Flat:
    """A Definition's activation as gates: gate g's operands are local gates (>= 0) or flat parameter leaves (-1 - j)."""

    prims: list
    operands: list[tuple[int, ...]]
    ret: list[int]


_FLAT: dict[int, tuple[object, Flat]] = {}


def flatten(fn) -> Flat:
    """``fn``'s activation gate by gate, in canonical layout order (PROTOCOL.md §6)."""
    hit = _FLAT.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1]
    n = param_bases(fn)[-1]
    if isinstance(fn, PrimitiveDefinition):
        flat = Flat([fn], [tuple(-1 - j for j in range(n))], [0])
    else:
        flat = Flat([], [], [])
        base = param_bases(fn)
        outs: list[list[int]] = []

        def src(r) -> int:
            s, i, j = r
            return -1 - (base[i] + j) if s == "p" else outs[i][j]

        def inline(f, args: list[int]) -> list[int]:
            if isinstance(f, PrimitiveDefinition):
                flat.prims.append(f)
                flat.operands.append(tuple(args))
                return [len(flat.operands) - 1]
            sub, off = flatten(f), len(flat.operands)
            flat.prims.extend(sub.prims)
            flat.operands.extend(tuple(off + o if o >= 0 else args[-1 - o] for o in ops) for ops in sub.operands)
            return [off + s if s >= 0 else args[-1 - s] for s in sub.ret]

        for node in fn.body.nodes:
            if node.form in ("primitive", "call"):
                outs.append(inline(node.fn, [src(r) for a in node.args for r in a.refs]))
            elif node.form == "batch":
                out: list[int] = []
                for m in range(node.n):
                    out += inline(node.fn, [src(r) for i in range(len(node.args)) for r in member_refs(node, m, i)])
                outs.append(out)
            else:
                carry, ys = [src(r) for r in node.args[0].refs], []
                for m in range(node.n):
                    r = inline(node.fn, carry + [src(x) for i in range(1, len(node.args)) for x in member_refs(node, m, i)])
                    carry, ys = r[:len(carry)], ys + r[len(carry):]
                outs.append(carry + ys)
        flat.ret = [src(r) for r in fn.body.ret]
    _FLAT[id(fn)] = (fn, flat)
    return flat


def _bits(mask: int) -> list[int]:
    s = bin(mask)[:1:-1]
    return [i for i, c in enumerate(s) if c == "1"]


class OpShape:
    """One counted operation's Definition under a site: its lanes, and per product k the local gates m(p_k)
    (`members`), the representative gate (`rep`, the last member reading both operands), the activation's leaves
    (`x_leaves`, positions in parameter ``x``), the returned leaves m(p_k) reaches (`reach`, a bitmask) and per returned
    leaf the flat parameter leaves it reads (`ret_need`)."""

    def __init__(self, fn, site: Site) -> None:
        self.fn, self.site = fn, site
        params = [n for n, _t in fn.signature()[0]]
        types = [t for _n, t in fn.signature()[0]]
        for name in (site.x, site.w):
            if name not in params:
                raise ValueError(f"{fn.id} has no parameter {name!r} (it has {params})")
        self.xi, self.wi = params.index(site.x), params.index(site.w)
        tx, tw = types[self.xi], types[self.wi]
        if site.lanes:
            if not (isinstance(tx, Array) and isinstance(tw, Array) and tx.n == tw.n):
                raise ValueError(f"{fn.id}: lanes need x and w arrays of one length, got {tx!r} and {tw!r}")
            L, lx, lw = tx.n, tx.elem.leaves, tw.elem.leaves
        else:
            L, lx, lw = 1, tx.leaves, tw.leaves
        self.lanes = L
        base = param_bases(fn)
        self.n_params = base[-1]

        def slot(j: int) -> int:
            if base[self.xi] <= j < base[self.xi + 1]:
                return 1 << ((j - base[self.xi]) // lx)
            if base[self.wi] <= j < base[self.wi + 1]:
                return 1 << (L + (j - base[self.wi]) // lw)
            return 1 << (2 * L)

        flat = flatten(fn)
        self.gates = len(flat.operands)
        dep: list[int] = []
        for ops in flat.operands:
            d = 0
            for o in ops:
                d |= slot(-1 - o) if o < 0 else dep[o]
            dep.append(d)
        members: list[list[int]] = [[] for _ in range(L)]
        rep: list[int | None] = [None] * L
        for g, d in enumerate(dep):
            if not d or d >> (2 * L):
                continue
            lane_bits = (d | (d >> L)) & ((1 << L) - 1)
            if lane_bits & (lane_bits - 1):
                continue
            k = lane_bits.bit_length() - 1
            members[k].append(g)
            if d == (1 << k) | (1 << (L + k)):
                rep[k] = g
        for k in range(L):
            if rep[k] is None:
                raise NotSeparable(f"{fn.id}: lane {k} has no gate of its own reading both x[{k}] and w[{k}] "
                                   f"(the op computes its products jointly, so no product has gates of its own)")
        self.members = [tuple(m) for m in members]
        self.rep = rep
        self.sizes = [len(m) for m in members]
        self.x_leaves = [tuple(range(k * lx, (k + 1) * lx)) for k in range(L)]
        R = len(flat.ret)
        rr = [0] * self.gates
        for ell, s in enumerate(flat.ret):
            if s >= 0:
                rr[s] |= 1 << ell
        for g in range(self.gates - 1, -1, -1):
            if rr[g]:
                for o in flat.operands[g]:
                    if o >= 0:
                        rr[o] |= rr[g]
        self.reach = []
        for m in self.members:
            r = 0
            for g in m:
                r |= rr[g]
            self.reach.append(r)
        pneed: list[int] = []
        for ops in flat.operands:
            d = 0
            for o in ops:
                d |= (1 << (-1 - o)) if o < 0 else pneed[o]
            pneed.append(d)
        self.ret_need = [pneed[s] if s >= 0 else 1 << (-1 - s) for s in flat.ret]
        self.ret_leaves = R

    def need(self, mask: int) -> int:
        """The flat parameter leaves the returned leaves in ``mask`` read."""
        out = 0
        for ell in _bits(mask):
            out |= self.ret_need[ell]
        return out

    def reaching(self, mask: int) -> int:
        """How many of the activation's products reach a returned leaf in ``mask``."""
        return sum(1 for r in self.reach if r & mask)


_SHAPES: dict[tuple[int, int], tuple[object, OpShape]] = {}


def op_shape(fn, site: Site) -> OpShape:
    key = (id(fn), id(site))
    hit = _SHAPES.get(key)
    if hit is None or hit[0] is not fn:
        hit = (fn, OpShape(fn, site))
        _SHAPES[key] = hit
    return hit[1]
