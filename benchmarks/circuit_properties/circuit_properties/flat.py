"""The definitions gate by gate, for circuits small enough to enumerate: every counted product as its root gates, and the
properties computed from them directly. `measure` must agree with these wherever both run.

Each counted activation is found by walking the circuit's activations (`layout.Scope`); its products' gates are the
activation's offset plus `ops.OpShape`'s members, and their activation operands are the root gates bound to x[k]
(`layout.param_leaf`). Fan-in reads the root circuit's wiring (`layout.Circuit.gate`): a product reaches an entry's
output when a gate of m(p) is in that output's cone inside the entry's activation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import Circuit, Scope, param_leaf, resolve

from .measure import Measurer
from .ops import Designation, is_constant, op_shape, param_bases


@dataclass(frozen=True)
class Product:
    op: str
    gates: tuple[int, ...]
    rep: int
    act: tuple[int, ...]


@dataclass(frozen=True)
class EntryActivation:
    id: str
    lo: int
    hi: int
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class Explicit:
    circuit: Circuit
    products: tuple[Product, ...]
    entries: tuple[EntryActivation, ...]


def explicit(program, designation: Designation) -> Explicit:
    """Every counted product and entry activation of ``program``, as root gates."""
    M = Measurer(program, designation)
    C = Circuit(program.root, explicit_limit=10 ** 8)
    products: list[Product] = []
    entries: list[EntryActivation] = []

    def visit(scope: Scope, ctx: tuple) -> None:
        F = scope.spec
        if M.match(designation.entries_rx, F):
            entries.append(EntryActivation(F.id, scope.offset, scope.offset + F.gates,
                                           tuple(resolve(scope, r) for r in F.body.ret)))
        for k, node in enumerate(F.body.nodes):
            f = node.fn
            site, matched = M.site_of(f, ctx)
            n = node.n if node.form in ("batch", "scan") else 1
            if site is not None:
                shape = op_shape(f, site)
                for m in range(n):
                    if isinstance(f, PrimitiveDefinition):
                        g = scope.offset + F.body.offsets[k] + m
                        ops = C.gate(g).operands
                        xb = param_bases(f)[shape.xi]
                        products.append(Product(f.id, (g,), g, tuple(ops[xb + j] for j in shape.x_leaves[0])))
                        continue
                    child = scope.child(k, m)
                    for lane, gates in enumerate(shape.members):
                        products.append(Product(f.id, tuple(child.offset + g for g in gates), child.offset + shape.rep[lane],
                                                tuple(param_leaf(child, shape.xi, j) for j in shape.x_leaves[lane])))
            elif not matched and not isinstance(f, PrimitiveDefinition):
                for m in range(n):
                    visit(scope.child(k, m), M.enter(f, ctx))

    visit(C.root_scope, M.ctx0)
    return Explicit(C, tuple(products), tuple(entries))


@dataclass(frozen=True)
class Properties:
    products: int
    s_x: int | None
    a_out: int
    a_out_values: int
    activation_values: int
    constant_leaves: dict
    a_in: int | None
    entries: dict


def properties(program, designation: Designation, ex: Explicit | None = None) -> Properties:
    """|𝒥|, s_×, a_out and a_in straight from their definitions."""
    ex = ex or explicit(program, designation)
    C = ex.circuit
    const = {g: C.gate(g).prim.id for p in ex.products for g in p.act if is_constant(C.gate(g).prim)}
    uses: Counter = Counter()
    cuses: Counter = Counter()
    for p in ex.products:
        for v in p.act:
            if v in const:
                cuses[const[v]] += 1
        for v in set(p.act) - const.keys():
            uses[v] += 1
    a_out = max(uses.values(), default=0)
    owner: dict[int, int] = {}
    for i, p in enumerate(ex.products):
        for g in p.gates:
            owner[g] = i
    per_entry: dict[str, list[int]] = {}
    for e in ex.entries:
        reach: dict[int, int] = {}
        inside: set[int] = set()
        for g in range(e.lo, e.hi):
            r = 0
            for o in C.gate(g).operands:
                if e.lo <= o < e.hi:
                    r |= reach.get(o, 0)
            i = owner.get(g)
            if i is not None:
                r |= 1 << i
                inside.add(i)
            if r:
                reach[g] = r
        for o in e.outputs:
            c = reach.get(o, 0).bit_count() if e.lo <= o < e.hi else 0
            per_entry.setdefault(e.id, []).append(c)
    hit = [c for cs in per_entry.values() for c in cs if c]
    return Properties(len(ex.products), min((len(p.gates) for p in ex.products), default=None), a_out,
                      sum(1 for c in uses.values() if c == a_out) if a_out else 0, len(uses), dict(sorted(cuses.items())),
                      min(hit) if hit else None, per_entry)


def whole(program, designation: Designation, owner, ex: Explicit | None = None) -> tuple[Counter, int]:
    """(products whole in each block, products split or outside every block), ``owner[g]`` being gate g's block."""
    ex = ex or explicit(program, designation)
    per: Counter = Counter()
    split = 0
    for p in ex.products:
        bs = {owner[g] for g in p.gates}
        b = bs.pop() if len(bs) == 1 else None
        if b is None:
            split += 1
        else:
            per[b] += 1
    return per, split
