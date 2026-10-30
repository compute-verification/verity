"""Canonical layout, root-scoped value references, and the three gate-selection classes (IR §4.4, §6, §D).

Layout: a function body is laid out in body order; each node is one interval; `batch` members and `scan`
iterations are consecutive in structural order. Gate counts and offsets are *intrinsic summaries* cached per
specialization (`SpecializedDefinition.intrinsic()`), so gate `i` is located by bisection and division along the call
hierarchy — never by expansion.

Canonical reference of a produced value = its producing gate's root-scoped index. Prescribed inputs are
`Input[k] : () -> Value<w_k>` gates (IR §D.3) and therefore have their own canonical references.

Three classes (IR §D.1/§D.2), not synonyms:

* `GateSelection` — any selected set of gates of the root circuit + the wiring among them + its induced
  boundary. `S.size`, `S[i]` (i-th selected gate in root canonical order), `S.in_` (the boundary gates
  entering from outside: distinct gates outside S with a wire into S), `S.out` (the boundary gates leaving:
  distinct gates inside S with a wire to a gate outside S, or designated root outputs). Boundaries are sets of
  *gates* — a producer counts once regardless of fan-out; there is no `distinct()` operation. Canonical
  boundary order: ascending root-scoped producing-gate index. These explicit computations are the oracle for
  the compact per-part analysis in `verity.ml.circuits.boundary` (`in_gates` / `out_gates` / `w_out`).
* `Region` — the contiguous special case `C[i:j]`, efficiently addressable.
* `FunctionCall` — a hierarchy-defined gate selection (always a region in this layout) **plus** the declared interface
  `call.args` / `call.returns`, which keep signature order and repeated references. `F(v, v)` returning
  `(t, a, t)` has `args = [v, v]`, `returns = [t, v, t]`, `in_ = [v]`, `out = [t]` (if only `t` escapes).

Intrinsic vs contextual summaries: gate count, interface types and offsets are intrinsic to a specialized
definition and cached on it. Anything depending on consumers, enclosing wiring, protected outputs or a plan
(`S.out` of an arbitrary gate selection, reachability, cuts) is contextual and is computed per instance, never
cached by definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from .defs import Node, PrimitiveDefinition, SpecializedDefinition
from .refs import Ref


class Scope:
    """One activation of a composite in the canonical circuit: which specialization, at which absolute
    gate offset, reached from which node/member of the parent."""

    __slots__ = ("spec", "offset", "parent", "node_idx", "member")

    def __init__(self, spec: SpecializedDefinition, offset: int, parent: "Scope | None", node_idx: int, member: int) -> None:
        self.spec, self.offset, self.parent, self.node_idx, self.member = spec, offset, parent, node_idx, member

    def child(self, node_idx: int, member: int = 0) -> "Scope":
        node = self.spec.body.nodes[node_idx]
        off = self.offset + self.spec.body.offsets[node_idx] + member * node.fn.gates
        return Scope(node.fn, off, self, node_idx, member)

    def path(self) -> list[str]:
        out: list[str] = []
        s: Scope | None = self
        while s is not None:
            if s.parent is not None and s.parent.spec.body.nodes[s.node_idx].form != "call":
                out.append(f"{s.spec.id}#{s.member}")
            else:
                out.append(s.spec.id)
            s = s.parent
        return out[::-1]


def resolve(scope: Scope, ref: Ref) -> int:
    """Root-scoped gate index of the value denoted by body-scoped `ref` in `scope`."""
    return resolve_gate(scope, ref)[0]


def param_leaf(scope: Scope, pi: int, leaf: int) -> int:
    """Gate index bound to leaf `leaf` of parameter `pi` of the activation `scope`."""
    return resolve_gate(scope, ("p", pi, leaf))[0]


def resolve_gate(scope: Scope, ref: Ref) -> tuple[int, PrimitiveDefinition]:
    """Root-scoped gate index and producing primitive of the value denoted by body-scoped `ref` in `scope`: one walk
    up through parameters (to the argument the parent passed) and down through returns (to the producing node)."""
    space, idx, leaf = ref
    while True:
        if space == "p":
            parent = scope.parent
            if parent is None:
                raise ValueError("the root has no parameters (prescribed inputs are Input gates)")
            node = parent.spec.body.nodes[scope.node_idx]
            if node.form == "call":
                space, idx, leaf = node.args[idx].refs[leaf]
            elif node.form == "batch":
                ax, a = node.axes[idx], node.args[idx]
                if ax is None:
                    space, idx, leaf = a.refs[leaf]
                elif ax == 0:
                    space, idx, leaf = a.refs[scope.member * a.type.elem.leaves + leaf]
                else:
                    space, idx, leaf = a.take(ax, scope.member).refs[leaf]
            else:  # scan: params are (carry, xs..., shared...)
                if idx == 0:
                    if scope.member == 0:
                        space, idx, leaf = node.args[0].refs[leaf]
                    else:  # the previous iteration's returned carry
                        scope = Scope(scope.spec, scope.offset - scope.spec.gates, parent, scope.node_idx, scope.member - 1)
                        space, idx, leaf = scope.spec.body.ret[leaf]
                        continue
                elif idx <= node.n_xs:
                    a = node.args[idx]
                    space, idx, leaf = a.refs[scope.member * a.type.elem.leaves + leaf]
                else:
                    space, idx, leaf = node.args[idx].refs[leaf]
            scope = parent
            continue
        body = scope.spec.body
        node = body.nodes[idx]
        node_off = scope.offset + body.offsets[idx]
        fn = node.fn
        if node.form == "primitive":
            return node_off, fn
        if node.form == "call":
            scope = Scope(fn, node_off, scope, idx, 0)
            space, idx, leaf = fn.body.ret[leaf]
            continue
        if node.form == "batch":
            m, j = divmod(leaf, fn.out_leaves)
            if isinstance(fn, PrimitiveDefinition):
                return node_off + m, fn
            scope = Scope(fn, node_off + m * fn.gates, scope, idx, m)
            space, idx, leaf = fn.body.ret[j]
            continue
        # scan: output is (carry_n, Array<n, Y>)
        if node.n == 0:
            raise ValueError("a scan with no iterations has no output to read")
        c = node.out.fields[0].leaves
        if leaf < c:
            m, j = node.n - 1, leaf
        else:
            y = node.out.fields[1].elem.leaves
            m, r = divmod(leaf - c, y)
            j = c + r
        scope = Scope(fn, node_off + m * fn.gates, scope, idx, m)
        space, idx, leaf = fn.body.ret[j]


class LazyOperands(Sequence):
    """Operand references of a wide-fan-in gate (e.g. an embedding column of 151,936 weight values), resolved
    on demand: `len`, `ops[j]`, iteration. Semantically identical to the tuple `_prim_member_args` returns."""

    __slots__ = ("scope", "node", "m", "_segments", "_n")

    def __init__(self, scope: Scope, node: Node, m: int) -> None:
        self.scope, self.node, self.m = scope, node, m
        segs = []  # (start, count, refs-getter)
        start = 0
        axes = node.axes if node.form == "batch" else (None,) * len(node.args)
        for a, ax in zip(node.args, axes):
            if ax is None:
                refs, n = a.refs, a.refs.count
            elif ax == 0:
                L = a.type.elem.leaves
                refs, n = a.refs.slice(m * L, (m + 1) * L), L
            else:
                refs = a.take(ax, m).refs
                n = refs.count
            segs.append((start, n, refs))
            start += n
        self._segments, self._n = segs, start

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, j):
        if isinstance(j, slice):
            return tuple(self[k] for k in range(*j.indices(self._n)))
        if j < 0:
            j += self._n
        if not (0 <= j < self._n):
            raise IndexError(j)
        for start, n, refs in self._segments:
            if j < start + n:
                return resolve(self.scope, refs[j - start])
        raise IndexError(j)

    def __iter__(self):
        for start, n, refs in self._segments:
            for k in range(n):
                yield resolve(self.scope, refs[k])

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))


WIDE_FAN_IN = 4096


def _prim_member_args(scope: Scope, node: Node, m: int):
    """Operand gate indices of member `m` of a primitive node (`prim`, or a `batch` of a primitive)."""
    if sum(p.leaves for _, p in node.fn.params) > WIDE_FAN_IN:
        return LazyOperands(scope, node, m)
    if node.form == "primitive":
        return tuple(resolve(scope, r) for a in node.args for r in a.refs)
    out: list[int] = []
    for a, ax in zip(node.args, node.axes):
        if ax is None:
            out.extend(resolve(scope, r) for r in a.refs)
        elif ax == 0:
            L = a.type.elem.leaves
            out.extend(resolve(scope, a.refs[m * L + j]) for j in range(L))
        else:
            out.extend(resolve(scope, r) for r in a.take(ax, m).refs)
    return tuple(out)


@dataclass(frozen=True)
class Gate:
    """`C[i]`: the primitive at gate `i`, its operand references (gate indices, ordered, repeats kept), and
    the activation that produced it."""

    index: int
    prim: PrimitiveDefinition
    operands: Sequence[int]
    scope: Scope

    @property
    def path(self) -> list[str]:
        return self.scope.path()

    def __repr__(self) -> str:
        return f"C[{self.index}] = {self.prim.id}({', '.join(map(str, self.operands))})"


# ---------------------------------------------------------------------------------------------------------
# GateSelection / Region / FunctionCall
# ---------------------------------------------------------------------------------------------------------


class GateSelection:
    """A selected set of gates of the root circuit + wiring among them + induced boundary (IR §D.1).
    Constructed explicitly from a gate set; intended for small/flattened circuits (cones, hand-picked sets).
    `in_`/`out` are *contextual*: computed on demand, never cached by definition."""

    def __init__(self, circuit: "Circuit", gates: Sequence[int]) -> None:
        self.circuit = circuit
        self._gates = tuple(sorted(set(int(g) for g in gates)))
        for g in self._gates[:1] + self._gates[-1:]:
            if not (0 <= g < circuit.size):
                raise IndexError(g)
        self._set = frozenset(self._gates)

    @property
    def size(self) -> int:
        return len(self._gates)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, g: int) -> bool:
        return g in self._set

    def gate_indices(self) -> tuple[int, ...]:
        return self._gates

    def __getitem__(self, i: int) -> Gate:
        return self.circuit.gate(self._gates[i])

    def __iter__(self) -> Iterator[Gate]:
        for g in self._gates:
            yield self.circuit.gate(g)

    @property
    def in_(self) -> tuple[int, ...]:
        """Boundary gates entering from outside — distinct gates outside S with a wire into S — ascending by
        gate index (`in(S)` of `verity/ml/circuits/parts.py`)."""
        self.circuit._check_explicit(self.size)
        s: set[int] = set()
        for g in self:
            s.update(o for o in g.operands if o not in self._set)
        return tuple(sorted(s))

    @property
    def out(self) -> tuple[int, ...]:
        """Boundary gates leaving — distinct gates inside S with a wire to a gate outside S, or designated root
        outputs — ascending by gate index (`out(S)` of `verity/ml/circuits/parts.py`). Requires enumerating consumers:
        explicit over the whole circuit; the compact equivalent is `boundary.out_gates`."""
        c = self.circuit
        c._check_explicit(c.size)
        used: set[int] = set()
        for g in range(c.size):
            if g in self._set:
                continue
            for o in c.gate(g).operands:
                if o in self._set:
                    used.add(o)
        used |= {o for o in c.root_outputs() if o in self._set}
        return tuple(sorted(used))

    def __repr__(self) -> str:
        return f"GateSelection({self.size} gates)"

    def __eq__(self, other) -> bool:
        return isinstance(other, GateSelection) and self.circuit is other.circuit and self._gates == other._gates


class Region(GateSelection):
    """`C[i:j]`: the contiguous special case; addressable without materialising the gate set."""

    def __init__(self, circuit: "Circuit", lo: int, hi: int) -> None:
        if not (0 <= lo <= hi <= circuit.size):
            raise IndexError(f"region [{lo}:{hi}] outside circuit of size {circuit.size}")
        self.circuit, self.lo, self.hi = circuit, lo, hi

    @property
    def size(self) -> int:
        return self.hi - self.lo

    def __contains__(self, g: int) -> bool:
        return self.lo <= g < self.hi

    def gate_indices(self) -> tuple[int, ...]:
        return tuple(range(self.lo, self.hi))

    def __getitem__(self, i):
        if isinstance(i, slice):
            lo, hi, step = i.indices(self.size)
            if step != 1:
                raise ValueError("unit stride only")
            return Region(self.circuit, self.lo + lo, self.lo + hi)
        if i < 0:
            i += self.size
        if not (0 <= i < self.size):
            raise IndexError(i)
        return self.circuit.gate(self.lo + i)

    def __iter__(self) -> Iterator[Gate]:
        for g in range(self.lo, self.hi):
            yield self.circuit.gate(g)

    @property
    def interval(self) -> tuple[int, int]:
        return (self.lo, self.hi)

    def union(self, other: "Region") -> "Region":
        return Region(self.circuit, min(self.lo, other.lo), max(self.hi, other.hi))

    @property
    def in_(self) -> tuple[int, ...]:
        self.circuit._check_explicit(self.size)
        s: set[int] = set()
        lo, hi = self.lo, self.hi
        for g in self:
            s.update(o for o in g.operands if not (lo <= o < hi))
        return tuple(sorted(s))

    @property
    def out(self) -> tuple[int, ...]:
        c = self.circuit
        c._check_explicit(c.size)
        used: set[int] = set()
        lo, hi = self.lo, self.hi
        for g in range(c.size):
            if lo <= g < hi:
                continue
            for o in c.gate(g).operands:
                if lo <= o < hi:
                    used.add(o)
        used |= {o for o in c.root_outputs() if lo <= o < hi}
        return tuple(sorted(used))

    def __repr__(self) -> str:
        return f"[{self.lo}:{self.hi}]"

    def __eq__(self, other) -> bool:
        return isinstance(other, Region) and self.circuit is other.circuit and (self.lo, self.hi) == (other.lo, other.hi)

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))


class FunctionCall(Region):
    """A function-call instance: a hierarchy-defined region plus the declared interface (IR §D.2).

    `args` / `returns`: gate indices in signature order, repeats kept (leaf order of the declared types).
    `in_` / `out`: boundary gates (IR §D.2) — distinct, ascending by gate index. `out` is the set of gates
    produced inside the call with a wire to a gate outside it or designated as root outputs. A gate inside
    can only leave through the returns, so `out ⊆ distinct_returns`; a declared return nobody reads (a dead
    gate) is in `returns` and `distinct_returns` but **not** in `out`.
    `distinct_returns` is the cheap declared-interface adapter — a conservative superset of `out` computed
    without scanning consumers; it is named for what it is and never stands in for `out`.

    The interface is not the boundary in either direction. `in_` here is derived from `args` (a gate
    referenced by two arguments enters once), so it is the declared-interface adapter on the input side: an
    argument the callee never reads (a dead parameter, e.g. the state `AttnBlock_v3{FIRST=True}` ignores) is
    in `args` and in this `in_` but is **not** a wire into the call. The exact gate boundary `in(S)` is
    `boundary.in_gates`, which filters by the callee's used parameter leaves; `in_gates ⊆ in_`."""

    def __init__(self, circuit: "Circuit", parent: Scope, node_idx: int, member: int = 0) -> None:
        self.parent, self.node_idx, self.member = parent, node_idx, member
        node = parent.spec.body.nodes[node_idx]
        lo = parent.offset + parent.spec.body.offsets[node_idx] + member * node.fn.gates
        Region.__init__(self, circuit, lo, lo + node.fn.gates)

    @property
    def node(self) -> Node:
        return self.parent.spec.body.nodes[self.node_idx]

    @property
    def fn(self):
        return self.node.fn

    @property
    def is_primitive(self) -> bool:
        return isinstance(self.node.fn, PrimitiveDefinition)

    @property
    def name(self) -> str:
        return self.fn.name

    @property
    def id(self) -> str:
        return self.fn.id

    def scope(self) -> Scope:
        if self.is_primitive:
            raise TypeError("primitive calls have no activation scope")
        return self.parent.child(self.node_idx, self.member)

    def matches(self, name: str) -> bool:
        fn = self.fn
        return fn.name == name or fn.id == name or getattr(fn, "family", None) == name

    def children(self, primitive_batches: "str | None" = None) -> Iterator["FunctionCall"]:
        """Direct child calls in canonical order. `batch` members and `scan` iterations are direct children
        of the enclosing composite (the combinator node itself is transparent). Members of a *primitive*
        batch (e.g. 1.5 B `Input` gates) are enumerated only when `primitive_batches` names their primitive
        (they are gates; address them as `C[i]` otherwise)."""
        if self.is_primitive:
            return
        yield from _node_calls(self.circuit, self.scope(), primitive_batches)

    # -- declared interface -------------------------------------------------------------
    @property
    def args(self) -> tuple[int, ...]:
        if self.is_primitive:
            return _prim_member_args(self.parent, self.node, self.member)
        sc = self.scope()
        out: list[int] = []
        for pi, (_, t) in enumerate(sc.spec.body.params):
            for j in range(t.leaves):
                out.append(param_leaf(sc, pi, j))
        return tuple(out)

    @property
    def returns(self) -> tuple[int, ...]:
        if self.is_primitive:
            return (self.lo,)
        sc = self.scope()
        return tuple(resolve(sc, r) for r in sc.spec.body.ret)

    def arg_types(self) -> tuple:
        return tuple(t for _, t in self.fn.signature()[0])

    def return_type(self):
        return self.fn.signature()[1]

    # -- graph boundary -------------------------------------------------------------------
    @property
    def in_(self) -> tuple[int, ...]:
        lo, hi = self.lo, self.hi
        return tuple(sorted({g for g in self.args if not (lo <= g < hi)}))

    @property
    def distinct_returns(self) -> tuple[int, ...]:
        """Distinct returned values produced inside the call (declared-interface adapter; ascending). A
        conservative superset of `out`: it does not check that anything consumes them."""
        lo, hi = self.lo, self.hi
        return tuple(sorted({g for g in self.returns if lo <= g < hi}))

    @property
    def out(self) -> tuple[int, ...]:
        """The graph boundary: distinct values produced inside and consumed outside or protected (root
        outputs). Contextual — requires the consumers, i.e. an explicit scan of the circuit (`explicit_limit`);
        candidates are restricted to `distinct_returns` because nothing else can leave a call."""
        cands = set(self.distinct_returns)
        if not cands:
            return ()
        return tuple(g for g in Region.out.fget(self) if g in cands)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"[{self.lo}:{self.hi}] {self.id}#{self.member}"


def _node_calls(circuit: "Circuit", sc: Scope, primitive_batches: "str | None") -> Iterator["FunctionCall"]:
    for k, node in enumerate(sc.spec.body.nodes):
        if node.form in ("primitive", "call"):
            yield FunctionCall(circuit, sc, k, 0)
        elif isinstance(node.fn, PrimitiveDefinition):
            fn = node.fn
            if primitive_batches is not None and (fn.name == primitive_batches or fn.id == primitive_batches
                                                  or getattr(fn, "family", None) == primitive_batches):
                for m in range(node.n):
                    yield FunctionCall(circuit, sc, k, m)
        else:
            for m in range(node.n):
                yield FunctionCall(circuit, sc, k, m)


class Circuit:
    """The canonical circuit of a program: `C.size`, `C[i]`, `C[i:j]`, root calls, gate lookup by division."""

    def __init__(self, root: SpecializedDefinition, explicit_limit: int = 5_000_000) -> None:
        self.root = root
        self.root_scope = Scope(root, 0, None, -1, 0)
        self.explicit_limit = explicit_limit

    @property
    def size(self) -> int:
        return self.root.gates

    def _check_explicit(self, n: int) -> None:
        if n > self.explicit_limit:
            raise ValueError(f"explicit enumeration over {n} gates exceeds explicit_limit={self.explicit_limit}")

    def gate(self, i: int) -> Gate:
        if not (0 <= i < self.size):
            raise IndexError(i)
        scope = self.root_scope
        while True:
            body = scope.spec.body
            k = body.node_at(i - scope.offset)
            node = body.nodes[k]
            node_off = scope.offset + body.offsets[k]
            m = 0 if node.form in ("primitive", "call") else (i - node_off) // node.fn.gates
            if isinstance(node.fn, PrimitiveDefinition):
                return Gate(i, node.fn, _prim_member_args(scope, node, m), scope)
            scope = Scope(node.fn, node_off + m * node.fn.gates, scope, k, m)

    def __getitem__(self, i):
        if isinstance(i, slice):
            lo, hi, step = i.indices(self.size)
            if step != 1:
                raise ValueError("unit stride only")
            return Region(self, lo, hi)
        return self.gate(i)

    def region(self, lo: int, hi: int) -> Region:
        return Region(self, lo, hi)

    def gate_selection(self, gates: Sequence[int]) -> GateSelection:
        return GateSelection(self, gates)

    def root_calls(self, primitive_batches: "str | None" = None) -> Iterator[FunctionCall]:
        yield from _node_calls(self, self.root_scope, primitive_batches)

    def root_outputs(self) -> tuple[int, ...]:
        return tuple(resolve(self.root_scope, r) for r in self.root.body.ret)
