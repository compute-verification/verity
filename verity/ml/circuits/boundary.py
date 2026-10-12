"""Compact boundary analysis over `Part`s: in(S), out(S) and W_out(S) without expanded-gate traversal.

Circuit terminology (gate-centric, see `verity.ml.circuits.parts`): gates are `Value<w>` producers, wires are operand edges.

    in(S)  = { u not in S : exists v in S, (u, v) in E }          distinct gates outside S with a wire into S
    out(S) = { u in S : u designated output  or  exists v not in S, (u, v) in E }
    w_out(S) = sum_{g in out(S)} w(g)

Fan-out counts the producer once; a declared-but-unread parameter is not a wire; an unread nested return is not
automatically an escaping value; `Input<w>` gates are ordinary gates; designated outputs are `program.root_outputs()`.

Method.  Everything is derived from three *intrinsic*, per-specialization summaries — valid for
every activation of the specialization, cached once:

1. the canonical return form (`_ret_canon`): every return leaf of the body named by its *producing* position in the body
   — `("n", k, leaf)` = output leaf of node `k` whose gate lies inside node `k`; `("s", k, m*c + l)` = carry leaf `l`
   produced inside iteration `m < n-1` of scan node `k` (intermediate carries have no body-level name otherwise);
   `("p", i, l)` = the body passes its own parameter leaf through.  A node that merely returns an argument is therefore
   transparent: its consumers are charged to the true producer.
2. the consumer index (`consumer_index`): for every producer `(space, node)` the consumer nodes and *how their members*
   read it — `_All` (every member reads these leaves), `_Aff` (member m reads `base + m*stride + used`), `_Aff2`
   (member m reads `base + m*m_stride + i*i_stride, i in used`: a Strided view read row by row or column by column),
   `_One` (one member reads these leaves), `_Lazy` (only the union over members is known).  Reads are filtered by the consumer
   callee's used parameter leaves (3), so a dead parameter is not an edge.  Scan carries are intra-node edges in the
   "s" space.
3. the used parameter leaves (`used_param_leaves`): parameter leaves read by some gate of the body, transitively,
   collected from the same pass over canonical reads.

For a part (activation `scope`, items `(j, m)`):

    out(part) = { canonical leaves of the items read by a consumer item outside the part }
              U { returned leaves sourced in the part that the parent actually consumes (recursively; root: designated) }
    in(part)  = { producers of the items' canonical reads that are not items of the part }

Exactness rule: `out_gates` / `in_gates` / `w_out` return the exact set or `Unsupported`; they never substitute the
bound.  `w_out_bound` never returns `Unsupported`: it uses the same index with over-approximate unions, charges every
returned leaf sourced in the part, and on any internal limit falls back to the sum of the declared output widths of the
items (sound: a gate can only leave a member through the member's outputs).

Limits (`parts.Limits` plus the module caps below) turn expensive answers into `Unsupported`, never into a wrong one.
In particular a callee whose reads of a parameter are only known conservatively (a cap tripped while listing partial
per-member reads) makes every producer wired into that argument an unresolved edge (`_Lazy(exact=False)`) in the
enclosing body, recursively upwards: exact queries touching those producers are `Unsupported`; the other arguments and
every other part stay exact.
"""
from __future__ import annotations

import bisect
import weakref
from dataclasses import dataclass
from typing import Iterator

from .defs import Body, Node, PrimitiveDefinition, SpecializedDefinition
from .intervals import IntervalLimit, Intervals, contains, full, isect, norm, ref_intervals, total
from .layout import Scope, resolve_gate
from .parts import Limits, Part, Unsupported, n_members
from .refs import Affine, Refs, Strided, runs
from .types import Array, Record, Tuple as TupleT, Type, Value

Key = tuple[str, int]  # (space, idx): "n" node output leaves, "s" scan-internal carry leaves, "p" parameter leaves

_MEMBER_CAP = 1 << 16       # members of one consumer node resolved individually before the edge becomes `_Lazy`
_PASSTHROUGH_CAP = 1 << 18  # members / iterations walked while canonicalising through pass-through returns
_PASSTHROUGH_DEPTH = 400    # chained pass-through carries followed before giving up
_INTERVAL_CAP = 1 << 20     # interval tuples listed (transiently) for one two-dimensional parameter read before it goes conservative


class BoundaryLimit(Exception):
    """An exact answer would exceed a limit.  The exact entry points convert it into `Unsupported`; it escapes only
    from `used_param_leaves` (whose frozen signature has no `Unsupported`)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail

    def unsupported(self) -> Unsupported:
        return Unsupported(self.reason, self.detail)


def _restrict_refs(refs: Refs, used: Intervals) -> list[Refs]:
    """`refs` cut down to the leaf positions in `used` (the whole sequence when `used` covers it)."""
    if not used:
        return []
    if len(used) == 1 and used[0] == (0, refs.count):
        return [refs]
    return [refs.slice(lo, hi) for lo, hi in used if hi > lo]


def _leaf_sets(refs_list: list[Refs]) -> dict[Key, Intervals]:
    """{(space, idx): leaf intervals} referenced by the sequences (a set: order and repeats dropped).  Strided
    sequences use the exact interval decomposition of `intervals.strided_intervals`; one without interval structure and
    more than its enumeration limit raises `BoundaryLimit`."""
    acc: dict[Key, list[tuple[int, int]]] = {}
    try:
        for r in refs_list:
            ref_intervals(r, acc)
    except IntervalLimit as e:
        raise BoundaryLimit("strided_enumeration", str(e)) from None
    return {k: norm(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------------------------------------
# node geometry: output leaves <-> members
# ---------------------------------------------------------------------------------------------------------


def _scan_shape(node: Node) -> tuple[int, int]:
    """(carry leaves c, per-iteration y leaves) of a scan node; output leaves are `[0, c)` = final carry, then
    `c + m*y + r` = leaf `r` of iteration m's y."""
    return node.out.fields[0].leaves, node.out.fields[1].elem.leaves


def _member_leaves(node: Node, m: int) -> Intervals:
    """Output leaves ("n" space) produced by member `m` of `node` (all of them for primitive / call nodes)."""
    if node.form in ("primitive", "call"):
        return full(node.out_leaves)
    if node.form == "batch":
        L = node.fn.out_leaves
        return ((m * L, (m + 1) * L),) if L else ()
    c, y = _scan_shape(node)
    ivs: list[tuple[int, int]] = []
    if c and m == node.n - 1:
        ivs.append((0, c))
    if y:
        ivs.append((c + m * y, c + (m + 1) * y))
    return norm(ivs)


def _members_leaves(node: Node, ms: set[int]) -> Intervals:
    if node.form in ("primitive", "call") or len(ms) == n_members(node):
        return full(node.out_leaves)
    if node.form == "batch":
        L = node.fn.out_leaves
        return norm([(m * L, (m + 1) * L) for m in ms]) if L else ()
    ivs: list[tuple[int, int]] = []
    for m in ms:
        ivs.extend(_member_leaves(node, m))
    return norm(ivs)


def _leaf_member(node: Node, leaf: int) -> int:
    if node.form in ("primitive", "call"):
        return 0
    if node.form == "batch":
        return leaf // node.fn.out_leaves
    c, y = _scan_shape(node)
    return node.n - 1 if leaf < c else (leaf - c) // y


def _carry_leaves(node: Node, ms: set[int]) -> Intervals:
    """"s"-space leaves of the part's iterations of a scan node: `m*c + l` for iterations `m < n-1`."""
    c, _ = _scan_shape(node)
    if not c:
        return ()
    return norm([(m * c, (m + 1) * c) for m in ms if m < node.n - 1])


def _sum_widths(t: Type) -> int:
    if isinstance(t, Value):
        return t.w
    if isinstance(t, Array):
        return t.n * _sum_widths(t.elem)
    if isinstance(t, TupleT):
        return sum(_sum_widths(f) for f in t.fields)
    if isinstance(t, Record):
        return sum(_sum_widths(f) for _, f in t.fields)
    raise TypeError(t)


# ---------------------------------------------------------------------------------------------------------
# resolution to gates (layout.resolve_gate: the gate and its producing primitive)
# ---------------------------------------------------------------------------------------------------------


def _gate_of(scope: Scope, key: Key, leaf: int) -> tuple[int, PrimitiveDefinition]:
    """Gate of a canonical producer leaf of `scope`'s body."""
    space, idx = key
    if space == "s":
        node = scope.spec.body.nodes[idx]
        c, _ = _scan_shape(node)
        m, l = divmod(leaf, c)
        return resolve_gate(scope.child(idx, m), node.fn.body.ret[l])
    return resolve_gate(scope, (space, idx, leaf))


def gate_prim(program, g: int) -> PrimitiveDefinition:
    """The primitive producing root gate `g`, located by bisection along the hierarchy."""
    spec: SpecializedDefinition = program.root
    if not (0 <= g < spec.gates):
        raise IndexError(g)
    off = 0
    while True:
        body = spec.body
        k = body.node_at(g - off)
        node = body.nodes[k]
        node_off = off + body.offsets[k]
        fn = node.fn
        if isinstance(fn, PrimitiveDefinition):
            return fn
        m = 0 if node.form == "call" else (g - node_off) // fn.gates
        spec, off = fn, node_off + m * fn.gates


def gate_width(program, g: int) -> int:
    """`w(g)` of root gate `g`: the producing primitive's `ret.w`, located by bisection along the hierarchy."""
    return gate_prim(program, g).ret.w


# ---------------------------------------------------------------------------------------------------------
# canonical return form (pass-through returns resolved to producers)
# ---------------------------------------------------------------------------------------------------------


Piece = tuple[int, int, str, int, int]  # (pos_lo, pos_hi, space, idx, base): positions [pos_lo, pos_hi) -> (space, idx, base + (pos - pos_lo))


class _RetCanon:
    __slots__ = ("pieces", "starts", "passthrough", "by_src")

    def __init__(self, pieces: list[Piece]) -> None:
        self.pieces = pieces
        self.starts = [p[0] for p in pieces]
        self.passthrough = any(p[2] == "p" for p in pieces)
        by_src: dict[Key, list[tuple[int, int, int]]] = {}
        for plo, phi, sp, idx, base in pieces:
            if sp != "p":
                by_src.setdefault((sp, idx), []).append((plo, phi, base))
        self.by_src = by_src

    def slice(self, lo: int, hi: int) -> Iterator[Piece]:
        """Pieces clipped to return positions [lo, hi), in order."""
        k = max(bisect.bisect_right(self.starts, lo) - 1, 0)
        while k < len(self.pieces) and self.pieces[k][0] < hi:
            plo, phi, sp, idx, base = self.pieces[k]
            a, b = max(lo, plo), min(hi, phi)
            if b > a:
                yield (a, b, sp, idx, base + (a - plo))
            k += 1

    def positions(self, key: Key, leaf: int) -> list[int]:
        """Return positions whose canonical source is `leaf` of `key` (several when a value is returned twice)."""
        out = []
        for plo, phi, base in self.by_src.get(key, ()):
            if base <= leaf < base + (phi - plo):
                out.append(plo + (leaf - base))
        return out


_RET: "weakref.WeakKeyDictionary[SpecializedDefinition, _RetCanon]" = weakref.WeakKeyDictionary()


def _ret_canon(spec: SpecializedDefinition) -> _RetCanon:
    rc = _RET.get(spec)
    if rc is None:
        pieces: list[Piece] = []
        pos = 0
        for sp, idx, lo, hi in _canon_refs_ordered(spec.body, spec.body.ret, 0):
            pieces.append((pos, pos + hi - lo, sp, idx, lo))
            pos += hi - lo
        rc = _RetCanon(pieces)
        _RET[spec] = rc
    return rc


def _is_passthrough_node(node: Node) -> bool:
    return not isinstance(node.fn, PrimitiveDefinition) and _ret_canon(node.fn).passthrough


def _canon_refs_ordered(body: Body, refs: Refs, depth: int) -> list[tuple[str, int, int, int]]:
    out: list[tuple[str, int, int, int]] = []
    for sp, idx, lo, hi in runs(refs):
        out.extend(_canon_ordered(body, sp, idx, lo, hi, depth))
    return out


def _member_arg_refs(node: Node, m: int, pi: int, lo: int, hi: int) -> Refs:
    """Leaves [lo, hi) of parameter `pi` of member `m` of a batch / scan node, as references in the enclosing body."""
    a = node.args[pi]
    if node.form == "batch":
        ax = node.axes[pi]
        if ax is None:
            return a.refs.slice(lo, hi)
        if ax == 0:
            L = a.type.elem.leaves
            return a.refs.slice(m * L + lo, m * L + hi)
        return a.take(ax, m).refs.slice(lo, hi)
    # scan: pi == 0 handled by the caller (init / previous carry); xs per iteration; shared whole
    if 1 <= pi <= node.n_xs:
        L = a.type.elem.leaves
        return a.refs.slice(m * L + lo, m * L + hi)
    return a.refs.slice(lo, hi)


def _canon_ordered(body: Body, space: str, idx: int, lo: int, hi: int, depth: int) -> list[tuple[str, int, int, int]]:
    """Canonical producer runs, in order, of leaves [lo, hi) of `(space, idx)` in `body`."""
    if hi <= lo:
        return []
    if space != "n":
        return [(space, idx, lo, hi)]
    node = body.nodes[idx]
    fn = node.fn
    if isinstance(fn, PrimitiveDefinition):
        return [("n", idx, lo, hi)]
    rc = _ret_canon(fn)
    if not rc.passthrough:
        return [("n", idx, lo, hi)]
    if depth > _PASSTHROUGH_DEPTH:
        raise BoundaryLimit("passthrough_depth", f"node {idx} ({fn.id})")
    out: list[tuple[str, int, int, int]] = []
    if node.form == "call":
        for plo, phi, sp, i, base in rc.slice(lo, hi):
            if sp == "p":
                out.extend(_canon_refs_ordered(body, node.args[i].refs.slice(base, base + phi - plo), depth + 1))
            else:
                out.append(("n", idx, plo, phi))
        return out
    if node.form == "batch":
        L = fn.out_leaves
        m_lo, m_hi = lo // L, (hi - 1) // L + 1
        if m_hi - m_lo > _PASSTHROUGH_CAP:
            raise BoundaryLimit("passthrough_members", f"node {idx} ({fn.id}): {m_hi - m_lo} members")
        for m in range(m_lo, m_hi):
            a, b = max(lo, m * L) - m * L, min(hi, (m + 1) * L) - m * L
            for plo, phi, sp, i, base in rc.slice(a, b):
                if sp == "p":
                    out.extend(_canon_refs_ordered(body, _member_arg_refs(node, m, i, base, base + phi - plo), depth + 1))
                else:
                    out.append(("n", idx, m * L + plo, m * L + phi))
        return out
    # scan
    c, y = _scan_shape(node)
    if lo < c:
        out.extend(_canon_scan_iter(body, idx, node.n - 1, lo, min(hi, c), depth))
    if hi > c:
        a0, b0 = max(lo, c) - c, hi - c
        m_lo, m_hi = a0 // y, (b0 - 1) // y + 1
        if m_hi - m_lo > _PASSTHROUGH_CAP:
            raise BoundaryLimit("passthrough_members", f"node {idx} ({fn.id}): {m_hi - m_lo} iterations")
        for m in range(m_lo, m_hi):
            a, b = max(a0, m * y) - m * y, min(b0, (m + 1) * y) - m * y
            out.extend(_canon_scan_iter(body, idx, m, c + a, c + b, depth))
    return out


def _canon_scan_iter(body: Body, k: int, m: int, qlo: int, qhi: int, depth: int) -> list[tuple[str, int, int, int]]:
    """Canonical runs of return positions [qlo, qhi) of iteration `m` of scan node `k` of `body`."""
    node = body.nodes[k]
    fn = node.fn
    c, y = _scan_shape(node)
    rc = _ret_canon(fn)

    def name(plo: int, phi: int) -> tuple[str, int, int, int]:  # positions of iteration m produced inside it
        if plo < c:  # carry positions
            if m < node.n - 1:
                return ("s", k, m * c + plo, m * c + phi)
            return ("n", k, plo, phi)
        return ("n", k, c + m * y + (plo - c), c + m * y + (phi - c))

    if not rc.passthrough:
        out = []
        if qlo < c:
            out.append(name(qlo, min(qhi, c)))
        if qhi > c:
            out.append(name(max(qlo, c), qhi))
        return out
    if depth > _PASSTHROUGH_DEPTH:
        raise BoundaryLimit("passthrough_depth", f"scan node {k} ({fn.id})")
    out: list[tuple[str, int, int, int]] = []
    for plo, phi, sp, i, base in rc.slice(qlo, qhi):
        if sp != "p":
            # callers split their ranges at c, so a piece straddling the carry / y boundary is defensive only
            if plo < c < phi:
                out.append(name(plo, c))
                out.append(name(c, phi))
            else:
                out.append(name(plo, phi))
        elif i == 0:  # the step passes its carry through
            if m == 0:
                out.extend(_canon_refs_ordered(body, node.args[0].refs.slice(base, base + phi - plo), depth + 1))
            else:
                out.extend(_canon_scan_iter(body, k, m - 1, base, base + phi - plo, depth + 1))
        else:
            out.extend(_canon_refs_ordered(body, _member_arg_refs(node, m, i, base, base + phi - plo), depth + 1))
    return out


def _canon_sets(body: Body, refs_list: list[Refs]) -> dict[Key, Intervals]:
    """Canonical producer leaf sets of the sequences (unordered)."""
    raw = _leaf_sets(refs_list)
    out: dict[Key, list[tuple[int, int]]] = {}
    for (sp, idx), ivs in raw.items():
        if sp == "n" and _is_passthrough_node(body.nodes[idx]):
            for lo, hi in ivs:
                for s2, i2, l2, h2 in _canon_ordered(body, "n", idx, lo, hi, 0):
                    out.setdefault((s2, i2), []).append((l2, h2))
        else:
            out.setdefault((sp, idx), []).extend(ivs)
    return {k: norm(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------
# consumer index
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _All:
    """Every member of the consumer node reads `ivs` of the producer."""
    ivs: Intervals


@dataclass(frozen=True)
class _Aff:
    """Member m in [m_lo, m_hi) of the consumer node reads `{base + m*stride + u : u in used}` of the producer."""
    base: int
    stride: int
    used: Intervals
    m_lo: int
    m_hi: int

    def reader(self, leaf: int) -> int | None:
        q, r = divmod(leaf - self.base, self.stride)
        return q if self.m_lo <= q < self.m_hi and contains(self.used, r) else None


@dataclass(frozen=True)
class _Aff2:
    """Member m in [m_lo, m_hi) reads `{base + m*m_stride + i*i_stride : i in used}` (`used` within `[0, inner)`): a
    `Strided` argument read member by member (rows of a matrix view, columns of a table, ...).  The strides make the
    (m, i) of a leaf unique: `major == "m"` when `m_stride >= inner*i_stride` (a member's row lies before the next
    member's), `major == "i"` when `i_stride >= m_hi*m_stride` (each i-row spans every member, e.g. the columns of an
    embedding table)."""
    base: int
    m_stride: int
    i_stride: int
    inner: int
    used: Intervals
    m_lo: int
    m_hi: int
    major: str

    def reader(self, leaf: int) -> int | None:
        d = leaf - self.base
        if d < 0:
            return None
        if self.major == "m":
            m, rem = divmod(d, self.m_stride)
            i, r2 = divmod(rem, self.i_stride)
        else:
            i, rem = divmod(d, self.i_stride)
            m, r2 = divmod(rem, self.m_stride)
        if r2 or i >= self.inner or not (self.m_lo <= m < self.m_hi) or not contains(self.used, i):
            return None
        return m

    def leaves(self, cap: int) -> Intervals | None:
        """The leaves read by all members as intervals, or None when listing them would exceed `cap` tuples."""
        n = self.m_hi - self.m_lo
        n_used = total(self.used)
        if self.major == "i" and self.m_stride == 1:  # every i-row is one run over the members
            if n_used > cap:
                return None
            return norm([(self.base + i * self.i_stride + self.m_lo, self.base + i * self.i_stride + self.m_hi)
                          for lo, hi in self.used for i in range(lo, hi)])
        if self.major == "m" and self.i_stride == 1:  # every member's row is a run
            if n * len(self.used) > cap:
                return None
            return norm([(self.base + m * self.m_stride + lo, self.base + m * self.m_stride + hi)
                          for m in range(self.m_lo, self.m_hi) for lo, hi in self.used])
        if n * n_used > cap:
            return None
        return norm([(self.base + m * self.m_stride + i * self.i_stride, self.base + m * self.m_stride + i * self.i_stride + 1)
                      for m in range(self.m_lo, self.m_hi) for lo, hi in self.used for i in range(lo, hi)])

    def hull(self) -> Intervals:
        """One interval covering every leaf any member could read (a superset)."""
        last = self.base + (self.m_hi - 1) * self.m_stride + (self.inner - 1) * self.i_stride
        return ((self.base + self.m_lo * self.m_stride, last + 1),)


@dataclass(frozen=True)
class _One:
    """Member `m` of the consumer node reads `ivs` of the producer."""
    m: int
    ivs: Intervals


@dataclass(frozen=True)
class _Lazy:
    """Per-member reads were not resolved (too many members); `union` is a superset of what all members read
    together, exact iff `exact`."""
    union: Intervals
    exact: bool


Desc = object  # _All | _Aff | _Aff2 | _One | _Lazy


class ConsumerIndex:
    """Per-specialization consumer index (intrinsic; shared by every activation).

    `edges[(space, node)]` = `[(consumer node, descriptor)]`; `used[pi]` = parameter leaves read (intervals);
    `ret` = canonical return form.  `incomplete` names the reason when some edge could not be built exactly
    (then every exact query on this specialization is `Unsupported`; the bound falls back to declared widths)."""

    __slots__ = ("spec_id", "body", "edges", "used", "used_exact_params", "ret", "incomplete", "__weakref__")

    def __init__(self, spec: SpecializedDefinition) -> None:
        self.spec_id = spec.id  # (the spec itself is not held: the cache key must stay collectable)
        self.body = spec.body
        self.edges: dict[Key, list[tuple[int, Desc]]] = {}
        self.used: tuple[Intervals, ...] = ()
        self.used_exact_params: tuple[bool, ...] = tuple(True for _ in self.body.params)
        self.incomplete: str | None = None
        try:
            self.ret = _ret_canon(spec)
            self._build()
        except BoundaryLimit as e:
            self.incomplete = str(e)
            self.used = tuple(full(t.leaves) for _, t in self.body.params)  # conservative: everything read
            self.used_exact_params = tuple(False for _ in self.body.params)
            if not hasattr(self, "ret"):
                self.ret = _RetCanon([])

    @property
    def used_exact(self) -> bool:
        """Whether `used` is exact for every parameter.  A parameter is inexact (conservatively over-approximated)
        when a nested callee's own used leaves are inexact and the parameter flows into it, or when the index is
        incomplete; the enclosing body's index then marks every producer wired into that call as unresolved
        (`_Lazy(exact=False)`) so exact queries touching it are `Unsupported` rather than wrong."""
        return all(self.used_exact_params)

    # -- construction ---------------------------------------------------------------------------------
    def _build(self) -> None:
        body = self.body
        pacc: list[list[tuple[int, int]]] = [[] for _ in body.params]
        pexact = [True for _ in body.params]
        edges = self.edges

        def used_of(fn) -> tuple[tuple[Intervals, ...], tuple[bool, ...]]:
            if isinstance(fn, PrimitiveDefinition):
                return tuple(full(t.leaves) for _, t in fn.params), tuple(True for _ in fn.params)
            ix = consumer_index(fn)
            return ix.used, ix.used_exact_params

        def add_inexact(jp: int, node: Node, pi: int) -> None:
            """Argument `pi` flows into a callee whose reads of it are not exact: every producer it references is an
            unresolved consumer edge (exact queries on it become `Unsupported`), parameters become inexact here too."""
            for key, ivs in _canon_sets(body, [node.args[pi].refs]).items():
                if key[0] == "p":
                    pacc[key[1]].extend(ivs)
                    pexact[key[1]] = False
                else:
                    edges.setdefault(key, []).append((jp, _Lazy(ivs, False)))
            if node.form == "scan" and pi == 0 and node.n > 1:
                c, _ = _scan_shape(node)
                if c:
                    edges.setdefault(("s", jp), []).append((jp, _Lazy(full((node.n - 1) * c), False)))

        def add(jp: int, sets: dict[Key, Intervals], make) -> None:
            for key, ivs in sets.items():
                if key[0] == "p":
                    pacc[key[1]].extend(ivs)
                else:
                    edges.setdefault(key, []).append((jp, make(ivs)))

        def add_aff(jp: int, key: Key, base: int, stride: int, used: Intervals, m_lo: int, m_hi: int) -> None:
            if key[0] == "p":
                if used == full(stride):
                    pacc[key[1]].append((base + m_lo * stride, base + m_hi * stride))
                elif (m_hi - m_lo) * len(used) <= _MEMBER_CAP:
                    pacc[key[1]].extend((base + m * stride + lo, base + m * stride + hi) for m in range(m_lo, m_hi) for lo, hi in used)
                else:  # too many partial per-member reads to list: this parameter's reads become conservative
                    pacc[key[1]].append((base + m_lo * stride, base + m_hi * stride))
                    pexact[key[1]] = False
            else:
                edges.setdefault(key, []).append((jp, _Aff(base, stride, used, m_lo, m_hi)))

        def add_aff2(jp: int, key: Key, d: _Aff2) -> None:
            if key[0] == "p":
                ivs = d.leaves(_INTERVAL_CAP)
                if ivs is None:  # too many runs to list: this parameter's reads become conservative (its hull)
                    ivs = d.hull()
                    pexact[key[1]] = False
                pacc[key[1]].extend(ivs)
            else:
                edges.setdefault(key, []).append((jp, d))

        def per_member(jp: int, node: Node, pi: int, used: Intervals, L: int) -> None:
            """Axis-0 / xs argument `pi` read member by member (non-affine references)."""
            a = node.args[pi]
            r = a.refs
            if L == 0:
                return
            if isinstance(r, Affine) and (r.space != "n" or not _is_passthrough_node(body.nodes[r.idx])):
                add_aff(jp, (r.space, r.idx), r.base, L, used, 0, node.n)
                return
            if isinstance(r, Strided) and r.inner == L and r.count == node.n * L and \
                    (r.space != "n" or not _is_passthrough_node(body.nodes[r.idx])):
                key = (r.space, r.idx)
                if r.outer_stride == 0:  # broadcast: every member reads the same row
                    add(jp, _canon_sets(body, _restrict_refs(r.slice(0, L), used)), _All)
                    return
                if r.inner_stride == 1 and r.outer_stride >= L:
                    add_aff(jp, key, r.base, r.outer_stride, used, 0, node.n)  # non-overlapping rows: one reader per leaf
                    return
                if r.inner_stride > 0 and r.outer_stride > 0:
                    major = "m" if r.outer_stride >= L * r.inner_stride else ("i" if r.inner_stride >= node.n * r.outer_stride else None)
                    if major is not None:  # unique (member, i) per leaf: a two-dimensional affine read
                        add_aff2(jp, key, _Aff2(r.base, r.outer_stride, r.inner_stride, L, used, 0, node.n, major))
                        return
            if node.n <= _MEMBER_CAP:
                for m in range(node.n):
                    add(jp, _canon_sets(body, _restrict_refs(r.slice(m * L, (m + 1) * L), used)), lambda ivs, m=m: _One(m, ivs))
                return
            whole = used == full(L)
            for key, ivs in _canon_sets(body, [r]).items():
                if key[0] == "p":
                    pacc[key[1]].extend(ivs)
                    if not whole:
                        pexact[key[1]] = False
                else:
                    edges.setdefault(key, []).append((jp, _Lazy(ivs, whole)))

        for jp, node in enumerate(body.nodes):
            fn = node.fn
            if node.form == "primitive":
                add(jp, _canon_sets(body, [a.refs for a in node.args]), _All)
                continue
            used, exact = used_of(fn)
            for pi in range(len(node.args)):
                if not exact[pi]:
                    add_inexact(jp, node, pi)
            if node.form == "call":
                refs = [s for pi, a in enumerate(node.args) if exact[pi] for s in _restrict_refs(a.refs, used[pi])]
                add(jp, _canon_sets(body, refs), _All)
            elif node.form == "batch":
                for pi, (a, ax) in enumerate(zip(node.args, node.axes)):
                    u = used[pi]
                    if not u or not exact[pi]:
                        continue
                    if ax is None:
                        add(jp, _canon_sets(body, _restrict_refs(a.refs, u)), _All)
                    elif ax == 0:
                        per_member(jp, node, pi, u, a.type.elem.leaves)
                    elif node.n <= _MEMBER_CAP:
                        for m in range(node.n):
                            add(jp, _canon_sets(body, _restrict_refs(a.take(ax, m).refs, u)), lambda ivs, m=m: _One(m, ivs))
                    else:
                        whole = u == full(a.take(ax, 0).refs.count)
                        for key, ivs in _canon_sets(body, [a.refs]).items():
                            if key[0] == "p":
                                pacc[key[1]].extend(ivs)
                                if not whole:
                                    pexact[key[1]] = False
                            else:
                                edges.setdefault(key, []).append((jp, _Lazy(ivs, whole)))
            else:  # scan
                c, _y = _scan_shape(node)
                u0 = used[0] if exact[0] else ()
                if u0:
                    add(jp, _canon_sets(body, _restrict_refs(node.args[0].refs, u0)), lambda ivs: _One(0, ivs))
                    if node.n > 1:
                        if not _ret_canon(fn).passthrough:
                            edges.setdefault(("s", jp), []).append((jp, _Aff(-c, c, u0, 1, node.n)))
                        else:
                            if node.n > _PASSTHROUGH_CAP:
                                raise BoundaryLimit("passthrough_members", f"{self.spec_id} scan node {jp}")
                            for m in range(1, node.n):
                                acc: dict[Key, list[tuple[int, int]]] = {}
                                for lo, hi in u0:
                                    for s2, i2, l2, h2 in _canon_scan_iter(body, jp, m - 1, lo, hi, 0):
                                        acc.setdefault((s2, i2), []).append((l2, h2))
                                add(jp, {k: norm(v) for k, v in acc.items()}, lambda ivs, m=m: _One(m, ivs))
                for pi in range(1, 1 + node.n_xs):
                    if used[pi] and exact[pi]:
                        per_member(jp, node, pi, used[pi], node.args[pi].type.elem.leaves)
                for pi in range(1 + node.n_xs, len(node.args)):
                    if used[pi] and exact[pi]:
                        add(jp, _canon_sets(body, _restrict_refs(node.args[pi].refs, used[pi])), _All)
        self.used = tuple(norm(p) for p in pacc)
        self.used_exact_params = tuple(pexact)

    # -- queries --------------------------------------------------------------------------------------
    def is_read(self, key: Key, leaf: int) -> bool:
        """Whether some item of the body reads canonical producer leaf `leaf` of `key`.  Raises `BoundaryLimit`
        when an unresolved edge leaves it undetermined."""
        for _jp, d in self.edges.get(key, ()):
            if isinstance(d, (_All, _One)):
                if contains(d.ivs, leaf):
                    return True
            elif isinstance(d, (_Aff, _Aff2)):
                if d.reader(leaf) is not None:
                    return True
            elif contains(d.union, leaf):
                if d.exact:
                    return True
                raise BoundaryLimit("unresolved_consumer", f"{self.spec_id}: node {_jp} reads {key} with unresolved members")
        return False

    def consumers(self, key: Key) -> list[tuple[int, Desc]]:
        return self.edges.get(key, [])


_INDEX: "weakref.WeakKeyDictionary[SpecializedDefinition, ConsumerIndex]" = weakref.WeakKeyDictionary()
_USED: "weakref.WeakKeyDictionary[SpecializedDefinition, frozenset[tuple[int, int]]]" = weakref.WeakKeyDictionary()


def consumer_index(spec: SpecializedDefinition) -> ConsumerIndex:
    """The consumer index of `spec` (cached per specialization; the root of a Program costs O(root nodes) once)."""
    ix = _INDEX.get(spec)
    if ix is None:
        ix = ConsumerIndex(spec)
        _INDEX[spec] = ix
    return ix


def _used_params(spec: SpecializedDefinition) -> tuple[Intervals, ...]:
    """Exact used parameter leaves of `spec`; raises `BoundaryLimit` when some parameter's reads are only known
    conservatively (the exact paths must never consume an over-approximation as if it were exact)."""
    ix = consumer_index(spec)
    if not ix.used_exact:
        inexact = [pi for pi, ok in enumerate(ix.used_exact_params) if not ok]
        raise BoundaryLimit("used_params", ix.incomplete or f"{spec.id}: parameter reads {inexact} are not exact")
    return ix.used


def used_param_intervals(spec: SpecializedDefinition) -> tuple[Intervals, ...]:
    """Per parameter, the leaf intervals read (transitively) by some gate of `spec`'s body — the compact form of
    `used_param_leaves`, O(#runs) even for a weight tensor.  Raises `BoundaryLimit("used_params")` when a cap left
    some parameter's reads inexact (see `ConsumerIndex.used_exact`)."""
    return _used_params(spec)


def used_param_leaves(spec: SpecializedDefinition) -> frozenset[tuple[int, int]]:
    """`{(parameter index, leaf)}` read transitively by some gate of `spec` (cached).  A declared-but-unread leaf is
    absent: it is not a wire into the activation.  The expansion is O(#leaves read); use `used_param_intervals`
    for the compact form."""
    u = _USED.get(spec)
    if u is None:
        ivs = used_param_intervals(spec)
        u = frozenset((pi, leaf) for pi, iv in enumerate(ivs) for lo, hi in iv for leaf in range(lo, hi))
        _USED[spec] = u
    return u


# ---------------------------------------------------------------------------------------------------------
# per-item reads (for in_gates) and the boundary computations
# ---------------------------------------------------------------------------------------------------------


def _member_reads(spec: SpecializedDefinition, j: int, m: int) -> dict[Key, Intervals]:
    """Canonical producer leaves read by item `(j, m)` of `spec`'s body (used-filtered)."""
    body = spec.body
    node = body.nodes[j]
    fn = node.fn
    if node.form == "primitive":
        return _canon_sets(body, [a.refs for a in node.args])
    if node.form == "call":
        used = _used_params(fn)
        return _canon_sets(body, [s for pi, a in enumerate(node.args) for s in _restrict_refs(a.refs, used[pi])])
    if node.form == "batch":
        used = tuple(full(t.leaves) for _, t in fn.params) if isinstance(fn, PrimitiveDefinition) else _used_params(fn)
        refs: list[Refs] = []
        for pi, (a, ax) in enumerate(zip(node.args, node.axes)):
            u = used[pi]
            if not u:
                continue
            if ax is None:
                refs.extend(_restrict_refs(a.refs, u))
            elif ax == 0:
                L = a.type.elem.leaves
                refs.extend(_restrict_refs(a.refs.slice(m * L, (m + 1) * L), u))
            else:
                refs.extend(_restrict_refs(a.take(ax, m).refs, u))
        return _canon_sets(body, refs)
    # scan iteration m: (carry, xs..., shared...)
    used = _used_params(fn)
    refs = []
    acc: dict[Key, list[tuple[int, int]]] = {}
    if used[0]:
        if m == 0:
            refs.extend(_restrict_refs(node.args[0].refs, used[0]))
        else:
            for lo, hi in used[0]:
                for s2, i2, l2, h2 in _canon_scan_iter(body, j, m - 1, lo, hi, 0):
                    acc.setdefault((s2, i2), []).append((l2, h2))
    for pi in range(1, len(node.args)):
        if used[pi]:
            refs.extend(_restrict_refs(_member_arg_refs(node, m, pi, 0, node.args[pi].type.leaves if pi > node.n_xs else node.args[pi].type.elem.leaves), used[pi]))
    for key, ivs in _canon_sets(body, refs).items():
        acc.setdefault(key, []).extend(ivs)
    return {k: norm(v) for k, v in acc.items()}


def _check_part(part: Part) -> None:
    if len(part.items) > Limits.max_items_per_part:
        raise BoundaryLimit("too_many_items", f"{len(part.items)} items > Limits.max_items_per_part")
    if not part.items:
        raise BoundaryLimit("empty_part", "a Part needs at least one item")
    nodes = part.scope.spec.body.nodes
    for j, m in part.items:  # an item outside the body is a caller error, not an Unsupported case
        if not (0 <= j < len(nodes)) or not (0 <= m < n_members(nodes[j])):
            raise ValueError(f"item ({j}, {m}) is not in {part.scope.spec.id}: {len(nodes)} nodes"
                             + (f", node {j} has {n_members(nodes[j])} members" if 0 <= j < len(nodes) else ""))


def _is_root(scope: Scope) -> bool:
    """The root activation has no parent and no parameters.  A parameterised body presented with `parent=None` is a
    nested activation without its context: its returned leaves and parameter wires cannot be resolved exactly."""
    if scope.parent is not None:
        return False
    if scope.spec.body.params:
        raise BoundaryLimit("no_parent_scope", f"{scope.spec.id} has parameters but the Scope has no parent")
    return True


def _by_node(part: Part) -> dict[int, set[int]]:
    by_node: dict[int, set[int]] = {}
    for j, m in part.items:
        by_node.setdefault(j, set()).add(m)
    return by_node


def _aff_select(d: _Aff | _Aff2, pivs: Intervals, inside: set[int] | None, budget: list[int]) -> Iterator[int]:
    """Producer leaves in `pivs` read by a member of the `_Aff` / `_Aff2` consumer not in `inside`."""
    if isinstance(d, _Aff) and d.used == full(d.stride) and not inside:
        for lo, hi in isect(pivs, ((d.base + d.m_lo * d.stride, d.base + d.m_hi * d.stride),)):
            budget[0] -= hi - lo
            if budget[0] < 0:
                raise BoundaryLimit("boundary_too_large", "more boundary leaves than Limits.max_intervals")
            yield from range(lo, hi)
        return
    for lo, hi in pivs:
        budget[0] -= hi - lo
        if budget[0] < 0:
            raise BoundaryLimit("boundary_too_large", "more boundary leaves than Limits.max_intervals")
        for leaf in range(lo, hi):
            q = d.reader(leaf)
            if q is not None and (not inside or q not in inside):
                yield leaf


def _ret_consumed(scope: Scope, pos: int) -> bool:
    """Whether return position `pos` of the activation `scope` is read by some gate of an enclosing body or is a
    designated output (root).  Walks one level up per re-return.  Raises `BoundaryLimit` when undetermined."""
    while True:
        parent = scope.parent
        if parent is None:
            return True
        k, m = scope.node_idx, scope.member
        node = parent.spec.body.nodes[k]
        if node.form == "call":
            key, leaf = ("n", k), pos
        elif node.form == "batch":
            key, leaf = ("n", k), m * node.fn.out_leaves + pos
        else:
            c, y = _scan_shape(node)
            if pos < c:
                key, leaf = (("s", k), m * c + pos) if m < node.n - 1 else (("n", k), pos)
            else:
                key, leaf = ("n", k), c + m * y + (pos - c)
        ix = consumer_index(parent.spec)
        if ix.incomplete:
            raise BoundaryLimit("index_incomplete", ix.incomplete)
        if ix.is_read(key, leaf):
            return True
        positions = ix.ret.positions(key, leaf)
        if not positions:
            return False
        if len(positions) == 1:
            scope, pos = parent, positions[0]
            continue
        return any(_ret_consumed(parent, q) for q in positions)


def _out_leaves(part: Part, exact: bool) -> set[tuple[Key, int]]:
    """Canonical producer leaves of the part that are out (exact) or a superset of them (bound)."""
    _check_part(part)
    scope = part.scope
    spec = scope.spec
    body = spec.body
    ix = consumer_index(spec)
    if ix.incomplete:
        raise BoundaryLimit("index_incomplete", ix.incomplete)
    by_node = _by_node(part)
    budget = [Limits.max_intervals]
    out: set[tuple[Key, int]] = set()
    root = _is_root(scope) if exact else scope.parent is None  # the bound charges every returned leaf anyway

    def take(key: Key, ivs: Intervals) -> None:
        for lo, hi in ivs:
            budget[0] -= hi - lo
            if budget[0] < 0:
                raise BoundaryLimit("boundary_too_large", "more boundary leaves than Limits.max_intervals")
            out.update((key, leaf) for leaf in range(lo, hi))

    producers: dict[Key, Intervals] = {}
    for j, ms in by_node.items():
        node = body.nodes[j]
        producers[("n", j)] = _members_leaves(node, ms)
        if node.form == "scan":
            s = _carry_leaves(node, ms)
            if s:
                producers[("s", j)] = s
    for key, pivs in producers.items():
        if not pivs:
            continue
        for jp, d in ix.consumers(key):
            inside = by_node.get(jp)
            if inside is not None and len(inside) == n_members(body.nodes[jp]):
                continue  # every item of the consumer is inside the part
            if isinstance(d, _All):
                take(key, isect(d.ivs, pivs))
            elif isinstance(d, _One):
                if not inside or d.m not in inside:
                    take(key, isect(d.ivs, pivs))
            elif isinstance(d, (_Aff, _Aff2)):
                out.update((key, leaf) for leaf in _aff_select(d, pivs, inside, budget))
            else:  # _Lazy
                if exact and (inside or not d.exact):
                    raise BoundaryLimit("unresolved_consumer", f"{spec.id}: node {jp} reads {key} with unresolved members")
                take(key, isect(d.union, pivs))
    # returned leaves sourced in the part
    for plo, phi, sp, idx, base in ix.ret.pieces:
        if sp == "p":
            continue
        pivs = producers.get((sp, idx))
        if not pivs:
            continue
        for lo, hi in isect(((base, base + phi - plo),), pivs):
            budget[0] -= hi - lo
            if budget[0] < 0:
                raise BoundaryLimit("boundary_too_large", "more boundary leaves than Limits.max_intervals")
            for leaf in range(lo, hi):
                if root or not exact or _ret_consumed(scope, plo + (leaf - base)):
                    out.add(((sp, idx), leaf))
    return out


def _gates_of(part: Part, leaves: set[tuple[Key, int]]) -> dict[int, PrimitiveDefinition]:
    gates: dict[int, PrimitiveDefinition] = {}
    for key, leaf in leaves:
        g, prim = _gate_of(part.scope, key, leaf)
        gates[g] = prim
    return gates


def out_gates(part: Part) -> frozenset[int] | Unsupported:
    """EXACT out(part) as root gate indices: gates of the part consumed by a gate outside it or designated outputs.
    `Unsupported` when the compact index cannot resolve a consumer member-exactly or the parent-consumption walk of a
    returned leaf exceeds the limits; never the bound."""
    try:
        return frozenset(_gates_of(part, _out_leaves(part, True)))
    except BoundaryLimit as e:
        return e.unsupported()


def in_gates(part: Part) -> frozenset[int] | Unsupported:
    """EXACT in(part): distinct gates outside the part with a wire into it (Input members read nothing)."""
    try:
        _check_part(part)
        scope = part.scope
        _is_root(scope)  # a parameterised body without its parent cannot resolve its parameter wires
        spec = scope.spec
        body = spec.body
        ix = consumer_index(spec)
        if ix.incomplete:
            raise BoundaryLimit("index_incomplete", ix.incomplete)
        by_node = _by_node(part)
        budget = Limits.max_intervals
        gates: set[int] = set()
        for j, m in part.items:
            for (sp, idx), ivs in _member_reads(spec, j, m).items():
                for lo, hi in ivs:
                    budget -= hi - lo
                    if budget < 0:
                        raise BoundaryLimit("boundary_too_large", "more boundary leaves than Limits.max_intervals")
                    if sp == "p":
                        gates.update(resolve_gate(scope, ("p", idx, leaf))[0] for leaf in range(lo, hi))
                        continue
                    node = body.nodes[idx]
                    inside = by_node.get(idx)
                    if sp == "n":
                        for leaf in range(lo, hi):
                            if not inside or _leaf_member(node, leaf) not in inside:
                                gates.add(resolve_gate(scope, ("n", idx, leaf))[0])
                    else:  # "s": carry leaf m'*c + l produced inside iteration m' of scan `idx`
                        c, _ = _scan_shape(node)
                        for leaf in range(lo, hi):
                            mm, l = divmod(leaf, c)
                            if not inside or mm not in inside:
                                gates.add(resolve_gate(scope.child(idx, mm), node.fn.body.ret[l])[0])
        return frozenset(gates)
    except BoundaryLimit as e:
        return e.unsupported()


def w_out(part: Part) -> int | Unsupported:
    """Exact `sum(w(g) for g in out_gates(part))`."""
    try:
        return sum(prim.ret.w for prim in _gates_of(part, _out_leaves(part, True)).values())
    except BoundaryLimit as e:
        return e.unsupported()


def _declared_bound(part: Part) -> int:
    """Sum of the declared output widths of the items: every gate leaving a member leaves through the member's
    outputs, so this bounds w_out(part) for any part (type widths; no gate is resolved)."""
    body = part.scope.spec.body
    total = 0
    for j, m in part.items:
        node = body.nodes[j]
        if node.form in ("primitive", "call"):
            total += _sum_widths(node.out)
        elif node.form == "batch":
            total += _sum_widths(node.out.elem)
        else:
            total += _sum_widths(node.out.fields[0]) + _sum_widths(node.out.fields[1].elem)
    return total


def w_out_bound(part: Part) -> int:
    """SOUND UPPER BOUND on `w_out(part)`, cheap, never `Unsupported`: the consumer term with unresolved consumer
    members over-approximated by their union, every returned leaf sourced in the part charged (the parent may not
    read it), and on any internal limit the sum of the items' declared output widths."""
    try:
        return sum(prim.ret.w for prim in _gates_of(part, _out_leaves(part, False)).values())
    except BoundaryLimit:
        return _declared_bound(part)


@dataclass(frozen=True)
class BoundaryGate:
    """One out gate of a part, for debugging: `node`/`member` locate the item, `leaf` the body-level output leaf of
    the node (`kind == "out"`) or the carry leaf `l` of scan iteration `member` (`kind == "carry"`)."""
    gate: int
    node: int
    member: int
    leaf: int
    width: int
    kind: str
    prim: str


def explain(part: Part) -> list[BoundaryGate] | Unsupported:
    """The exact out gates of `part` with (node, member, leaf, width, primitive), ascending by gate."""
    try:
        leaves = _out_leaves(part, True)
    except BoundaryLimit as e:
        return e.unsupported()
    body = part.scope.spec.body
    seen: dict[int, BoundaryGate] = {}
    for (sp, idx), leaf in leaves:
        g, prim = _gate_of(part.scope, (sp, idx), leaf)
        node = body.nodes[idx]
        if sp == "s":
            c, _ = _scan_shape(node)
            m, l = divmod(leaf, c)
            bg = BoundaryGate(g, idx, m, l, prim.ret.w, "carry", prim.id)
        else:
            bg = BoundaryGate(g, idx, _leaf_member(node, leaf), leaf, prim.ret.w, "out", prim.id)
        seen.setdefault(g, bg)
    return [seen[g] for g in sorted(seen)]


__all__ = ["BoundaryLimit", "ConsumerIndex", "BoundaryGate", "used_param_leaves", "used_param_intervals",
           "consumer_index", "out_gates", "in_gates", "w_out", "w_out_bound", "gate_width", "explain"]
