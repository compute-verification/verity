"""a_out, a_in, |𝒥| and s_× of a circuit's counted operations, exactly, and which of them a partition computes whole.

The circuit is a Program in the `verity-ir/descriptor/v1` format (anything with a lowered ``root``: `Program`, or
`codec.decode_program`'s result), read through its canonical layout (PROTOCOL.md §6), the format NCI's check reads gate by
gate and `Nci.Cover.Circuit` formalizes. A served layer is 10^9 to 10^12 Boolean gates, so nothing is flattened except one
counted operation's Definition (`ops.OpShape`): every quantity is summarized once per Definition and context and mapped
through the call hierarchy, which equals the gate-by-gate definitions `flat` computes on small circuits (the tests hold
them equal).

* |𝒥| (`products`): the counted products, lane k of each counted activation (`ops.Site`).
* s_× (`s_x`): the fewest gates in any counted product's subcircuit m(p).
* a_out: the most counted products any one activation value feeds. A value is a gate (one bit): product p feeds v when v
  is one of the gates bound to p's activation operand x[k]. Constant gates are kept apart, since a zero bit padding many
  words is no activation: `constant_leaves` counts, per constant, the activation operand leaves bound to it.
* a_in: the fewest counted products summed into any one output: over every returned leaf of an activation of an entry
  Definition (`Designation.entries`, a GEMM coordinate), the counted products inside that activation with a gate of
  m(p) from which the leaf is reached, least over the leaves some product reaches.
* `whole`: given a partition of the gates into blocks, the products whose m(p) lies wholly inside one block.

`Ambiguous` refuses the one case the summaries can't count exactly: a Definition that binds one value to two activation
leaves of the products below it (above the op, where those leaves might be one product's), which `flat` measures instead.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field

from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.layout import Circuit

from .ops import (
    Designation,
    OpShape,
    Site,
    _bits,
    ids,
    is_constant,
    matches,
    member_refs,
    op_shape,
    param_bases,
)


class Ambiguous(ValueError):
    """One value is bound to two activation leaves a Definition's products read: counting its uses by leaf could count
    one product twice."""


#: a value inside a Definition: a gate of it (int >= 0), its flat parameter leaf j (-1 - j), or a constant (its id)
Key = int | str


@dataclass
class Summary:
    count: int = 0
    uses: dict = field(default_factory=dict)
    const: Counter = field(default_factory=Counter)
    ops: Counter = field(default_factory=Counter)
    excluded: Counter = field(default_factory=Counter)
    s_x: int | None = None
    s_x_max: int | None = None
    entries: dict = field(default_factory=dict)
    lanes: list | None = None

    @property
    def empty(self) -> bool:
        return not (self.count or self.excluded or self.entries)


@dataclass(frozen=True)
class EntryRow:
    """One entry Definition (in one context): its counted products, and over its returned leaves the least and most
    counted products reaching one, with the leaves none reaches."""

    id: str
    activations: int
    products: int
    outputs: int
    reached_min: int | None
    reached_max: int | None
    unreached: int


@dataclass(frozen=True)
class Measurement:
    gates: int
    products: int
    s_x: int | None
    s_x_max: int | None
    a_out: int
    a_out_values: int
    a_out_gate: int | None
    activation_values: int
    constant_leaves: dict
    a_in: int | None
    entries: tuple[EntryRow, ...]
    ops: dict
    excluded: dict

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["entries"] = [e.__dict__ for e in self.entries]
        return out


@dataclass(frozen=True)
class Whole:
    """Products whose m(p) lies inside one block (`per_block`), those split across blocks or with a gate in none."""

    per_block: dict
    split: int
    split_examples: tuple[tuple[int, int], ...]

    @property
    def whole(self) -> int:
        return sum(self.per_block.values())


class Measurer:
    """Summaries of one program under one designation, memoized per Definition and context."""

    def __init__(self, program, designation: Designation) -> None:
        self.root: SpecializedDefinition = program.root
        self.root_ids = ids(getattr(program, "program_fn_id", None) or self.root)
        self.d = designation
        self.sites: tuple[Site, ...] = designation.sites
        self._sum: dict = {}
        self._op: dict = {}
        self._canon: dict = {}
        self._live: dict = {}
        self._bases: dict = {}
        self._take: dict = {}
        self.const_exact = True
        self.ctx0 = self.enter(self.root, tuple(s.within is None for s in self.sites))

    # -- matching ---------------------------------------------------------------------------------------------------
    def match(self, rx: re.Pattern | None, fn) -> bool:
        if fn is not self.root:
            return matches(rx, fn)
        return rx is not None and any(rx.fullmatch(i) for i in self.root_ids)

    def enter(self, fn, ctx: tuple) -> tuple:
        return tuple(c or self.match(s.within_rx, fn) for s, c in zip(self.sites, ctx))

    def site_of(self, fn, ctx: tuple) -> tuple[Site | None, bool]:
        matched = False
        for s, c in zip(self.sites, ctx):
            if self.match(s.op_rx, fn):
                if c:
                    return s, True
                matched = True
        return None, matched

    def bases(self, fn) -> list[int]:
        b = self._bases.get(id(fn))
        if b is None:
            b = self._bases[id(fn)] = param_bases(fn)
        return b

    # -- values -----------------------------------------------------------------------------------------------------
    def src(self, P, ref) -> Key:
        s, i, j = ref
        if s == "p":
            return -1 - (self.bases(P)[i] + j)
        return self.node_src(P, i, j)

    def canon(self, F, j: int) -> Key:
        memo = self._canon.get(id(F))
        if memo is None:
            memo = self._canon[id(F)] = {}
        k = memo.get(j)
        if k is None:
            k = memo[j] = self.src(F, F.body.ret[j])
        return k

    def node_src(self, P, k: int, j: int) -> Key:
        node = P.body.nodes[k]
        f, off = node.fn, P.body.offsets[k]
        if isinstance(f, PrimitiveDefinition):
            return f.id if is_constant(f) else off + (j if node.form == "batch" else 0)
        if node.form == "call":
            return self.reloc(P, k, 0, self.canon(f, j))
        if node.form == "batch":
            m, jj = divmod(j, f.out_leaves)
            return self.reloc(P, k, m, self.canon(f, jj))
        c = node.out.fields[0].leaves
        if j < c:
            m, jj = node.n - 1, j
        else:
            m, r = divmod(j - c, node.out.fields[1].elem.leaves)
            jj = c + r
        return self.reloc(P, k, m, self.canon(f, jj))

    def reloc(self, P, k: int, m: int, key: Key) -> Key:
        """A value of member ``m`` of node ``k`` of ``P``, as a value of ``P``."""
        if isinstance(key, str):
            return key
        node = P.body.nodes[k]
        if key >= 0:
            return P.body.offsets[k] + m * node.fn.gates + key
        if node.form == "scan":
            carry = node.fn.signature()[0][0][1].leaves
            while key < 0 and -1 - key < carry and m > 0:
                key, m = self.canon(node.fn, -1 - key), m - 1
            if isinstance(key, str):
                return key
            if key >= 0:
                return P.body.offsets[k] + m * node.fn.gates + key
        b = self.bases(node.fn)
        jflat = -1 - key
        i = bisect_right(b, jflat) - 1
        return self.src(P, self.member_refs(node, m, i)[jflat - b[i]])

    def member_refs(self, node, m: int, i: int):
        if node.form == "call" or node.form == "primitive":
            return node.args[i].refs
        if node.form == "batch" and node.axes[i] not in (None, 0):
            key = (id(node), i)
            hit = self._take.get(key)
            if hit is None or hit[0] != m:
                hit = self._take[key] = (m, member_refs(node, m, i))
            return hit[1]
        return member_refs(node, m, i)

    def invariant(self, node, key: Key) -> bool:
        """``key`` of a member is the same value of the parent for every member."""
        if isinstance(key, str):
            return True
        if key >= 0:
            return False
        i = bisect_right(self.bases(node.fn), -1 - key) - 1
        if node.form == "batch":
            return node.axes[i] is None
        if node.form == "scan":
            return i > node.n_xs
        return True

    # -- |𝒥|, s_×, a_out ----------------------------------------------------------------------------------------------
    def op_summary(self, fn, site: Site) -> Summary:
        hit = self._op.get((id(fn), id(site)))
        if hit is not None:
            return hit
        shape = op_shape(fn, site)
        base = self.bases(fn)[shape.xi]
        S = Summary(count=shape.lanes, s_x=min(shape.sizes), s_x_max=max(shape.sizes))
        S.ops[fn.id] = shape.lanes
        S.lanes = [tuple(-1 - (base + j) for j in xs) for xs in shape.x_leaves]
        self._op[(id(fn), id(site))] = S
        return S

    def summary(self, fn, ctx: tuple) -> Summary:
        key = (id(fn), ctx)
        hit = self._sum.get(key)
        if hit is not None:
            return hit
        S = Summary()
        if self.match(self.d.entries_rx, fn):
            S.entries[(fn.id, ctx)] = (fn, ctx)
        for k, node in enumerate(fn.body.nodes):
            child = self.child(node, ctx)
            if child is None:
                _site, matched = self.site_of(node.fn, ctx)
                if matched:
                    S.excluded[node.fn.id] += node.n if node.form in ("batch", "scan") else 1
                continue
            if not child.empty:
                self.merge(S, fn, k, node, child)
        self._sum[key] = S
        return S

    def child(self, node, ctx: tuple) -> Summary | None:
        site, matched = self.site_of(node.fn, ctx)
        if site is not None:
            return self.op_summary(node.fn, site)
        if matched or isinstance(node.fn, PrimitiveDefinition):
            return None
        return self.summary(node.fn, self.enter(node.fn, ctx))

    def merge(self, S: Summary, P, k: int, node, child: Summary) -> None:
        n = node.n if node.form in ("batch", "scan") else 1
        S.count += child.count * n
        for c, add in ((S.ops, child.ops), (S.excluded, child.excluded)):
            for o, v in add.items():
                c[o] += v * n
        S.entries.update(child.entries)
        if child.s_x is not None:
            S.s_x = child.s_x if S.s_x is None else min(S.s_x, child.s_x)
            S.s_x_max = child.s_x_max if S.s_x_max is None else max(S.s_x_max, child.s_x_max)
        keys = [ck for lane in child.lanes for ck in lane] if child.lanes is not None else list(child.uses)
        shared = n == 1 or all(self.invariant(node, ck) for ck in keys)
        members, mult = ((0,), n) if shared else (range(n), 1)
        uses = S.uses
        for m in members:
            if child.lanes is not None:
                for lane in child.lanes:
                    pks = [self.reloc(P, k, m, ck) for ck in lane]
                    for pk in pks:
                        if isinstance(pk, str):
                            S.const[pk] += mult
                    for pk in {pk for pk in pks if not isinstance(pk, str)}:
                        uses[pk] = uses.get(pk, 0) + mult
                continue
            seen: dict = {}
            for ck, c in child.uses.items():
                pk = self.reloc(P, k, m, ck)
                if isinstance(pk, str):
                    S.const[pk] += c * mult
                    continue
                if pk in seen:
                    raise Ambiguous(f"{node.fn.id} (node {k} of {P.id}) is passed one value for two activation leaves "
                                    f"of the products below it; measure it with flat.properties")
                seen[pk] = ck
                uses[pk] = uses.get(pk, 0) + c * mult
            for cid, c in child.const.items():
                S.const[cid] += c * mult

    # -- a_in -------------------------------------------------------------------------------------------------------
    def live(self, F, mask: int, ctx: tuple) -> tuple[int, int]:
        """(counted products in ``F`` with a gate of m(p) reaching a returned leaf in ``mask``, the flat parameter
        leaves those leaves read)."""
        key = (id(F), ctx, mask)
        hit = self._live.get(key)
        if hit is not None:
            return hit
        body, base = F.body, self.bases(F)
        need_n = [0] * len(body.nodes)
        pneed = 0

        def mark(refs, leaves) -> None:
            nonlocal pneed
            for j in leaves:
                s, i, jj = refs[j]
                if s == "p":
                    pneed |= 1 << (base[i] + jj)
                else:
                    need_n[i] |= 1 << jj

        mark(body.ret, _bits(mask))
        cnt = 0
        for k in range(len(body.nodes) - 1, -1, -1):
            m = need_n[k]
            if not m:
                continue
            node = body.nodes[k]
            f = node.fn
            fb = self.bases(f)
            if node.form in ("primitive", "call"):
                c, need = self.live_member(f, m, ctx)
                cnt += c
                self.mark_member(node, 0, need, fb, mark)
            elif node.form == "batch":
                L = f.out_leaves
                subs: dict[int, int] = {}
                for b in _bits(m):
                    q, r = divmod(b, L)
                    subs[q] = subs.get(q, 0) | (1 << r)
                for q, sub in subs.items():
                    c, need = self.live_member(f, sub, ctx)
                    cnt += c
                    self.mark_member(node, q, need, fb, mark)
            else:
                cl = node.out.fields[0].leaves
                y = node.out.fields[1].elem.leaves
                carry, ys = m & ((1 << cl) - 1), m >> cl
                for it in range(node.n - 1, -1, -1):
                    sub = carry | (((ys >> (it * y)) & ((1 << y) - 1)) << cl)
                    if not sub:
                        carry = 0
                        continue
                    c, need = self.live_member(f, sub, ctx)
                    cnt += c
                    carry = need & ((1 << fb[1]) - 1)
                    self.mark_member(node, it, need & ~((1 << fb[1]) - 1), fb, mark)
                mark(node.args[0].refs, _bits(carry))
        self._live[key] = (cnt, pneed)
        return cnt, pneed

    def live_member(self, f, mask: int, ctx: tuple) -> tuple[int, int]:
        site, _matched = self.site_of(f, ctx)
        if site is not None:
            shape: OpShape = op_shape(f, site)
            return shape.reaching(mask), shape.need(mask)
        if isinstance(f, PrimitiveDefinition):
            return 0, (1 << self.bases(f)[-1]) - 1
        return self.live(f, mask, self.enter(f, ctx))

    def mark_member(self, node, m: int, need: int, fb: list[int], mark) -> None:
        by_param: dict[int, list[int]] = {}
        for b in _bits(need):
            i = bisect_right(fb, b) - 1
            by_param.setdefault(i, []).append(b - fb[i])
        for i, leaves in by_param.items():
            mark(self.member_refs(node, m, i), leaves)

    def entry_rows(self) -> list[EntryRow]:
        rows = []
        S = self.summary(self.root, self.ctx0)
        acts = self.activations()
        for (fid, ctx), (E, _ctx) in sorted(S.entries.items(), key=lambda kv: kv[0][0]):
            counts = [self.live(E, 1 << ell, ctx)[0] for ell in range(E.out_leaves)]
            hit = [c for c in counts if c]
            rows.append(EntryRow(fid, acts.get((fid, ctx), 0), self.summary(E, ctx).count, len(counts),
                                 min(hit) if hit else None, max(hit) if hit else None, len(counts) - len(hit)))
        return rows

    def activations(self) -> Counter:
        """Activations of each entry Definition (in its context) in the circuit."""
        memo: dict = {}

        def count(fn, ctx) -> Counter:
            key = (id(fn), ctx)
            if key in memo:
                return memo[key]
            out: Counter = Counter()
            if self.match(self.d.entries_rx, fn):
                out[(fn.id, ctx)] += 1
            for node in fn.body.nodes:
                child = self.child(node, ctx)
                if child is None or child.lanes is not None or not child.entries:
                    continue
                n = node.n if node.form in ("batch", "scan") else 1
                for e, c in count(node.fn, self.enter(node.fn, ctx)).items():
                    out[e] += c * n
            memo[key] = out
            return out

        return count(self.root, self.ctx0)

    # -- the result -------------------------------------------------------------------------------------------------
    def measure(self) -> Measurement:
        S = self.summary(self.root, self.ctx0)
        a_out = max(S.uses.values(), default=0)
        tops = [g for g, c in S.uses.items() if c == a_out] if a_out else []
        rows = self.entry_rows()
        hit = [r.reached_min for r in rows if r.reached_min is not None]
        return Measurement(
            gates=self.root.gates, products=S.count, s_x=S.s_x, s_x_max=S.s_x_max, a_out=a_out, a_out_values=len(tops),
            a_out_gate=min(tops) if tops else None, activation_values=len(S.uses),
            constant_leaves={k: v for k, v in sorted(S.const.items())}, a_in=min(hit) if hit else None,
            entries=tuple(rows), ops=dict(sorted(S.ops.items())), excluded=dict(sorted(S.excluded.items())))


def measure(program, designation: Designation) -> Measurement:
    """|𝒥|, s_×, a_out and a_in of ``program``'s counted operations under ``designation``, exactly."""
    return Measurer(program, designation).measure()


# -- partitions ------------------------------------------------------------------------------------------------------
_MIXED = object()


class Blocks:
    """A partition of a circuit's gates into blocks, as runs: gates from ``starts[i]`` up to the next start belong to
    block ``ids[i]`` (None: no block, the input gates)."""

    def __init__(self, runs) -> None:
        starts: list[int] = []
        ids: list = []
        for s, b in sorted(runs, key=lambda r: r[0]):
            if ids and ids[-1] == b:
                continue
            starts.append(s)
            ids.append(b)
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
            ids.insert(0, None)
        self.starts, self.ids = starts, ids

    @classmethod
    def from_owner(cls, owner) -> Blocks:
        """From each gate's block (None for a gate in no block)."""
        runs, prev = [], _MIXED
        for g, b in enumerate(owner):
            if g == 0 or b != prev:
                runs.append((g, b))
                prev = b
        return cls(runs)

    @classmethod
    def by_root(cls, program) -> Blocks:
        """One block per activation of the root's body, named by its annotation (else its index)."""
        body = program.root.body
        return cls([(body.offsets[k], node.name or k) for k, node in enumerate(body.nodes)])

    def of(self, g: int):
        return self.ids[bisect_right(self.starts, g) - 1]

    def span(self, lo: int, hi: int):
        i = bisect_right(self.starts, lo) - 1
        return self.ids[i] if i == bisect_right(self.starts, hi - 1) - 1 else _MIXED


def whole(program, designation: Designation, blocks: Blocks, examples: int = 16) -> Whole:
    """Which counted products a partition of ``program``'s gates computes wholly inside one block."""
    M = Measurer(program, designation)
    per_block: Counter = Counter()
    split = [0]
    ex: list[tuple[int, int]] = []

    def add(b, c: int) -> None:
        if b is None:
            split[0] += c
        else:
            per_block[b] += c

    def op(shape: OpShape, off: int) -> None:
        b = blocks.span(off, off + shape.gates)
        if b is not _MIXED:
            add(b, shape.lanes)
            return
        for k, gates in enumerate(shape.members):
            b = blocks.span(off + gates[0], off + gates[-1] + 1)
            if b is _MIXED:
                bs = {blocks.of(off + g) for g in gates}
                b = bs.pop() if len(bs) == 1 else None
            add(b, 1)
            if b is None and len(ex) < examples:
                ex.append((off + shape.rep[k], k))

    def walk(F, off: int, ctx: tuple) -> None:
        b = blocks.span(off, off + F.gates)
        if b is not _MIXED:
            add(b, M.summary(F, ctx).count)
            return
        for k, node in enumerate(F.body.nodes):
            child = M.child(node, ctx)
            if child is None or child.count == 0:
                continue
            f = node.fn
            n = node.n if node.form in ("batch", "scan") else 1
            noff = off + F.body.offsets[k]
            b = blocks.span(noff, noff + n * f.gates)
            if b is not _MIXED:
                add(b, child.count * n)
                continue
            site, _m = M.site_of(f, ctx)
            for m in range(n):
                moff = noff + m * f.gates
                if site is not None:
                    op(op_shape(f, site), moff)
                else:
                    walk(f, moff, M.enter(f, ctx))

    walk(M.root, 0, M.ctx0)
    return Whole(dict(per_block), split[0], tuple(ex))


def describe(program, gate: int) -> str:
    """Where gate ``gate`` is: the chain of activations down to it."""
    g = Circuit(program.root).gate(gate)
    return " / ".join(g.path)
