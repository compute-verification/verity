"""`upstream.required` on circuits too large to list gate by gate: a region that is a union of whole activations, and the
size of the largest certified G in it, computed along the call hierarchy.

`upstream.required(reads, A, fin_into)` is the set of gates of A that reach, inside A, a gate of `into F ∩ A` for every
finishing set F. Number the distinct `into F ∩ A` as classes; a gate's **class mask** is the set of classes it reaches
inside A; G is the gates whose mask holds every class. When A is a disjoint union of whole activations (`layout.Scope`s),
a path inside A leaves an activation only through a gate its returns name (a value leaves an activation only as a
return, and a gate outside it that reads one comes later than all of its gates, so the path cannot come back). So a gate's
mask is the union of the masks of the returned gates it reaches inside its own activation, and an activation's
parameters pull back the masks of whatever reads them. That recursion runs down each Definition's body:

* `Pull.pull(fn, masks, can_full)`: given the class mask at each of fn's returned leaves, the number of fn's gates whose
  mask (over fn's own paths, then the given masks) holds every class, and the mask each parameter leaf pulls back;
* a Definition small enough is swept gate by gate (`ops.flatten`), and a larger one node by node, its members pulled in
  turn from the last;
* results are memoized on (Definition, masks), the masks first rewritten over their atoms (the classes no mask tells
  apart are merged, and numbered in order of first appearance), so that the members of a batch that see the same pattern
  of classes are computed once.

`required_size(circuit_root, acts, classes)` does it for a region: `acts` the activations whose gates make up A, `classes`
the distinct `into F ∩ A`. It agrees with `upstream.required` on the explicit wiring (`tests/test_regions.py`).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import Scope, resolve_gate

from .ops import flatten, member_refs, param_bases

#: Definitions with at most this many gates are swept gate by gate
FLAT_MAX = 200_000


def _bits(m: int):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def canonical(masks: Sequence[int]) -> tuple[tuple[int, ...], list[int]]:
    """``masks`` over their atoms, and the atoms: the classes grouped by which of the distinct masks hold them, the atoms
    numbered by that set (the distinct masks numbered in order of first appearance), so that two lists of masks that
    differ only in what the classes are called get one canonical form."""
    distinct = list(dict.fromkeys(m for m in masks if m))
    sig: dict[int, list[int]] = {}
    for d, m in enumerate(distinct):
        for b in _bits(m):
            sig.setdefault(b, []).append(d)
    groups: dict[tuple[int, ...], int] = {}
    for b, s in sig.items():
        t = tuple(s)
        groups[t] = groups.get(t, 0) | (1 << b)
    order = sorted(groups)
    atoms = [groups[t] for t in order]
    over = [0] * len(distinct)
    for a, t in enumerate(order):
        for d in t:
            over[d] |= 1 << a
    index = {m: over[d] for d, m in enumerate(distinct)}
    return tuple(index[m] if m else 0 for m in masks), atoms


def _back(m: int, atoms: Sequence[int]) -> int:
    out = 0
    for a in _bits(m):
        out |= atoms[a]
    return out


class Pull:
    """Memoized `pull` (the module docstring)."""

    def __init__(self, flat_max: int = FLAT_MAX) -> None:
        self.flat_max = flat_max
        self.memo: dict[tuple, tuple[int, tuple[int, ...]]] = {}
        self._keep: dict[int, object] = {}

    def pull(self, fn, masks: Sequence[int], can_full: bool) -> tuple[int, tuple[int, ...]]:
        """(gates of fn whose class mask is the union of ``masks`` when ``can_full`` (else 0), each parameter leaf's mask)."""
        n_params = param_bases(fn)[-1]
        if not any(masks):
            return 0, (0,) * n_params
        canon, atoms = canonical(masks)
        key = (id(fn), canon, can_full)
        hit = self.memo.get(key)
        if hit is None:
            self._keep[id(fn)] = fn
            hit = self._pull(fn, canon, (1 << len(atoms)) - 1, can_full)
            self.memo[key] = hit
        count, back = hit
        cache: dict[int, int] = {0: 0}
        out = []
        for m in back:
            v = cache.get(m)
            if v is None:
                v = cache[m] = _back(m, atoms)
            out.append(v)
        return count, tuple(out)

    def _pull(self, fn, masks: tuple[int, ...], full: int, can_full: bool) -> tuple[int, tuple[int, ...]]:
        base = param_bases(fn)
        preach = [0] * base[-1]
        if isinstance(fn, PrimitiveDefinition):
            m = masks[0]
            return int(can_full and m == full), (m,) * base[-1]
        if fn.gates <= self.flat_max:
            flat = flatten(fn)
            cm = [0] * len(flat.operands)
            for ell, s in enumerate(flat.ret):
                if s >= 0:
                    cm[s] |= masks[ell]
                else:
                    preach[-1 - s] |= masks[ell]
            count = 0
            for g in range(len(cm) - 1, -1, -1):
                m = cm[g]
                if not m:
                    continue
                if m == full:
                    count += 1
                for o in flat.operands[g]:
                    if o >= 0:
                        cm[o] |= m
                    else:
                        preach[-1 - o] |= m
            return (count if can_full else 0), tuple(preach)
        body = fn.body
        reach: dict[tuple[int, int], int] = {}

        def add(ref, m: int) -> None:
            space, i, j = ref
            if space == "p":
                preach[base[i] + j] |= m
            else:
                reach[(i, j)] = reach.get((i, j), 0) | m

        for ell, r in enumerate(body.ret):
            if masks[ell]:
                add(r, masks[ell])
        count = 0
        for k in range(len(body.nodes) - 1, -1, -1):
            node = body.nodes[k]
            f = node.fn
            fb = param_bases(f)
            L = f.out_leaves
            if node.form in ("primitive", "call", "batch"):
                n = node.n if node.form == "batch" else 1
                for m in range(n - 1, -1, -1):
                    outs = tuple(reach.pop((k, m * L + j), 0) for j in range(L))
                    if not any(outs):
                        continue
                    c, pr = self._member(f, outs, full, can_full)
                    count += c
                    for i in range(len(fb) - 1):
                        refs = member_refs(node, m, i) if node.form == "batch" else node.args[i].refs
                        for leaf in range(fb[i + 1] - fb[i]):
                            if pr[fb[i] + leaf]:
                                add(refs[leaf], pr[fb[i] + leaf])
                continue
            # a scan: member m's parameters are (carry, xs..., shared...), its returns (carry, y); member m's returned carry is
            # member m + 1's carry parameter, the last member's the node's
            c0 = node.out.fields[0].leaves
            y = node.out.fields[1].elem.leaves
            carry = tuple(reach.pop((k, j), 0) for j in range(c0))
            for m in range(node.n - 1, -1, -1):
                outs = carry + tuple(reach.pop((k, c0 + m * y + r), 0) for r in range(y))
                if not any(outs):
                    carry = (0,) * c0
                    continue
                c, pr = self._member(f, outs, full, can_full)
                count += c
                for i in range(len(fb) - 1):
                    if i == 0 and m > 0:
                        continue
                    refs = member_refs(node, m, i)
                    for leaf in range(fb[i + 1] - fb[i]):
                        if pr[fb[i] + leaf]:
                            add(refs[leaf], pr[fb[i] + leaf])
                carry = tuple(pr[fb[0]:fb[1]])
        return count, tuple(preach)

    def _member(self, f, outs: tuple[int, ...], full: int, can_full: bool) -> tuple[int, tuple[int, ...]]:
        u = 0
        for m in outs:
            u |= m
        return self.pull(f, outs, can_full and u == full)


@dataclass(frozen=True)
class Required:
    """A region's largest certified G, by size: `gates` in A, `required` reaching every class, `absent` the classes with no
    gate in A (then nothing is required: `required` is 0)."""

    gates: int
    required: int
    classes: int
    absent: tuple[int, ...]


class Resolver:
    """`layout.resolve_gate` of activations' parameter leaves, memoized where a batch shares an argument (axis None)."""

    def __init__(self) -> None:
        self.memo: dict[tuple, int] = {}

    def param(self, scope: Scope, pi: int, leaf: int) -> int:
        parent = scope.parent
        node = parent.spec.body.nodes[scope.node_idx]
        shared = node.form == "call" or (node.form == "batch" and node.axes[pi] is None)
        key = (parent.offset, scope.node_idx, None if shared else scope.member, pi, leaf)
        hit = self.memo.get(key)
        if hit is None:
            hit = resolve_gate(scope, ("p", pi, leaf))[0]
            self.memo[key] = hit
        return hit


def activations(scope: Scope):
    """Every activation below ``scope``, each before the ones inside it."""
    for k, node in enumerate(scope.spec.body.nodes):
        for m in range(node.n if node.form in ("batch", "scan") else 1):
            s = scope.child(k, m)
            yield s
            if not isinstance(s.spec, PrimitiveDefinition):
                yield from activations(s)


def returns(scope: Scope) -> list[int]:
    """The root gate each returned leaf of the activation ``scope`` names."""
    fn = scope.spec
    if isinstance(fn, PrimitiveDefinition):
        return [scope.offset]
    return [resolve_gate(scope, r)[0] for r in fn.body.ret]


def required_size(acts: Sequence[Scope], classes: Sequence[frozenset[int]], pull: Pull | None = None,
                  resolver: Resolver | None = None) -> Required:
    """`len(upstream.required(reads, A, classes))` for A the gates of the disjoint activations ``acts``."""
    pull = pull or Pull()
    resolver = resolver or Resolver()
    acts = sorted(acts, key=lambda s: s.offset)
    lo = [s.offset for s in acts]
    hi = [s.offset + s.spec.gates for s in acts]
    for a in range(1, len(acts)):
        if lo[a] < hi[a - 1]:
            raise ValueError(f"the region's activations overlap: {acts[a - 1].path()} and {acts[a].path()}")

    def inside(g: int) -> bool:
        a = bisect_right(lo, g) - 1
        return a >= 0 and g < hi[a]

    rets = [returns(s) for s in acts]
    named = {g for a, rg in enumerate(rets) for g in rg if lo[a] <= g < hi[a]}
    reach: dict[int, int] = {}
    absent = []
    for k, T in enumerate(classes):
        hit = [g for g in T if inside(g)]
        if not hit:
            absent.append(k)
        for g in hit:
            if g not in named:
                raise ValueError(f"class {k}'s gate {g} is inside the region but no activation of it returns it")
            reach[g] = reach.get(g, 0) | (1 << k)
    total = sum(h - l for l, h in zip(lo, hi))
    if absent:
        return Required(total, 0, len(classes), tuple(absent))
    full = (1 << len(classes)) - 1
    count = 0
    for a in range(len(acts) - 1, -1, -1):
        s = acts[a]
        masks = [reach.get(g, 0) if lo[a] <= g < hi[a] else 0 for g in rets[a]]
        u = 0
        for m in masks:
            u |= m
        c, pr = pull.pull(s.spec, masks, u == full)
        count += c
        if a == 0:
            continue
        base = param_bases(s.spec)
        for pi in range(len(base) - 1):
            for leaf in range(base[pi + 1] - base[pi]):
                m = pr[base[pi] + leaf]
                if m:
                    g = resolver.param(s, pi, leaf)
                    if g < lo[a] and inside(g):
                        reach[g] = reach.get(g, 0) | m
    return Required(total, count, len(classes), ())
