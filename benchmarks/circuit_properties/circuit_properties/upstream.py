"""The upstream certificate, `Nci.Capability.Consumers.Upstream` (`Security/Definitions/Nci/Capability/Consumers.lean`),
decided on a named circuit, with the overlap weights it is used with.

Lean's objects, one for one:

* the wiring: `reads[g]`, the gates gate g reads, each earlier than g (`Verity.Circuit.reads`, `reads_lt`);
* `into(reads, S)`, `Verity.Circuit.into`: the gates outside S with a wire into S;
* `feeds_within(reads, R, g, t)`, `Nci.Covering.FeedsWithin`: a path of wires from g to t all of whose gates, its ends
  included, lie in R;
* `Consumers`, `Nci.Capability.Consumers`: consumer c's finishing sets `fins[c]` (a block finishes c by computing one of
  them whole), its upstream region `A[c]`, which holds no registered gate (`A_θ`), and its required gates `G[c] ⊆ A[c]`
  (`G_sub`). `of_uses` and `of_outputs` are `Consumers.ofUses` (use j finished by computing one of its counted products
  m(i, j) whole, `MulBlocks`) and `Consumers.ofOutputs` (output o finished by computing its gate);
* `upstream` decides `Consumers.Upstream`: for every consumer c, every F in fins c and every g in G c, some
  a ∈ into F ∩ A c with FeedsWithin (A c) g a. It returns every violation, not just the first;
* `overlap_weights` decides `Consumers.OverlapWeights ω`: ω ≥ 0, and over the regions holding any one gate the weights
  sum to at most 1.

Upstream reads a finishing set only through `into F`, so a consumer's finishing sets may be given by their sets of gates
or, when there are too many to list (an LM head's products), by `into F` with the number of finishing sets that have it
(`Consumers.fin_into`). `required` is the largest G the certificate holds for, given the finishing sets and the region:
the gates of the region that reach, inside it, a gate of `into F ∩ A` for every F.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

#: a circuit's wiring, as far as the check reads it: gate -> the gates it reads (each smaller)
Reads = Mapping[int, Sequence[int]]


class NotConsumers(ValueError):
    """The data is not a `Consumers` value: a required gate outside its region (`G_sub`), a region holding a registered gate
    (`A_θ`), or a finishing set reading a gate the wiring doesn't know."""


class NotMulBlocks(ValueError):
    """The products are not `Nci.Covering.MulBlocks`: overlapping, holding a source, or a gate not reaching the output."""


def check_wiring(reads: Reads) -> None:
    """`Circuit.reads_lt`: every gate reads only earlier gates (the order `reaching` sweeps in)."""
    for g, rs in reads.items():
        for a in rs:
            if not a < g:
                raise ValueError(f"gate {g} reads gate {a}, not an earlier one")


def into(reads: Reads, S: Iterable[int]) -> frozenset[int]:
    """`Circuit.into S`: the gates outside S with a wire into S."""
    S = S if isinstance(S, (set, frozenset)) else frozenset(S)
    try:
        return frozenset(a for g in S for a in reads[g] if a not in S)
    except KeyError as e:
        raise NotConsumers(f"the wiring has no reads for gate {e.args[0]}") from None


def feeds_within(reads: Reads, R: Iterable[int], g: int, t: int) -> bool:
    """`Nci.Covering.FeedsWithin R g t`, by its definition: `refl` (t ∈ R) and `step` (g ∈ R, g read by some h with
    FeedsWithin R h t), searched backward from t over the wires inside R."""
    R = R if isinstance(R, (set, frozenset)) else frozenset(R)
    if t not in R:
        return False
    seen, todo = {t}, [t]
    while todo:
        h = todo.pop()
        if h == g:
            return True
        for a in reads.get(h, ()):
            if a in R and a not in seen:
                seen.add(a)
                todo.append(a)
    return False


def reaching(reads: Reads, R: Iterable[int], targets: Sequence[Iterable[int]]) -> dict[int, int]:
    """For each gate g of R that reaches a target inside R, the bitmask of the target sets T_k with some a ∈ T_k ∩ R and
    FeedsWithin R g a. One sweep of R from its last gate down: a gate's mask is final once every gate reading it, all
    later, has passed it on."""
    R = R if isinstance(R, (set, frozenset)) else frozenset(R)
    mask: dict[int, int] = {}
    for k, T in enumerate(targets):
        for a in T:
            if a in R:
                mask[a] = mask.get(a, 0) | (1 << k)
    for h in sorted(R, reverse=True):
        m = mask.get(h)
        if m:
            for a in reads[h]:
                if a in R:
                    mask[a] = mask.get(a, 0) | m
    return mask


@dataclass(frozen=True)
class MulBlocks:
    """`Nci.Covering.MulBlocks` without its token computations: the counted products P ⊆ weights × uses, each p's gates m[p]
    and output gate o[p]."""

    P: tuple[tuple[int, int], ...]
    m: Mapping[tuple[int, int], frozenset[int]]
    o: Mapping[tuple[int, int], int]

    def check(self, reads: Reads) -> None:
        """`m_disjoint`, `m_computed` (no gate of m is a source), `o_mem` and `reach`, or `NotMulBlocks` naming the first
        that fails."""
        owner: dict[int, tuple[int, int]] = {}
        for p in self.P:
            m = self.m[p]
            for g in m:
                if g in owner:
                    raise NotMulBlocks(f"m_disjoint: gate {g} is in m{owner[g]} and m{p}")
                owner[g] = p
                if not reads.get(g):
                    raise NotMulBlocks(f"m_computed: gate {g} of m{p} reads nothing")
            if self.o[p] not in m:
                raise NotMulBlocks(f"o_mem: o{p} = {self.o[p]} is not in m{p}")
            mask = reaching(reads, m, [(self.o[p],)])
            if any(g not in mask for g in m):
                raise NotMulBlocks(f"reach: a gate of m{p} doesn't reach o{p} inside m{p}")


@dataclass(frozen=True)
class Consumers:
    """`Nci.Capability.Consumers`: for each consumer c, its finishing sets, region `A[c]` and required gates `G[c]`.

    `fins[c]` lists the finishing sets as sets of gates; `fin_into[c]`, when given instead, is `into F` of each, with how
    many finishing sets have it. Construction checks `G_sub` and `A_θ` against the registered gates `theta`."""

    A: tuple[frozenset[int], ...]
    G: tuple[frozenset[int], ...]
    theta: frozenset[int]
    fins: tuple[tuple[frozenset[int], ...], ...] | None = None
    fin_into: tuple[Counter, ...] | None = None
    names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        n = len(self.A)
        if len(self.G) != n or (self.fins is None) == (self.fin_into is None):
            raise NotConsumers("one region and one G a consumer, and its finishing sets given one way (fins or fin_into)")
        if len(self.fins if self.fins is not None else self.fin_into) != n:
            raise NotConsumers("one list of finishing sets a consumer")
        for c in range(n):
            if not self.G[c] <= self.A[c]:
                raise NotConsumers(f"G_sub: {self.name(c)}'s required gate {min(self.G[c] - self.A[c])} is outside its region")
            hit = self.A[c] & self.theta
            if hit:
                raise NotConsumers(f"A_θ: {self.name(c)}'s region holds the registered gate {min(hit)}")

    def __len__(self) -> int:
        return len(self.A)

    def name(self, c: int) -> str:
        return self.names[c] if c < len(self.names) else f"consumer {c}"

    def into_counts(self, reads: Reads, c: int) -> Counter:
        """`into F` of consumer c's finishing sets, with multiplicity."""
        if self.fin_into is not None:
            return self.fin_into[c]
        return Counter(into(reads, F) for F in self.fins[c])


def of_uses(B: MulBlocks, uses: Sequence[int], A: Sequence[frozenset[int]], G: Sequence[frozenset[int]],
            theta: frozenset[int], names: Sequence[str] = ()) -> Consumers:
    """`Consumers.ofUses`: use j (each of ``uses``) is finished by computing one of its counted products m(i, j) whole."""
    fins = tuple(tuple(B.m[p] for p in B.P if p[1] == j) for j in uses)
    return Consumers(A=tuple(A), G=tuple(G), theta=theta, fins=fins, names=tuple(names))


def of_outputs(out: Sequence[int], A: Sequence[frozenset[int]], G: Sequence[frozenset[int]], theta: frozenset[int],
               names: Sequence[str] = ()) -> Consumers:
    """`Consumers.ofOutputs`: output c is finished by computing its gate ``out[c]``."""
    return Consumers(A=tuple(A), G=tuple(G), theta=theta, fins=tuple((frozenset((o,)),) for o in out), names=tuple(names))


@dataclass(frozen=True)
class Violation:
    """Upstream fails at consumer c, for its finishing sets with `into F` = ``fin_into`` (``fins`` of them): either
    into F ∩ A c is empty (`gate` None: no required gate can reach it), or required gate `gate` reaches no gate of
    into F ∩ A c inside A c."""

    consumer: int
    fin_into: frozenset[int]
    fins: int
    gate: int | None

    @property
    def why(self) -> str:
        if self.gate is None:
            return "into F ∩ A is empty: F reads no gate of the region"
        return f"required gate {self.gate} reaches no gate of into F ∩ A inside A"


@dataclass(frozen=True)
class Verdict:
    """One consumer's certificate: `holds`, its finishing sets (and their distinct `into F ∩ A`), |G|, and the violations."""

    consumer: int
    name: str
    holds: bool
    fins: int
    classes: int
    required: int
    region: int
    violations: tuple[Violation, ...]


def _classes(reads: Reads, K: Consumers, c: int) -> list[tuple[frozenset[int], frozenset[int], int]]:
    """(into F, into F ∩ A c, finishing sets) for each distinct into F of consumer c."""
    A = K.A[c]
    return [(I, I & A, n) for I, n in K.into_counts(reads, c).items()]


def check_consumer(reads: Reads, K: Consumers, c: int, examples: int | None = None) -> Verdict:
    """Upstream at consumer c: every violation (or the first ``examples`` per finishing class)."""
    A, G = K.A[c], K.G[c]
    cls = _classes(reads, K, c)
    targets = sorted({T for _I, T, _n in cls if T}, key=sorted)
    index = {T: k for k, T in enumerate(targets)}
    mask = reaching(reads, A, targets) if G else {}
    out: list[Violation] = []
    for I, T, n in cls:
        if not G:
            continue
        if not T:
            out.append(Violation(c, I, n, None))
            continue
        bit = 1 << index[T]
        bad = [g for g in sorted(G) if not mask.get(g, 0) & bit]
        out.extend(Violation(c, I, n, g) for g in (bad if examples is None else bad[:examples]))
    return Verdict(c, K.name(c), not out, sum(n for _I, _T, n in cls), len({T for _I, T, _n in cls}), len(G), len(A),
                   tuple(out))


def upstream(reads: Reads, K: Consumers, examples: int | None = None) -> list[Verdict]:
    """`Consumers.Upstream`, consumer by consumer: it holds when every verdict does."""
    return [check_consumer(reads, K, c, examples) for c in range(len(K))]


def holds(reads: Reads, K: Consumers) -> bool:
    return all(v.holds for v in upstream(reads, K, examples=1))


def required(reads: Reads, A: frozenset[int], fin_into: Iterable[frozenset[int]]) -> frozenset[int]:
    """The largest G ⊆ A for which Upstream holds at a consumer with region A and finishing sets of these `into F`: the
    gates of A reaching, inside A, a gate of into F ∩ A for every F; empty when some into F ∩ A is."""
    targets = sorted({frozenset(I) & A for I in fin_into}, key=sorted)
    if not targets or any(not T for T in targets):
        return frozenset()
    full = (1 << len(targets)) - 1
    return frozenset(g for g, m in reaching(reads, A, targets).items() if m == full)


def multiplicity(A: Sequence[frozenset[int]]) -> Counter:
    """For each gate, how many regions hold it."""
    out: Counter = Counter()
    for R in A:
        out.update(R)
    return out


def overlap_weights(A: Sequence[frozenset[int]], w: Sequence[Fraction]) -> tuple[bool, int | None]:
    """`Consumers.OverlapWeights ω`, exactly: (holds, a gate whose regions' weights sum past 1, if any)."""
    if any(x < 0 for x in w):
        return False, None
    total: dict[int, Fraction] = {}
    for R, x in zip(A, w, strict=True):
        if x:
            for g in R:
                total[g] = total.get(g, Fraction(0)) + x
    worst = max(total, key=total.__getitem__, default=None)
    return (worst is None or total[worst] <= 1), (worst if worst is not None and total[worst] > 1 else None)


def weights_by_multiplicity(A: Sequence[frozenset[int]]) -> list[Fraction]:
    """ω_c = 1 / (the most regions any gate of A c is in): valid overlap weights, 1 for a region no other meets, since at a
    gate x held by n regions each of them weighs at most 1/n."""
    mult = multiplicity(A)
    return [Fraction(1, max((mult[g] for g in R), default=1)) for R in A]
