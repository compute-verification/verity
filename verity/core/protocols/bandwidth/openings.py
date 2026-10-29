"""Openings of a remote window of chained passes: the verifier's draw, what it certifies, and what it costs.

A remote window runs `P` passes over the store in sequence, pass `j`'s seed bound to the root before it, and commits
one root per pass and device.  After the window's last root the verifier draws `s` openings uniformly from every receipt
the window committed (`draw`): one pooled sample, not one per pass.  Each opening is a label with its path to the
store's registration root, and its receipt leaf with its path to its pass's root.

**What the draw needs.**  Only that it is uniform over the committed receipts and independent of them, which drawing it
after the last root from the verifier's own source gives.  The passes need no coins of their own: the sampling term
counts the window's receipts that the prover never computed correctly, whichever pass they are in, so
`Pr[accept ∧ fewer than (1 − ε)·M correct of M] ≤ (1 − ε)^s` for the window as for one pass (`window_error`).  The
chain matters elsewhere: each pass's compression argument (`RateAudit` at that pass) needs the pass's oracle fresh
when the pass starts, and in the random-oracle model `H(σ, j, root_{j−1})` is fresh for every pass once the pass
index `j` is in its input, whatever roots the prover picks.  Without `j`, a prover that commits a garbage pass whose
root repeats an earlier one replays that pass's seed, and the pass after it is the earlier pass's receipts again,
correct and read once.

**Across windows.**  With each window's sample drawn after that window's roots, the deficits `d_i` of `n` windows
satisfy `Pr[all accepted ∧ mean d ≥ d̄] ≤ (1 − d̄)^(s·n)` however the prover adapts (`horizon_error_log2`): by
induction on `n`, `sup over d₁ of (1 − d₁)^s·(1 − (n·d̄ − d₁)/(n − 1))^(s·(n − 1))` is attained at `d₁ = d̄`.  So a
time-averaged claim over a monitoring period needs `ln(1/δ)/d̄` openings in total, spread over its windows (in proportion to
their receipts when the windows differ).  The proved `RateAuditWindows` is a Markov bound (error `n·ε/m`), which gives
no such saving at small `δ`: it reads every window's error at the per-window `s`.

**λ, the compression slack.**  `RateAudit`'s error term is `C(q, k)/2^λ`: its code names which `k` of the `q` oracle
queries hit the store.  `slack_bits` evaluates it three ways (`Slack`):

- `crude`: `C(q, k) ≤ q^k`, `λ = k·log₂q + t`, the deployed v0's `64·k + 41` (`audit.SLACK_LOG2`);
- `exact`: the same theorem with `C(q, k) ≤ (e·q/k)^k`, `λ = k·log₂(e·q/k) + t` (exactly `log₂ C(q, k) + t` for
  `q` below `2^60`);
- `named`: `log₂ C(N, k) + t`, a proposal: the code names the `k` labels instead, and the decoder finds each label's
  query as the one whose answer is the label's committed receipt.  It fails only on two queries for one label with
  equal answers, `C(q, 2)/2^h` more error, so `λ ≈ ε·N·log₂(e/ε)`, about zero, and needs no bound on `q`.

**Bytes and compute.**  `Geometry` describes a window; `opening_cost` gives the expected bytes and the verifier's work
per opening with every tree's openings sent as one multiproof (`verity.ml.commitments.multiproof`), from the exact
expectation of the multiproof's size under the draw (`expected_siblings`).  The work is counted in operations: SHA-512
compressions in the audit's framing (`pous.verifier`'s rows, leaves and nodes, `tags`' leaves), the codec's squarings
to decode the label, and the sketch's multiply-adds.  `Rates` turns them into core-seconds, and `achievable` gives
`μ_sample + λ` at the most openings a verifier's bandwidth and cores afford.  `retention` is the prover's side: the
receipt trees it keeps until the draw, and the labels it reads again to serve it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from verity.ml.randomness import Key

DOMAIN = "verity/bandwidth/openings/v1"
SHA512_BLOCK = 128
NODE_BYTES = 64
ROW_PREFIX_BYTES = 128
STORE_LEAF_BYTES = 1 + 32 + 2 + 64
NODE_MESSAGE_BYTES = 1 + 2 * NODE_BYTES
RECEIPT_PREFIX_BYTES = 128 + 8
SLACK_ERROR_LOG2 = -41


def draw(key: Key, counts: Sequence[int], s: int) -> tuple[tuple[int, int], ...]:
    """`s` openings, uniform with replacement over every receipt of the window: pass `p` committed `counts[p]`.

    Draw `t` is `key.uniform(Σ counts, ("opening", t))`, read pass-major as `(pass, index)`.  `key` must be derived
    after the window's last root from the verifier's own source, so the draw is independent of what was committed."""
    if type(s) is not int or s < 0 or not counts or any(type(c) is not int or c <= 0 for c in counts):
        raise ValueError("a window draws s >= 0 openings over passes of positive receipt counts")
    ends = [0]
    for c in counts:
        ends.append(ends[-1] + c)
    total = ends[-1]
    out = []
    for t in range(s):
        u = key.uniform(total, ("opening", t))
        p = bisect.bisect_right(ends, u) - 1
        out.append((p, u - ends[p]))
    return tuple(out)


def window_error(receipts: int, epsilon: float, s: int) -> float:
    """`((k − 1)/M)^s` with `k = ⌈(1 − ε)·M⌉`: the chance that `s` pooled openings all land on correct receipts while
    fewer than `k` of the window's `M` are correct."""
    k = math.ceil((1 - epsilon) * receipts)
    if k <= 1:
        return 0.0
    return math.exp(s * math.log((k - 1) / receipts))


def samples_for(epsilon: float, error_log2: float, receipts: int | None = None) -> int:
    """The fewest openings with sampling error at most `2^error_log2` at deficit `ε` (`window_error`; at `M = ∞`
    when `receipts` is None, which is `(1 − ε)^s` and at most one more opening)."""
    ratio = 1 - epsilon if receipts is None else (math.ceil((1 - epsilon) * receipts) - 1) / receipts
    return math.ceil(error_log2 * math.log(2) / math.log(ratio))


def horizon_error_log2(s: int, windows: int, deficit: float) -> float:
    """`log₂ (1 − d̄)^(s·n)`: the chance that all `n` windows of `s` openings each are accepted while their mean deficit
    is at least `d̄`, against a prover that adapts each window to the draws before it (the module's induction)."""
    return s * windows * math.log2(1 - deficit)


def horizon_deficit(s: int, windows: int, error_log2: float) -> float:
    """The mean deficit `d̄` a monitoring period of `n` windows of `s` openings certifies at error `2^error_log2`."""
    return -math.expm1(error_log2 * math.log(2) / (s * windows))


def _log2_choose(n: int, k: int) -> float:
    if not 0 <= k <= n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


class Slack(Enum):
    CRUDE = "crude"
    EXACT = "exact"
    NAMED = "named"


def slack_bits(form: Slack, *, queries_log2: float, blocks: int, certified: int, error_log2: float) -> float:
    """`λ` in bits for one pass of `blocks` labels of which `certified` (`k`) are counted, at compression error
    `2^error_log2` (`error_log2 < 0`); see the module's three forms.  `named` is a proposal, not yet a Lean statement."""
    t = -error_log2
    if form is Slack.CRUDE:
        return certified * queries_log2 + t
    if form is Slack.EXACT:
        if queries_log2 < 60:
            return _log2_choose(round(2.0 ** queries_log2), certified) + t
        return certified * (queries_log2 - math.log2(certified) + math.log2(math.e)) + t
    return _log2_choose(blocks, certified) + t


def expected_known(leaves: int, weight: float, s: int) -> list[float]:
    """Expected known nodes at each level `0..D` of the padded tree over `leaves` real leaves
    (`verity.ml.commitments.multiproof`'s), each leaf drawn with probability `weight` per opening, after `s` openings
    with replacement.  The audit's RFC 6962 tree is the padded one with its padding pruned, so it needs at most as
    many siblings."""
    depth = (leaves - 1).bit_length()
    known = []
    for level in range(depth + 1):
        width = 1 << level
        full, rest = divmod(leaves, width)
        hit = -math.expm1(s * math.log1p(-width * weight)) if width * weight < 1 else 1.0
        part = 0.0 if not rest else 1.0 if rest * weight >= 1 else -math.expm1(s * math.log1p(-rest * weight))
        known.append(full * hit + part)
    return known


def expected_siblings(leaves: int, weight: float, s: int) -> float:
    """Expected sibling digests of the tree's multiproof: `Σ_ℓ (2·K_(ℓ+1) − K_ℓ)` over levels `0..D−1`, since each
    known node either pairs with its known sibling or takes one from the proof."""
    k = expected_known(leaves, weight, s)
    return sum(2 * k[level + 1] - k[level] for level in range(len(k) - 1))


def expected_node_hashes(leaves: int, weight: float, s: int) -> float:
    """Expected node hashes the verifier computes for the tree: one per known node above the leaves."""
    return sum(expected_known(leaves, weight, s)[1:])


def compressions(message_bytes: int) -> int:
    """SHA-512 compressions of a message of `message_bytes` bytes, padding included."""
    return (message_bytes + 17 + SHA512_BLOCK - 1) // SHA512_BLOCK


@dataclass(frozen=True, slots=True)
class Geometry:
    """A window: `stores` registration trees (a device's or a tier's store) of `blocks` labels each, read `passes`
    times, each pass committing one receipt tree per store with `receipts_per_leaf` receipts a leaf.  A receipt is
    `H(σ_j, i, sketch(E_i), z)`: `sketch_rows` multiply-adds per label byte, and `z`'s digest.  `bound_bytes` is what an
    opening carries of the matmul output its receipt binds (`z_(j−1)`), zero when PoUW's draw already opens it.  The
    verifier decodes each opened label, `codec_rounds` squarings per `word_bytes`."""

    stores: int
    blocks: int
    passes: int
    block_bytes: int
    receipt_bytes: int = 64
    receipts_per_leaf: int = 1
    bound_bytes: int = 0
    sketch_rows: int = 16
    sketch_bytes: int = 64
    codec_rounds: int = 2
    word_bytes: int = 64

    def __post_init__(self) -> None:
        if min(self.stores, self.blocks, self.passes, self.block_bytes, self.receipts_per_leaf) < 1:
            raise ValueError("a window has at least one store, block, pass and receipt a leaf")

    @property
    def receipts(self) -> int:
        return self.stores * self.blocks * self.passes

    @property
    def store_bytes(self) -> int:
        return self.stores * self.blocks * self.block_bytes


@dataclass(frozen=True, slots=True)
class Rates:
    """The verifier's core-seconds per operation: a SHA-512 compression, a codec squaring, a sketch multiply-add."""

    compression: float
    squaring: float
    sketch_op: float


@dataclass(frozen=True, slots=True)
class OpeningCost:
    """Expected cost per opening of a window's `s` openings: WAN bytes (`block`, `store_path`, `leaf`, `receipt_path`,
    `bound`) and the verifier's operations (SHA-512 `compressions`, codec `squarings`, `sketch_ops`)."""

    s: int
    block: float
    store_path: float
    leaf: float
    receipt_path: float
    bound: float
    compressions: float
    squarings: float
    sketch_ops: float

    @property
    def bytes(self) -> float:
        return self.block + self.store_path + self.leaf + self.receipt_path + self.bound

    def core_seconds(self, rates: Rates) -> float:
        return (self.compressions * rates.compression + self.squarings * rates.squaring
                + self.sketch_ops * rates.sketch_op)


def opening_cost(g: Geometry, s: int) -> OpeningCost:
    """Expected per-opening cost of `s >= 1` pooled openings, every tree's paths sent as one multiproof.

    A label's registration leaf is drawn in every pass (weight `passes/M`), a receipt leaf in one (weight
    `receipts_per_leaf/M`).  Distinct openings send their label and leaf once and the verifier hashes and decodes each
    once; repeats are free.  Every opening recomputes its receipt."""
    if type(s) is not int or s < 1:
        raise ValueError("a cost is per opening of s >= 1 openings")
    m = g.receipts
    leaves = -(-g.blocks // g.receipts_per_leaf)
    w_store, w_leaf = g.passes / m, g.receipts_per_leaf / m
    store_k = expected_known(g.blocks, w_store, s)
    leaf_k = expected_known(leaves, w_leaf, s)
    receipt_trees = g.stores * g.passes
    distinct_blocks = g.stores * store_k[0]
    distinct_leaves = receipt_trees * leaf_k[0]
    node = compressions(NODE_MESSAGE_BYTES)
    hashes = (distinct_blocks * (compressions(ROW_PREFIX_BYTES + g.block_bytes) + compressions(STORE_LEAF_BYTES))
              + g.stores * sum(store_k[1:]) * node
              + distinct_leaves * compressions(RECEIPT_PREFIX_BYTES + g.receipts_per_leaf * g.receipt_bytes)
              + receipt_trees * sum(leaf_k[1:]) * node
              + s * compressions(RECEIPT_PREFIX_BYTES + g.sketch_bytes + NODE_BYTES))
    return OpeningCost(
        s,
        block=distinct_blocks * g.block_bytes / s,
        store_path=g.stores * expected_siblings(g.blocks, w_store, s) * NODE_BYTES / s,
        leaf=distinct_leaves * g.receipts_per_leaf * g.receipt_bytes / s,
        receipt_path=receipt_trees * expected_siblings(leaves, w_leaf, s) * NODE_BYTES / s,
        bound=g.bound_bytes,
        compressions=hashes / s,
        squarings=distinct_blocks * g.codec_rounds * -(-g.block_bytes // g.word_bytes) / s,
        sketch_ops=g.sketch_rows * g.block_bytes,
    )


@dataclass(frozen=True, slots=True)
class Retention:
    """What serving a window's draw costs the prover: the bytes of receipt trees it keeps until the draw, and the
    expected bytes of labels it reads again to rebuild the receipts under the nodes the draw opens."""

    level: int
    kept: float
    reread: float


def retention(g: Geometry, s: int, level: int) -> Retention:
    """The prover keeps every receipt tree from `level` up until the window's `s` openings are drawn (level 0: every
    receipt), and for each opened node at `level` rebuilds the receipts under it, reading each one's label again.
    Rebuilding a receipt needs its label as it was in that pass and its `z`, so a label may not change, nor its `z`
    become unrecoverable, before the window's draw."""
    leaves = -(-g.blocks // g.receipts_per_leaf)
    depth = (leaves - 1).bit_length()
    if type(s) is not int or s < 1 or not 0 <= level <= depth:
        raise ValueError(f"a draw has s >= 1 openings, and a kept level is in 0..{depth}")
    trees = g.stores * g.passes
    nodes = sum(-(-leaves // (1 << up)) for up in range(max(level, 1), depth + 1))
    kept = trees * (nodes * NODE_BYTES + (g.blocks * g.receipt_bytes if level == 0 else 0))
    if level == 0:
        return Retention(level, kept, 0.0)

    def rebuilt(receipts: int) -> float:
        hit = 1.0 if receipts >= g.receipts else -math.expm1(s * math.log1p(-receipts / g.receipts))
        return receipts * hit

    span = (1 << level) * g.receipts_per_leaf
    full, rest = divmod(g.blocks, span)
    return Retention(level, kept, trees * g.block_bytes * (full * rebuilt(span) + (rebuilt(rest) if rest else 0.0)))


@dataclass(frozen=True, slots=True)
class Point:
    """A parameter point: `s` openings a window over a monitoring period of `windows`, the λ form, and what it certifies."""

    s: int
    windows: int
    slack: Slack
    mu_sample: float
    lam: float
    cost: OpeningCost

    @property
    def total(self) -> float:
        return self.mu_sample + self.lam

    @property
    def window_bytes(self) -> float:
        return self.s * self.cost.bytes

    def window_core_seconds(self, rates: Rates) -> float:
        return self.s * self.cost.core_seconds(rates)


def point(g: Geometry, s: int, *, windows: int = 1, slack: Slack = Slack.NAMED, error_log2: float = -40,
          queries_log2: float = 64, slack_error_log2: float = SLACK_ERROR_LOG2) -> Point:
    """`μ_sample + λ` as fractions of the window's reads at `s` openings a window over `windows` windows: `μ_sample`
    from `horizon_deficit` at `2^error_log2`, `λ` per pass over the window's stores at the pass's certified labels,
    as a share of a pass's bits.  The prover picks each pass's certified count, so the slack's error is shared over
    every pass and count, `log₂(P·(N + 1))` more bits."""
    mu = horizon_deficit(s, windows, error_log2)
    blocks = g.stores * g.blocks
    k = math.ceil((1 - mu) * blocks)
    union = math.log2(g.passes * (blocks + 1))
    lam = slack_bits(slack, queries_log2=queries_log2, blocks=blocks, certified=k, error_log2=slack_error_log2 - union)
    return Point(s, windows, slack, mu, lam / (8 * g.store_bytes), opening_cost(g, s))


def _first(ok, *, start: int = 1) -> int:
    """The least `s >= start` with `ok(s)`, for `ok` false then true."""
    lo, hi = start, start
    while not ok(hi):
        lo, hi = hi + 1, 2 * hi
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def cheapest(g: Geometry, target: float, *, windows: int = 1, slack: Slack = Slack.NAMED, error_log2: float = -40,
             queries_log2: float = 64) -> Point | None:
    """The fewest openings a window whose `μ_sample + λ` is at most `target`, or None if `λ` alone reaches it.  `λ`
    grows with the certified labels, so the floor is its value at all `N` certified, which no `s` gets under."""
    def at(s: int) -> Point:
        return point(g, s, windows=windows, slack=slack, error_log2=error_log2, queries_log2=queries_log2)

    blocks = g.stores * g.blocks
    floor = slack_bits(slack, queries_log2=queries_log2, blocks=blocks, certified=blocks,
                       error_log2=SLACK_ERROR_LOG2 - math.log2(g.passes * (blocks + 1)))
    if floor / (8 * g.store_bytes) >= target:
        return None
    return at(_first(lambda s: at(s).total <= target))


def achievable(g: Geometry, *, bandwidth: float, cores: float, rates: Rates, window_s: float, windows: int = 1,
               slack: Slack = Slack.NAMED, error_log2: float = -40, queries_log2: float = 64) -> Point | None:
    """`μ_sample + λ` at the most openings a window that the verifier affords: `bandwidth` bytes a second of WAN and
    `cores` cores at `rates`, over windows of `window_s` seconds.  None if it affords no opening."""
    def at(s: int) -> Point:
        return point(g, s, windows=windows, slack=slack, error_log2=error_log2, queries_log2=queries_log2)

    def over(s: int) -> bool:
        p = at(s)
        return p.window_bytes > bandwidth * window_s or p.window_core_seconds(rates) > cores * window_s

    if over(1):
        return None
    return at(_first(over) - 1)


__all__ = [
    "DOMAIN",
    "Geometry",
    "OpeningCost",
    "Point",
    "Rates",
    "Retention",
    "Slack",
    "achievable",
    "cheapest",
    "compressions",
    "draw",
    "expected_known",
    "expected_node_hashes",
    "expected_siblings",
    "horizon_deficit",
    "horizon_error_log2",
    "opening_cost",
    "point",
    "retention",
    "samples_for",
    "slack_bits",
    "window_error",
]
