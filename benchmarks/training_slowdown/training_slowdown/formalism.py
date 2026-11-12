"""A record's slowdown at a policy (c, k), by the training formalism v2: its theorems Fit and Slowdown per record,
the one bound's products term, and the attack side beside them. |C|, c, w and d are Boolean gates
(training_slowdown.gates), every other quantity bits.

Measured per record (`Measured`), each with its source in `labels` ("measured: ...", "counted: ...", "stub: ...",
"assumption (A1): ..." for a profile the claim takes from an assumption, or "upper bound (witness): ..." for a value a
witness bounds from above):

- `parts`: one per distinct step structure, each with its gates |C_j|, its token blocks (one per token per step,
  `tokens`), the gates in them (`token_gates`: each token's forward and backward; the optimizer is in no token block),
  the affected gates among those, |A ∩ ∪T| (`affected_gates`: the gates that depend on θ'), and the gradient and
  optimizer state its blocks pass between them (`delta_grad_bits`, Δ_grad; None: unmeasured);
- `tau_bits` τ, the inputs outside θ a token block reads; `delta_bits` Δ, the bits that describe θ' given θ, by a
  decoder of `decoder_gates` d gates (`decoder`; None: unmeasured, so Δ never counts and Fit fails); `delta_t_bits`
  Δ_t, the weights in the form the update rule reads and writes: a straight-through run's fp32 latents at 32 bits a
  value (None: Δ, for every other run);
- `a_bits` a, the cut cost: the computed bits a block reads to do any of a token's affected work without computing that
  token from its inputs;
- `u`, A1's update profile: u(x) the update bits a block must read to do a share x of one token's affected work, as
  points (x, bits) from (0, 0) to (1, u(1)), linear between them (two points at one x are a jump), u(1) (`u1_bits`) the
  per-token update; `u_in`, A1's input profile: the update bits a block must read to do the first share x, in
  dependency order, while computing the token from its inputs. Update bits are values of θ', or of an encoding of it,
  that a block reads, disjoint from the cut values A1 charges as a·m. For an evenly spread update both are x·u(1)
  (`even`); with frozen layers above changed ones, their share t of a token's affected work (`tail_share`), u is 0 up
  to t and u_in jumps to the changed layers' full update at once (`tail`): the frozen work is always charged. Where Δ
  counts, both are at most Δ (`capped`): a block can read the update code instead ("a shorter update encoding lowers
  u"), a witness the record exhibits (the checkpoint's, for every step of its prefix);
- `products`: the run's product rows as the products term reads them (`work.products_of_trace`).

At (c, k), with s = c/w, w = token_gates / tokens (each part's for Fit; the record's for Slowdown) and
σ = token_gates / |C|:

- **Fit** (construction, checked per part): the training's three kinds of block, compute Δ + sτ ≤ k, reduction
  2Δ_grad ≤ k and update application Δ_t + Δ_grad ≤ k (Δ counts only when its decoder fits, d < c, and so does Δ_t;
  a part with no token blocks has only the compute kind). Where every part fits, the record's slowdown is c/(c − d) (`rho_fit`).
- **Slowdown** (the counting lemma at B = A \\ R; stated, not yet proved in Lean): its token form with the largest
  per-token affected count α_max where token blocks vary, slowdown ≥ σ(1 − x0)·(s/s*)·(ᾱ/α_max), ᾱ the mean per-token
  affected count (for identical token blocks the last factor is 1). The claim takes the assumption's upper bounds,
  never best-known values: s* ≤ max_x x(k − u(x))/a and x0 ≤ max{x : u_in(x) ≤ k} (A1). A target, not a bound: the
  one bound publishes it as a sensitivity (`token_form`).
- **The attack side** (tightness, never the claim): the best-known x0 and s* from exhibited blocks. `x0_known` is 1
  where the compute block fits (it computes s tokens from their inputs), else 0; `s_star_known` is the frozen tail's
  block's, which is handed each token's state at the tail (a + τ bits) and does the tail's share t of its affected
  work reading no update bits, t·min(⌊k/(a + τ)⌋, ⌊c/(t·α_max)⌋), and 0 where there is no tail or x0_known is 1.

Where Fit holds, Δ + sτ ≤ k bounds u_in by k through the cap, so x0's bound is 1 and Slowdown's bound 0: the two never
disagree.

**The context charge** (`Measured.context`, a sensitivity for A1, off by default). A1 charges a block a per token it
serves. The charge adds, for each token whose affected work a block does without computing the token from its inputs,
at each layer the block works on for it, the keys and values of every earlier token of the sequence at that layer that
the block doesn't compute itself: κ = 2·d_kv·16 bits per earlier token per layer, each counted once per block. Only K/V
that depend on θ' are charged, at the layers ℓ ≥ ℓ_kv (`context`). A block whose tokens are the span [q, q + m) of a
sequence, doing a share x of each one's affected work and some of it at such a layer, reads u(x) + a·m + κ·n·q ≤ k
(n ≥ 1 such layers). With a ≤ κ a cut token is cheaper than one earlier token's K/V at one layer, so its best span
starts at the sequence start and every token p it serves has a(p + 1) ≤ k − u(x). With L = ctx positions, h the share
of a token's affected work at no such layer (`free_share`) and h' = min(h, 1 − x0), Slowdown's bound becomes

    the record's slowdown ≥ σ·s·(ᾱ/α_max)·(1/L)·Σ_{p<L} [h'/max(S0, S1(p)) + (1 − x0 − h')/S1(p)],
    S1(p) = max{x(k − u(x))/a : u(x) + a(p + 1) ≤ k},   S0 = max{x(k − u(x))/a : x ≤ h},

which is the token form with s* the harmonic mean (1 − x0)/((1/L)·Σ_{p<L}[...]) (`s_star`; v2's A1 value is
`s_star_a1`). S1(p) is v2's s* while p + 1 ≤ (k − u(x*))/a at v2's optimum x*, so a record whose optimal block holds
L tokens keeps v2's slowdown. S1(p) is 0 once a(p + 1) > k, so a charged share left there makes the record's slowdown infinite (s* 0). The
choices favourable to the block:
- its span starts at the sequence start;
- a token in its span counts as computed by it at every layer (its K/V are never charged), and K/V that depend on θ
  alone are free;
- u(x) and the charge are minimised separately;
- h is an upper bound (`context`) and lies outside x0;
- tokens are fractional, as in v2's room.
A block that computes a token from its inputs (x0) is not charged, and the attack side is v2's.

**The one bound** (`bound`, which `at` reads): the counting lemma over kinds K of block work, each a lower
bound on its own, the record's slowdown ≥ max_K (|B_K|/|C|)/(cap_K(k)/c): each kind's term, the max, the kind it
binds on and what each kind rests on (`RESTS_ON`). Where Fit fails the record's slowdown is the entry point's, max(1,
that max); where Fit holds it is c/(c − d), and every term is at most that. Its one kind:
- products: the theorem of record's count on M (`RESTS_ON` has its statement), a theorem for the run's circuit given
  (H) and (E). The term is `calculator.slowdown`'s, the benchmark's one entry point to a slowdown, before its floor at
  1: it reads the record's products (`Measured.products`, `work.products_of_trace` of the trace's rows) at the
  record's |C|, τ and update code, chooses the work and asks the slowdown calculator's bound of record. Nothing here
  restates the count or computes the term; `training_slowdown.work` states M and `calculator` the choice. Where the
  record states no M the term is None with the entry point's reason. At a fixed k the term is linear in c.
A kind's term is None where its inputs are unmeasured, with the reason (`unmeasured`), and the max is over the measured
ones (None, and so the record's slowdown where Fit fails, where none is). Beside the kinds, never in the max, the sensitivities
(`SENSITIVITIES`), each labelled with what it rests on:
- reduction: where a step's token work fills b ≥ 2 blocks by gates (b = ⌈N·w/c⌉), its b − 1 partial gradients at
  their passed width (`Part.partial_bits`) are read by blocks of k bits each, c·Σ steps·(b − 1)·partial/(k|C|)
  (assumption (R));
- update: there, the blocks applying the update read the gradient's cut (`Part.grad_bits`), the weights read once and θ
  free, c·Σ steps·γ/(k|C|) (assumption (U));
- token_form: Slowdown's token form under A1, a target;
- decoding_form: a block reading the update code holding at most max(c − d, min(c, α_max(x0·s + s*))) affected gates
  (assumption (D) with A1), a target.
(R) and (U) hold only for partitions split by tokens, not by layers or tensors (a block holding only frozen layers'
work passes no partial), and the token and decoding forms are targets, so none is a lower bound on its own; `kinds`
refuses them. Each result is labelled "training's slowdown code VERSION" (`THE_CODE`). `published` is the record's slowdown as the console shows it: Fit's c/(c − d)
where it holds; else the entry point's slowdown and what it says binds: the products' count where it exceeds 1, None
where the record gives none, else 1, binding none. Each comes with the kind it binds on and what that rests on, short
(`RESTS_ON_CELL`; the products term's with the subset of M that binds, its F, β and b_y, `he_rests_on`, which
`products_rests_on` carries wherever the term is measured), and the reduction and update sensitivities.

Where the quantities come from (`measure`): a row's `formalism` block, which `of_record` writes from the run record or
a re-score attaches, else the stubs below from a v2.3 row's own fields. training_slowdown 4.0 measures them on the
record's traced step (training_slowdown.capture.circuit, `TRACED` its labels): each traced structure's σ, affected
share, token factor and Δ_grad (`TRACED_STRUCTURE`), and the record's τ (`tau_ids` token positions read per consumed
token at ⌈log2 V⌉ bits, plus `tau_other_bits`), a, u_in, u and t (`of_trace`, with every measured field under
`trace`). Δ stays 8 |E|, the submitted code: the trace's own encoding of θ' (`trace.update_cut_bits`) is the carried
state, θ' in full. A run record may also give its own block, {"parts": [...], "tau_bits", "a_bits", "u", "u_in",
"u1_bits", "tail_share", "delta_bits", "delta_t_bits", "decoder_gates", "products", "labels"}, which `of_record` passes
through field by field. The trace's Δ_grad leaves θ' out of its carried state (a latent the step quantizes is θ' as the
step reads it), so Δ_t + Δ_grad counts the latents once.

**The decoder** (`decoder`): d is the block's `decoder_gates` where it gives one, else the gates the record's program
reports for the checkpoint's E (`output.decoder_gates`, from the submission's `notes.decoder`), else the witness's: the
harness's decoder for E, `rebuild_macs(θ, E)` fp32 multiply-adds (`output.rebuild_macs`) at F32Fma_v4's gates each. d
is 0 only where no decoder exists (E stores θ' directly: rebuild_macs 0, or a full or model E), and unmeasured (Fit
refused) where the row has none of these.

**Refusals** (`refusal`, `measure` raising `Refused`): a traced row's measured quantities are refused, never scored,
with the reason, when
- its trace lists no read-backs: it was traced before 4.2, whose capture lists them (unchecked);
- its capture lists a read-back that decides what the graph can't carry (`trace.data_dependent`: a Python clip or
  branch on a read-back loss): the update the trace shows is the stand-in's, not the run's;
- a Python number is folded over steps (`trace.carried_scalars`, an EMA of read-back estimates kept as a float): the
  trace carries every earlier step's read-back, so its Δ_grad grows with the step;
- E is not empty (Δ > 0, or E's size unknown) but the trace shows no update: its affected share is 0, or u(1) is 0
  (each where the trace states it: a job's own trace states no u, and its affected share is its products').
Any row with a nonempty E whose prefix has no affected gates (no step through its checkpoint reads θ') is refused too.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import calculator, policy
from .gates import gates, of_log, per_mac, split_key
from .work import THE_CODE, products_of_trace

ACT_BITS = 16


@dataclass(frozen=True)
class Part:
    gates: float
    tokens: float
    token_gates: float
    affected_gates: float
    steps: int = 1
    delta_grad_bits: float | None = None
    #: the bound's reduction and update kinds: one token block's partial gradient and the step's gradient cut, each at
    #: its passed width (the trace's `partial_cut_bits` and `grad_cut_bits`; None: unmeasured)
    partial_bits: float | None = None
    grad_bits: float | None = None


@dataclass(frozen=True)
class Measured:
    parts: tuple[Part, ...]
    tau_bits: float
    a_bits: float
    u: tuple[tuple[float, float], ...]
    u_in: tuple[tuple[float, float], ...]
    delta_bits: float | None
    decoder_gates: float | None = None
    labels: dict = field(default_factory=dict)
    tail_share: float = 0.0
    context: dict | None = None
    delta_t_bits: float | None = None
    #: the products kind's inputs, the run's product rows as the term reads them (`products_of_trace`: {"k", "he",
    #: "p_match", ...}; None: no captured step and no trace product rows)
    products: dict | None = None

    @property
    def delta_t(self) -> float | None:
        """Δ_t, the update application's weights: `delta_t_bits`, else Δ."""
        return self.delta_t_bits if self.delta_t_bits is not None else self.delta_bits

    @property
    def gates(self) -> float:
        return sum(p.gates for p in self.parts)

    @property
    def tokens(self) -> float:
        return sum(p.tokens for p in self.parts)

    @property
    def token_gates(self) -> float:
        return sum(p.token_gates for p in self.parts)

    @property
    def sigma(self) -> float:
        """σ, the share of |C| in token blocks."""
        return self.token_gates / self.gates if self.gates else 0.0

    @property
    def w(self) -> float:
        return self.token_gates / self.tokens if self.tokens else math.inf

    @property
    def alpha_max(self) -> float:
        """The largest per-token affected count over the parts with token blocks."""
        return max((p.affected_gates / p.tokens for p in self.parts if p.tokens), default=0.0)

    @property
    def alpha_mean(self) -> float:
        return sum(p.affected_gates for p in self.parts) / self.tokens if self.tokens else 0.0

    @property
    def u1_bits(self) -> float:
        """u(1), the per-token update: Slowdown's, a field of its own beside Fit's Δ."""
        return self.u[-1][1]

    @property
    def alpha_ratio(self) -> float:
        """ᾱ/α_max, Slowdown's token-form factor (0 where nothing is affected)."""
        return self.alpha_mean / self.alpha_max if self.alpha_max else 0.0

    @property
    def w_fit(self) -> float:
        """The least w of a part with token blocks, the one Fit's compute block binds on (inf: none has any)."""
        return min((p.token_gates / p.tokens for p in self.parts if p.tokens), default=math.inf)

    @property
    def fit_delta_grad_bits(self) -> float | None:
        """The Δ_grad Fit's reduction and update blocks bind on: the largest of the parts that have them (None where a
        part with token blocks measures none, so Fit fails at every k; 0 where no part has them)."""
        if any(p.tokens and p.delta_grad_bits is None for p in self.parts):
            return None
        return max((p.delta_grad_bits for p in self.parts if p.delta_grad_bits is not None), default=0.0)

    @property
    def delta_grad_bits(self) -> float | None:
        """The largest Δ_grad of the parts that measure one (None: none does)."""
        return max((p.delta_grad_bits for p in self.parts if p.delta_grad_bits is not None), default=None)


# ---- the update profiles -----------------------------------------------------------------------------------------------

def even(u1: float) -> tuple[tuple[float, float], ...]:
    """The evenly spread profile u(x) = x·u(1)."""
    return ((0.0, 0.0), (1.0, float(u1)))


def tail(u1: float, t: float) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """(u, u_in) with frozen layers above changed ones, their share t of a token's affected work: a block does the
    frozen work reading no update bits, so u is 0 up to t and the changed update evenly spread over the rest; computing
    the token from its inputs passes through the changed layers first, so u_in jumps to their full update at once. At
    t = 0 both are `even`."""
    if not 0 < t < 1:
        return even(u1), even(u1)
    return ((0.0, 0.0), (t, 0.0), (1.0, float(u1))), ((0.0, 0.0), (0.0, float(u1)), (1.0, float(u1)))


def with_tail(u_in, t: float) -> tuple[tuple[float, float], ...]:
    """A1's profile from a measured u_in and the share t of a token's affected work that reads no update bits: a block
    does that share handed the cut values (A1's a·m), so 0 up to t; beyond it, the first share x in dependency order
    with it reads at most u_in(x)."""
    if not 0 < t < 1:
        return tuple(map(tuple, u_in))
    return ((0.0, 0.0), (t, 0.0), (t, u_at(u_in, t)), *((x, b) for x, b in u_in if x > t))


def capped(u, cap: float) -> tuple[tuple[float, float], ...]:
    """min(u(x), cap) at every x, as points: a segment crossing `cap` gets its crossing point."""
    out = [(u[0][0], min(u[0][1], cap))]
    for (x0, b0), (x1, b1) in zip(u, u[1:]):
        if x1 > x0 and (b0 - cap) * (b1 - cap) < 0:
            out.append((x0 + (x1 - x0) * (cap - b0) / (b1 - b0), cap))
        out.append((x1, min(b1, cap)))
    return tuple(out)


def u_at(u, x: float) -> float:
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if x <= x1:
            return u0 if x1 == x0 else u0 + (u1 - u0) * (x - x0) / (x1 - x0)
    return u[-1][1]


def bottom_share(u, k: float) -> float:
    """The largest x in [0, 1] with u(x) ≤ k (u non-decreasing): A1's bound on x0, read on the input profile u_in."""
    if u[-1][1] <= k:
        return 1.0
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if u1 > k:
            return x0 if u1 == u0 or u0 > k else x0 + (x1 - x0) * (k - u0) / (u1 - u0)
    return 0.0


def room(u, k: float, upto: float = 1.0) -> float:
    """max over x in (0, upto] of x·(k − u(x)), A1's a·s* (`upto` 1): on each linear piece a concave quadratic, so its
    ends and its vertex."""
    best = 0.0
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if x0 > upto:
            break
        end = min(x1, upto)
        xs = [x0, end]
        if x1 > x0 and u1 > u0:
            m = (u1 - u0) / (x1 - x0)
            v = (k - u0 + m * x0) / (2 * m)
            if x0 < v < end:
                xs.append(v)
        best = max((best, *(x * (k - u_at(u, x)) for x in xs if x > 0)))
    return best


def context_rooms(u, k: float, a: float, positions: int) -> np.ndarray:
    """S1(p) for p = 0, ..., positions − 1 (the context charge, module docstring): the most of a token's affected work a
    block can do, in tokens, when its span runs from the sequence start through p, max x(k − u(x))/a over the x with
    u(x) + a(p + 1) ≤ k; 0 where none has. The max is at a point of u or a piece's vertex whose u fits, or where a
    piece crosses the limit."""
    lim = k - a * np.arange(1, positions + 1, dtype=float)
    pts = [(b, x * (k - b)) for x, b in u]
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if x1 > x0 and u1 > u0:
            m = (u1 - u0) / (x1 - x0)
            v = (k - u0 + m * x0) / (2 * m)
            if x0 < v < x1:
                pts.append((u0 + m * (v - x0), v * (k - u0 - m * (v - x0))))
    pts.sort()
    need = np.array([b for b, _ in pts])
    best = np.maximum.accumulate(np.array([w for _, w in pts]))
    i = np.searchsorted(need, lim, side="right")
    out = np.where(i > 0, best[np.maximum(i - 1, 0)], 0.0)
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if x1 > x0 and u1 > u0:
            cross = (lim > u0) & (lim < u1)
            x = x0 + (lim[cross] - u0) * (x1 - x0) / (u1 - u0)
            out[cross] = np.maximum(out[cross], x * (k - lim[cross]))
    return np.where(lim >= 0, np.maximum(out, 0.0), 0.0) / a


def context_s_star(ctx: dict, u, k: float, a: float, x0: float, s_star: float) -> float:
    """s* under A1 and the context charge `ctx` (`context`): the harmonic mean over the L positions of S1(p) for the
    charged share 1 − x0 − h' and of max(S0, S1(p)) for the context-free share h' = min(h, 1 − x0) (module
    docstring); v2's `s_star` where nothing is left to charge."""
    rest = 1 - x0
    if rest <= 0:
        return s_star
    if a > ctx["kv_bits"]:
        raise ValueError(f"a = {a} bits exceeds one earlier token's K/V at one layer ({ctx['kv_bits']} bits): the "
                         "context charge's best span is no longer the sequence start, which context_s_star assumes")
    free = min(ctx["free_share"], rest)
    s1 = context_rooms(u, k, a, ctx["positions"])
    with np.errstate(divide="ignore"):
        per = np.zeros_like(s1)
        if free > 0:
            per += free / np.maximum(room(u, k, ctx["free_share"]) / a, s1)
        if rest - free > 0:
            per += (rest - free) / s1
    mean = float(per.mean())
    return rest / mean if mean > 0 else s_star


# ---- the theorems at (c, k) --------------------------------------------------------------------------------------------

def _token_form(sigma: float, x0: float, s: float, s_star: float, ratio: float) -> float:
    lead = sigma * (1 - x0) * s * ratio
    if lead == 0:
        return 0.0
    return lead / s_star if s_star > 0 else math.inf


def _blocks(p: Part, c: float) -> float:
    """The fewest blocks a step's token work of part `p` fills by gates, ⌈N·w/c⌉ (N its tokens a step)."""
    return math.ceil(p.token_gates / p.steps / c) if p.tokens else 0


#: the one bound's kinds (`bound`), each a lower bound on its own
KINDS = ("products",)


def _beta_rows() -> str:
    """The slowdown calculator's rows of β(k) (`calculator.beta_rows`), as a sentence states them."""
    (k0, b0), *rest = calculator.beta_rows()
    return ", ".join([f"{b0:g} up to k = {k0:,.0f}", *(f"{b:g} at {k:.0e}".replace("e+0", "e") for k, b in rest)])


#: what each kind rests on
RESTS_ON = {
    "products": "theorem of record for this circuit: every partition into blocks of at most c gates, each "
                "reading at most k bits outside θ, does N ≤ m(k + α) + F·(k/2)·√(mO/(β_w·β_x)) of M's N MACs "
                "in its m blocks, α = F·½·√(k³/(β_w·β_x·b_y)), so the record's slowdown ≥ c/|C| times the least m "
                "that count allows, floored at 1, O M's output wires; given these premises "
                "on M, each refuted by a single counterexample: (H) operands are hard: every set of M's words "
                "with a cut of at most k bits from the unregistered inputs has a cut of at least each word's β, "
                "the casts of one value counting as one word: a weight's (an operand no token indexes) its "
                "width as the circuit passes it, at most the run record's code bits a trained weight (β_w "
                "from the run record's code size), an activation's at most a uniform β(k) (β(k) per "
                f"the red-team of (H): {_beta_rows()}, between two of them the next one's, none past it), and "
                "β_eff = β/(1 + (K*/r)²), "
                "K* = ½√(k/β), for "
                "an operand the update reaches only through a bottleneck of rank r a row or column (an "
                "adapter's Ax, a broadcast vector, a frozen edge); each product of M has β_w·β_x at most its "
                "two operands' product, and a product below it is left out of M; (E) exports are wide: every "
                "gate set's cut to the outputs it alone feeds is at least b_y bits an output, b_y M's least "
                "output width; and the structure, which the product rows don't record: M's matmuls have "
                "distinct words, W and X disjoint, no two share both a w- and an x-word, and no four arrays "
                "are pairwise multiplied (a layer's forward, input-gradient and weight-gradient products share "
                "three: F = 2/√3 where M holds a backward product, else 1, and 1 on weight gradients alone). M "
                "is the grader's, the best of three: the products whose two operands both depend on the update, "
                "less one array times itself, a product reading a gather's rows (the lowest layer's q, k and "
                "v), attention's products, the optimizer's (Newton-Schulz's B·X), from k = 8e7 the head's "
                "input and weight gradients, the positions sequences share, and the steps before the run's "
                "first update and the ⌊k/(τT)⌋ after it (T a step's tokens), whose weights a block's k bits "
                "can rebuild; that M less the products whose contraction K is below the K where the outputs' "
                "term outweighs the main one; and the weight gradients alone",
}
#: what the bound publishes beside its kinds, never in its max, each with what it rests on and why it isn't a kind
SENSITIVITIES = {
    "reduction": "sensitivity, assumption (R): each block holding a step's token work passes one partial gradient at "
                 "its passed width (the trace's partial cut), and the step's token work fills at least ceil(N w / c) "
                 "blocks; it holds only for partitions split by tokens (a block holding only frozen layers passes no "
                 "partial)",
    "update": "sensitivity, assumption (U): where two or more blocks hold a step's token work, the block applying the "
              "update to a coordinate reads its share of the gradient's cut, the weights read once and theta free; "
              "it holds only for partitions split by tokens (a tensor's gradient made and applied in one block "
              "breaks it)",
    "token_form": "sensitivity, assumption (A1): Slowdown's token form, s* the largest x(k - u(x))/a over x and x0 "
                  "from the input profile, u capped at Delta for a block reading the update code; a target, not a "
                  "bound",
    "decoding_form": "sensitivity, assumption (D) with A1: a block reading the update code runs the whole decoder, so "
                     "it holds at most max(c - d, min(c, α (x0 s + s*))) affected gates, α the most affected gates a "
                     "token has in any part, x0 and s* from the uncapped profiles; a target, not a bound",
}
#: the matrix products a trace's rows and a run's op log both give as (m, K, n), by one rule (training_slowdown.capture.
#: structure's GEMMs and `log_products`): a batched product is left out, since a capture's attention, by the math
#: backend, has no counterpart in a record written before training_slowdown 4.22.0, which holds a fused attention
#: kernel as one row
PLAIN_PRODUCTS = ("mm", "_int_mm", "_scaled_mm", "addmm", "addmm_", "mv", "addmv", "addmv_", "dot", "vdot", "addr",
                  "addr_")
#: the fused attention kernels an op log written before training_slowdown 4.22.0 names, one row a kernel call
SDPA = ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention")


def log_products(op: str, shapes: list) -> list[tuple[int, int, int, int]] | None:
    """An op log's op's matrix products as (b, m, K, n), from its tensor operands' shapes in training_slowdown.flops's
    order; None for an op this doesn't place, and an error for shapes that aren't its operands'. A fused attention
    kernel's row (`SDPA`, [B, H, L, E] operands) is its products as torch's FLOP formula counts them, which is how
    the record that holds it was counted: the query heads sharing a key head stacked as rows of one product."""
    s = [tuple(x) for x in shapes]
    if op in ("mm", "_int_mm", "_scaled_mm"):
        (m, K), (_, n) = s[0], s[1]
    elif op in ("addmm", "addmm_", "_addmm_activation"):
        (m, K), (_, n) = s[1], s[2]
    elif op == "bmm":
        (b, m, K), (_, _, n) = s[0], s[1]
        return [(b, m, K, n)]
    elif op in ("baddbmm", "baddbmm_"):
        (b, m, K), (_, _, n) = s[1], s[2]
        return [(b, m, K, n)]
    elif op in ("addbmm", "addbmm_"):
        (b, m, K), (_, _, n) = s[1], s[2]
        K *= b
    elif op == "mv":
        (m, K), n = s[0], 1
    elif op in ("addmv", "addmv_"):
        (m, K), n = s[1], 1
    elif op in ("dot", "vdot"):
        (K,), m, n = s[0], 1, 1
    elif op in ("addr", "addr_"):
        (m,), (n,), K = s[1], s[2], 1
    elif op in SDPA or op.removesuffix("_backward") in SDPA:
        back = op.endswith("_backward")
        (B, hq, L, E), (_, hkv, S, _), (*_, Ev) = s[1:4] if back else s[0:3]
        b, rows = B * hkv, hq // hkv * L
        scores = (b, rows, E, S)
        if not back:
            return [scores, (b, rows, S, Ev)]
        return [scores, (b, rows, Ev, S), (b, S, rows, Ev), (b, rows, S, E), (b, S, rows, E)]
    else:
        return None
    return [(1, m, K, n)]


def other_program(rows: list[dict], log: dict) -> dict | None:
    """Whether a trace's product rows are of the run whose op log is `log` (its `ops`), as far as their shapes say: None
    where every shape (m, K, n) of the rows' `PLAIN_PRODUCTS` is a product of one of the log's ops (`log_products`), in
    either orientation; else how many distinct shapes the rows run, how many of those the log never ran, how many the
    log ran, and the three largest the log never ran by their multiplications, with how many of the log's entries it
    couldn't read (`unread`) where there are any. A trace keeps no list of the tensors it
    trains, but a weight's gradient is a product with that weight's shape as m by n, so a rank, a width or a batch the
    trace took and the run didn't shows here; a program that differs from the run's only in a vector it trains, or only
    in a batched product, doesn't."""
    ran: set = set()
    unread = []
    for key in log.get("ops", {}):
        op, shapes = split_key(key)[2:4]
        try:
            prods = log_products(op, json.loads(shapes))
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            unread.append(key)
            continue
        ran |= {(m, K, n) for b, m, K, n in prods or () if b == 1}
    runs: dict = {}
    for r in rows:
        if r["op"] in PLAIN_PRODUCTS:
            s = (r["dims"]["m"], r["dims"]["k"], r["dims"]["n"])
            runs[s] = runs.get(s, 0.0) + r["count"] * r["macs"]
    miss = sorted((s for s in runs if s not in ran and s[::-1] not in ran), key=lambda s: (-runs[s], s))
    if not miss:
        return None
    return {"shapes": len(runs), "unmatched": len(miss), "ran": len(ran), "largest": [list(s) for s in miss[:3]]} | (
        {"unread": len(unread)} if unread else {})


def traced_products(p: dict | None, block: dict | None, log: dict | None = None) -> dict | None:
    """Products a record took from a trace's rows (`p`, `products_of_trace` of `block`) as this training_slowdown takes
    them from the same rows over the same traced steps (their kinds' MACs over the rows' own), so the rows are read by
    this code's rules whatever code took them first. Given the record's op log (`log`, its ops), rows with a shape the
    log never ran carry `other_program`, which the products term never prices (`work.OTHER_PROGRAM`). Products a record
    has of its own (no `kinds_macs`), and a block with no trace rows, stand."""
    if p is None or block is None or "kinds_macs" not in p:
        return p
    per = sum(r["count"] * r["macs"] for r in block["rows"])
    got = products_of_trace(block, sum(p["kinds_macs"].values()) / per if per else 1.0) | {
        k: p[k] for k in ("stand_in",) if k in p}
    other = other_program(block["rows"], log) if (log or {}).get("ops") else None
    return got if other is None else got | {"other_program": other}


#: a weight's β as a row's words name it where the run's code bits a trained weight are below it
CODE_LABEL = "β_w from the run record's code size"
#: F as a row's words name it where the subset that binds holds a backward product (`calculator.slowdown`'s figure)
SHARED_LABEL = "F = 2/√3, a layer's forward, input-gradient and weight-gradient products sharing arrays"


def he_rests_on(got: dict) -> str:
    """What a products term rests on with the subset, F, β_w, β_x and b_y it took (`calculator.slowdown`'s `figure`,
    as `bound` keeps it in its `inputs`), short, as a row publishes it: the subset of M that binds and the F it is
    charged (`SHARED_LABEL` where its products share arrays), (H) with β(k) and casts of one value as one word, the
    code-size β_w where it caps a weight; the generic `RESTS_ON_CELL` where M is empty."""
    if got.get("beta_w") is None:
        return RESTS_ON_CELL["products"]
    which = ("M charges an operand the update reaches through a low-rank bottleneck at its β_eff"
             if got["below_width"] else "M leaves out every product with operands below that β_w·β_x")
    code = f" ({CODE_LABEL})" if got.get("code_bound") else ""
    shared = SHARED_LABEL if got["F"] > 1 else "F = 1"
    return (f"theorem of record on {got['subset']} at {shared}, given (H): operands are hard at β_w = "
            f"{got['beta_w']:.4g}{code} and β_x = {got['beta_x']:.4g} bits a word ({calculator.BETA_K_LABEL}; the "
            f"casts of one value count as one word), and (E): exports are wide at b_y = {got['b_y']:g} bits an "
            f"output, premises a single counterexample refutes; {which}")


def bound(m: Measured, c: float, k: float, kinds: tuple[str, ...] = KINDS) -> dict:
    """The counting lemma over the kinds of block work that are lower bounds on their own (module docstring) at (c, k):
    each kind's term (None: no figure, its reason in `unmeasured`), the record's slowdown ≥ their max over `kinds`,
    the kind that max binds on (`binding`; None where no kind has a figure) and what each kind rests on (`rests_on`);
    and beside them the sensitivities (`SENSITIVITIES`), never in the max, reduction and update each the sum over the
    parts that measure it (None where none does; `unmeasured` names it where a part doesn't). `inputs` holds what each
    term reads beyond `m`'s fields. The products term is `calculator.slowdown`'s, before its floor, on `m`'s products
    at the record's |C|, τ and update code; `entry` is that call's whole result, the slowdown and the bound of record
    among it."""
    off = [kind for kind in kinds if kind in SENSITIVITIES]
    if off:
        raise ValueError(f"kinds {off!r}: sensitivities, never in the bound's max (they hold only for partitions split "
                         f"by tokens, or are Slowdown's token form, a target); bound() returns them under "
                         f"'sensitivities'")
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError(f"kinds {kinds!r}: some of {KINDS}")
    from . import VERSION

    total = m.gates
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    code = m.delta_bits is not None and d < c
    share = sum(p.affected_gates for p in m.parts) / total if total else 0.0
    terms, unmeasured = {}, {}
    entry = calculator.slowdown({"c_gates": c, "k_bits": k}, m.products,
                                {"gates": total, "tau_bits": m.tau_bits, "code_bits": m.delta_bits})
    terms["products"], he = entry["term"], entry["read"] | (entry["figure"] or {})
    if entry["why"] is not None:
        unmeasured["products"] = entry["why"]
    u, u_in = (capped(m.u, m.delta_bits), capped(m.u_in, m.delta_bits)) if code else (m.u, m.u_in)
    x0, a1 = bottom_share(u_in, k), room(u, k) / m.a_bits
    s_star = a1 if m.context is None else context_s_star(m.context, u, k, m.a_bits, x0, a1)
    x0r, sr = bottom_share(m.u_in, k), room(m.u, k) / m.a_bits
    if m.context is not None:
        sr = context_s_star(m.context, m.u, k, m.a_bits, x0r, sr)
    cap = max(c - d if code else 0.0, min(c, m.alpha_max * (x0r * c / m.w + sr)))
    sens = {"token_form": _token_form(m.sigma, x0, c / m.w, s_star, m.alpha_ratio),
            "decoding_form": share * c / cap if cap > 0 else (math.inf if share else 0.0)}
    blocks = [_blocks(p, c) for p in m.parts]
    split = [(p, b) for p, b in zip(m.parts, blocks) if b >= 2]
    red = [p.steps * (b - 1) * p.partial_bits for p, b in split if p.partial_bits is not None]
    upd = [p.steps * p.grad_bits for p, b in split if p.grad_bits is not None]
    for name, got, cut in (("reduction", red, "partial gradient's cut"), ("update", upd, "gradient's cut")):
        sens[name] = (c * sum(got) / (k * total) if total else 0.0) if got or not split else None
        if len(got) < len(split):
            unmeasured[name] = (f"unmeasured: a part its token work splits has no {cut} (traced before {THE_CODE} 4.4, "
                                f"or not traced)" + ("; the term sums the parts that have it" if got else ""))
    steps = [p.steps for p in m.parts]
    inputs = {"products": he,
              "token_form": {"sigma": m.sigma, "x0": x0, "s": c / m.w, "s_star": s_star, "alpha_ratio": m.alpha_ratio}
                            | ({"s_star_a1": a1} if m.context is not None else {}),
              "decoding_form": {"share": share, "delta_bits": m.delta_bits if code else None,
                                "decoder_gates": m.decoder_gates, "x0": x0r, "s_star": sr, "alpha_max": m.alpha_max,
                                "cap": cap},
              "reduction": {"blocks": blocks, "steps": steps, "partial_bits": [p.partial_bits for p in m.parts]},
              "update": {"blocks": blocks, "steps": steps, "grad_bits": [p.grad_bits for p in m.parts]}}
    on = [kind for kind in KINDS if kind in kinds and terms[kind] is not None]
    top = max(on, key=lambda kind: terms[kind]) if on else None
    return {"version": VERSION, "label": f"{THE_CODE} {VERSION}", "terms": terms,
            "kinds": [kind for kind in KINDS if kind in kinds], "max": terms[top] if top else None, "binding": top,
            "rests_on": dict(RESTS_ON), "unmeasured": unmeasured,
            "sensitivities": {name: sens[name] for name in SENSITIVITIES}, "inputs": inputs,
            "labels": dict(SENSITIVITIES), "entry": entry}


def at(m: Measured, c: float, k: float, kinds: tuple[str, ...] = KINDS) -> dict:
    """The record's slowdown at (c, k): Fit's c/(c − d) where the training fits (each part's s, w and Fit kinds under `parts`; an
    unmeasured decoder, d None, fits no block), else the slowdown `calculator.slowdown` gives (`bound`'s `entry`:
    None where the record gives none). Beside it the one bound (`bound`: its terms, its sensitivities, Slowdown's
    token form among them, and its label) and the attack side (module docstring)."""
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    delta, delta_t = (m.delta_bits, m.delta_t) if d < c else (None, None)
    slowdown_fit = c / (c - d) if d < c else math.inf
    parts, fits = [], []
    for p in m.parts:
        w = p.token_gates / p.tokens if p.tokens else math.inf
        s = c / w
        dg = p.delta_grad_bits
        checks = {"compute": delta is not None and delta + s * m.tau_bits <= k}
        if p.tokens or dg is not None:
            checks["reduction"] = dg is not None and 2 * dg <= k
            checks["update"] = delta_t is not None and dg is not None and delta_t + dg <= k
        fits.append(all(checks.values()))
        parts.append({"w": w, "s": s, "fit": checks, "fit_training": fits[-1]})
    fit = all(fits)
    x0_known = 1.0 if all(p["fit"]["compute"] for p in parts) else 0.0
    t = m.tail_share
    tail_tokens = min(math.floor(k / (m.a_bits + m.tau_bits)),
                      math.floor(c / (t * m.alpha_max)) if t * m.alpha_max else math.inf) if 0 < t < 1 else 0
    s_star_known = t * tail_tokens if x0_known < 1 else 0.0
    b = bound(m, c, k, kinds)
    return {"c": c, "k": k, "delta_counted": delta is not None, "parts": parts, "fit": fit, "rho_fit": slowdown_fit,
            "rho": slowdown_fit if fit else b["entry"]["slowdown"], "x0_known": x0_known, "s_star_known": s_star_known,
            "bound": b}


#: what a record's published slowdown rests on, by its binding kind, short (a console cell on every row; `RESTS_ON` has each kind's in
#: full): Fit's c/(c − d), a kind of the one bound (the measured kind its max takes, where that exceeds 1), `none`
#: where every kind is measured and none exceeds 1 (the record's slowdown is the floor's 1, not any kind's), or `unmeasured` (it is None)
#: where Fit fails, no measured term exceeds 1 and a kind is unmeasured
RESTS_ON_CELL = {
    "fit": "Fit's construction",
    "products": "theorem of record, given (H): operands are hard at the β_w and β_x the row's products term takes, "
                "and (E): exports are wide at its b_y, premises a single counterexample refutes",
    "none": "no kind exceeds 1",
    "unmeasured": "unmeasured",
}
#: the sensitivities a record's published slowdown carries beside it, so the others (`SENSITIVITIES`) stay in `bound`
CELL_SENSITIVITIES = ("reduction", "update")


def published(m: Measured, c: float | None = None, k: float | None = None) -> dict:
    """The record's published slowdown at (c, k), by default the console's policy (training_slowdown.policy's `CONSOLE`, one full
    session of the reference model), `at`'s: Fit's c/(c − d) where it holds; else the slowdown
    `calculator.slowdown` gives and what it says binds: `products` where the theorem's count exceeds 1, `partial`
    where a kind is unmeasured (the record's slowdown is then only a lower bound); None (binding `unmeasured`) where the record gives
    none, with `why` saying so, never 1, which would read as no slowdown; else 1, binding `none` (the count is
    computed and not above 1: the floor sets that 1). Beside it what it rests on (`RESTS_ON_CELL`; where the products
    term binds, `he_rests_on`, naming the subset of M that binds with its F, (H) with its β and (E) with its b_y),
    `products_rests_on`, the products term's own wherever it is measured (None where it isn't), the terms its max
    takes (None: no figure), the bound's `CELL_SENSITIVITIES`, and `capture_code`, the VERSION of the code that wrote
    the trace its products came from (None: no trace, or one that records none). This is the one place that rule
    lives: nothing downstream relabels its output."""
    if (c is None) != (k is None):
        raise ValueError("published: give both c and k, or neither for the console's policy")
    if c is None:
        c, k = policy.CONSOLE
    from . import VERSION

    v = at(m, c, k)
    b = v["bound"]
    slowdown, partial, why = v["rho"], False, None
    off = uncomputed(b)
    if v["fit"]:
        binding = "fit"
    elif b["entry"]["binds"] == "products":
        binding, partial = "products", bool(off)
    elif off:
        binding = "unmeasured"
        names = ", ".join(kind.replace("_", " ") for kind in off)
        why = ("no kind is measured" if len(off) == len(b["kinds"]) else f"{names} unmeasured" if b["binding"] is None
               else f"the measured kinds give at most 1; {names} unmeasured")
    else:
        binding = "none"
    he = b["inputs"]["products"]
    products = he_rests_on(he) if b["terms"]["products"] is not None else None
    return {"label": f"{THE_CODE} {VERSION}", "rho": slowdown, "binding_kind": binding,
            "rests_on": products if binding == "products" else RESTS_ON_CELL["unmeasured" if slowdown is None else binding],
            "products_rests_on": products, "partial": partial, "why": why, "terms": dict(b["terms"]),
            "sensitivities": {name: b["sensitivities"][name] for name in CELL_SENSITIVITIES},
            "capture_code": he.get("capture_code")}


def uncomputed(b: dict) -> list[str]:
    """The kinds of `bound`'s result `b` with no figure."""
    return [kind for kind in b["kinds"] if b["terms"][kind] is None]


# ---- measurement -------------------------------------------------------------------------------------------------------

STUBS = {
    "tau_bits": "stub: one token id per token block, ceil(log2 V) bits (its target is the next token's id)",
    "a_bits": "upper bound (witness): the model's width x 16, one token's bf16 residual stream",
    "u": "assumption (A1): evenly spread, u(x) = x * u(1)",
    "u_in": "assumption (A1): evenly spread, the input profile equal to u",
    "u1_bits": "upper bound (witness): u(1) = Delta, every token's path reads the whole update code",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "delta_grad_bits": "unmeasured: training Fit unchecked",
    "token_gates": "stub: every gate in a token block (sigma 1)",
}
#: a rebuild multiply-add's circuit Definition (`decoder`)
DECODER_DEFINITION = "F32Fma_v4"
DECODER = {
    "notes": "reported: the record's program's decoder gates for this checkpoint's E (the submission's notes)",
    "witness": "upper bound (witness): the harness's decoder for E, its multiply-adds rebuilding theta' from theta, in "
               "fp32 multiply-adds at the fp32 fused multiply-add Definition's gates each",
    "none": "measured: no decoder, E stores theta' directly (the harness rebuilds nothing, or E is a full or model E)",
    "unmeasured": "unmeasured: no decoder cost for E (the submission's notes give none and the harness has no rebuild "
                  "cost for it), so Fit is refused",
}
#: formats whose E stores the updated tensors themselves (`rebuild_macs` 0)
STORED = ("full", "model")
V23 = {
    "gates": "v2.3 op log in gates (no dtypes, so the gate prices' assumptions)",
    "tokens": "stub: forward MACs over the start's forward MACs per token (a v2.3 row has no token count)",
    "token_gates": "v2.3 op log's forward and backward in gates (the optimizer in no token block)",
    "affected_gates": "stub: forward products that read an updated weight (v2.3's count; understates A)",
}
TAIL = "assumption ({a}): frozen layers above changed ones, their share t of a token's affected work, {how}"
TRACED = {
    "tokens": "measured (trace): the steps' token counts times the traced step's consumed tokens over its count (the "
              "union of the positions its token-index ops read, per storage)",
    "token_gates": "measured (trace): sigma times each step's |C|, sigma the traced step's token ops (forward and "
                   "backward, the optimizer in none) over its gates",
    "affected_gates": "measured (trace): the traced step's token ops that depend on theta' over its gates, times "
                      "each step's |C|; 0 for a step before the first update",
    "delta_grad_bits": "upper bound (witness, trace): the gradient's narrowest cut, each value at the width the step "
                       "passes it (a coded message at its code's) and a reduction over the tokens at 32 bits or more "
                       "an element, plus the carried state the token ops don't read",
    "tau_bits": "measured (trace): the token inputs' positions the step reads per consumed token at ceil(log2 V) bits, "
                "and its other inputs",
    "a_bits": "upper bound (witness, trace): the narrowest cut across the forward's affected path, per consumed token",
    "u_in": "assumption (A1), from the trace: the input profile, the narrowest cut from the carried values to the update values one "
            "token's ops read so far, in program order",
    "u": "assumption (A1), from the trace: 0 up to t, then the input profile",
    "tail_share": "measured (trace): the share of a token's affected work in ops that read no update value",
}
TRACED_STRUCTURE = ("sigma", "affected_share", "token_factor", "delta_grad_bits", "partial_cut_bits", "grad_cut_bits")
PRODUCTS = ("measured: the captured steps' products of two fresh operands, each at its passed widths and gates a MAC, "
            "at the block's k")
TRACED_PRODUCTS = ("measured (trace): the traced step's products of two fresh operands, each at its passed widths and "
                   "gates a MAC, times the traced steps' tokens over the traced step's")
#: the products' approximation where a run's traced steps differ in shape (their structures, or their tokens)
STAND_IN = ("approximated: the run's traced steps differ in shape, so the dominant traced step's products stand in "
            "for all of them, scaled by tokens")
#: the capture's lists of read-backs that decide what the graph can't carry, which `refusal` reads under a row's
#: `trace`
READ_BACKS = ("data_dependent", "carried_scalars")
CONTEXT = ("charged (the context charge): every earlier token's keys and values at each layer from the lowest whose "
           "keys and values depend on theta' up, 2 x the model's width x 16 bits; h an upper bound from the start's "
           "shape and the run's changed tensors")
#: a layer's tensors its own keys and values read
KV_READS = ("attn_norm", "wk", "wv")


def context(changed, shape: dict) -> dict:
    """The context charge's quantities (module docstring) for a run whose θ' changes the tensors `changed` (the facts'
    `changed`), at its start's `shape`. κ (`kv_bits`) is 2·d_model·16: the model has no grouped K/V heads, so d_kv is
    d_model. L (`positions`) is ctx. ℓ_kv (`kv_layer`, None: no layer) is the lowest layer whose keys and values
    depend on θ': the embedding gives 0, a layer's attn_norm, wk or wv that layer, any other of its tensors the next
    one, and the final norm none. h (`free_share`) is an upper bound on the context-free share of a token's affected
    work, from forward weight MACs per token (a layer's (P − V·d − d)/n_layer, the tied head's V·d):
    - the context-free side is the layers from the lowest changed one up to ℓ_kv, all counted affected, and the head,
      at 3 times their forward (a backward is at most twice it);
    - the charged side is the layers from ℓ_kv up at twice their forward (a backward is at least their input
      gradients), less ℓ_kv's Q, K and V projections where its input depends on θ alone;
    - attention and the loss's elementwise ops are left out, which raises h at every position."""
    n, d, v = shape["n_layer"], shape["d_model"], shape["vocab"]
    low = kv = below = n
    for name in changed:
        part = name.split(".")
        if part[0] == "embed":
            low = kv = below = 0
        elif part[0] == "blocks":
            i = int(part[1])
            low, below, kv = min(low, i), min(below, i + 1), min(kv, i if part[2] in KV_READS else i + 1)
        elif part[0] != "norm":
            raise ValueError(f"changed tensor {name!r}: formalism.context places only embed, blocks.* and norm")
    layer, head = (shape["parameters"] - v * d - d) / n, v * d
    h = 1.0
    if kv < n:
        free = (kv - low) * layer + head
        h = 3 * free / (3 * free + 2 * ((n - kv) * layer - (3 * d * d if below > kv else 0)))
    return {"kv_bits": 2 * d * ACT_BITS, "positions": shape["ctx"], "kv_layer": kv if kv < n else None,
            "free_share": h}


def of_trace(f: dict) -> dict:
    """The record-level `formalism` fields of a traced step's quantities (training_slowdown.capture.circuit.measure), with every
    measured field under `trace`. t is 0 where no token op reads an update value (u_in is 0 then)."""
    t = f["tail_share"] if f["tail_share"] < 1 else 0.0
    u_in = tuple(map(tuple, f["u_in"]))
    out = {k: f[k] for k in ("tau_ids", "tau_other_bits", "u1_bits")} | {
        "u_in": [list(p) for p in u_in], "u": [list(p) for p in with_tail(u_in, t)], "tail_share": t}
    labels = {k: TRACED[k] for k in ("tau_bits", "u", "u_in", "tail_share")} | {"u1_bits": TRACED["u_in"]}
    if f.get("a_bits") is not None:
        out["a_bits"], labels["a_bits"] = f["a_bits"], TRACED["a_bits"]
    return out | {"labels": labels, "trace": f}


def _token_log(log: dict) -> dict:
    """A macs_log's token blocks' share: its forward and backward, the optimizer's (`other`) left out."""
    return {"ops": {k: v for k, v in log.get("ops", {}).items() if not k.startswith("other ")},
            "by": {k: v for k, v in log.get("by", {}).items() if not k.startswith("other/")}}


def _v23_part(row: dict, shape: dict) -> Part:
    log = row["macs_log"]
    fwd = sum(v for k, v in log.get("by", {}).items() if k.startswith("fwd/"))
    upd = {"ops": {k: v for k, v in log.get("ops", {}).items() if k.startswith("fwd updated ")},
           "by": {"fwd/updated": log.get("by", {}).get("fwd/updated", 0)}}
    return Part(of_log(log)["gates"], fwd / shape["forward_macs_per_token"], of_log(_token_log(log))["gates"],
                of_log(upd)["gates"], (row.get("annotation") or {}).get("steps", 1))


def _part(p: dict, labels: dict) -> Part:
    """A part from a block, a training_slowdown 4.0 block's `dep_gates` read as its affected gates."""
    p = dict(p)
    if "dep_gates" in p:
        p.setdefault("affected_gates", p.pop("dep_gates"))
    if "token_gates" not in p:
        p["token_gates"] = p["gates"]
        labels.setdefault("token_gates", STUBS["token_gates"])
    return Part(**p)


def complete(f: dict, row: dict, shape: dict) -> Measured:
    """`f`'s quantities (a `formalism` block, possibly partial), each missing one stubbed (`STUBS`) from the row and the
    start's shape, and a trace's products this training_slowdown's of its rows, checked against the row's op log
    (`traced_products`, `other_program`); `labels` names every quantity's source."""
    labels = dict(f.get("labels", {}))
    if f.get("parts"):
        parts = tuple(_part(p, labels) for p in f["parts"])
    else:
        parts = (_v23_part(row, shape),)
        labels |= {k: v for k, v in V23.items() if k not in labels}
    out = {}
    if "tau_bits" not in f and "tau_ids" in f:
        f = f | {"tau_bits": f["tau_ids"] * math.ceil(math.log2(shape["vocab"])) + f.get("tau_other_bits", 0.0)}
    e = (row.get("output") or {}).get("e_bytes")
    defaults = {"tau_bits": math.ceil(math.log2(shape["vocab"])), "a_bits": shape["d_model"] * ACT_BITS,
                "delta_bits": 8 * e if e is not None else None}
    for name, default in defaults.items():
        out[name] = f.get(name, default)
        if name not in f:
            labels.setdefault(name, STUBS[name] if default is not None else "unmeasured: no update code")
    out["decoder_gates"], how = decoder(f, row)
    if how is None:
        labels.setdefault("decoder_gates", "the record's block: its decoder_gates")
    else:
        labels["decoder_gates"] = how
    t = float(f.get("tail_share") or 0.0)
    if not 0 <= t < 1:
        raise ValueError(f"tail share {t} is not in [0, 1)")
    u1 = f["u1_bits"] if f.get("u1_bits") is not None else (f["u"][-1][1] if f.get("u") else out["delta_bits"])
    if u1 is None:
        raise ValueError("no update profile and no update code: the record's slowdown is unmeasured")
    u, u_in = tail(u1, t)
    for name, default, a, how in (("u", u, "A1", "done reading no update bits, the changed update even over the rest"),
                                  ("u_in", u_in, "A1", "u_in jumps to the changed layers' full update at once")):
        if f.get(name):
            out[name] = tuple(map(tuple, f[name]))
            if not math.isclose(out[name][-1][1], u1):
                raise ValueError(f"u(1) is {u1} but {name} ends at {out[name][-1][1]}")
            labels.setdefault(name, f"assumption ({a}): the record's profile")
        else:
            out[name] = default
            labels.setdefault(name, TAIL.format(a=a, how=how) if t else STUBS[name])
    if f.get("u1_bits") is None:
        labels.setdefault("u1_bits", STUBS["u1_bits"] if not f.get("u") else labels["u"])
    if any(p.delta_grad_bits is None and p.tokens for p in parts):
        labels.setdefault("delta_grad_bits", STUBS["delta_grad_bits"])
    if f.get("context"):
        labels.setdefault("context", CONTEXT)
    log = row.get("macs_log") or {}
    products = traced_products(f.get("products"), (f.get("trace") or {}).get("products"), log)
    if products is not None:
        labels.setdefault("products", PRODUCTS)
    return Measured(parts, out["tau_bits"], out["a_bits"], out["u"], out["u_in"], out["delta_bits"],
                    out["decoder_gates"], labels, t, f.get("context"), f.get("delta_t_bits"), products)


def decoder(f: dict, row: dict) -> tuple[float | None, str | None]:
    """d and its source (module docstring): the block's own `decoder_gates` (its label stands: None here), else the
    row's output's (`DECODER`)."""
    if f.get("decoder_gates") is not None:
        return float(f["decoder_gates"]), None
    out = row.get("output") or {}
    if out.get("decoder_gates") is not None:
        return float(out["decoder_gates"]), DECODER["notes"]
    if out.get("rebuild_macs"):
        return float(out["rebuild_macs"]) * gates(DECODER_DEFINITION), DECODER["witness"]
    if out.get("rebuild_macs") == 0 or out.get("format") in STORED:
        return 0.0, DECODER["none"]
    return None, DECODER["unmeasured"]


class Refused(ValueError):
    """A row whose quantities are refused, never scored (module docstring); its message is the reason."""


def _events(es: list[dict], n: int = 3) -> str:
    one = [f"{e['kind']} at {e.get('where') or 'an unknown line'}, steps {e['steps'][0]} to {e['steps'][1]}"
           for e in es[:n]]
    return "; ".join(one) + (f"; and {len(es) - n} more" if len(es) > n else "")


def _size(e) -> str:
    return "of unknown size" if e is None else f"{e} bytes"


def refusal(row: dict) -> str | None:
    """Why a traced row's measured quantities are refused (module docstring), or None: they aren't, or the row has no
    trace (its quantities are stubs or another source's, labelled)."""
    tr = (row.get("formalism") or {}).get("trace")
    if tr is None:
        return None
    missing = [k for k in READ_BACKS if k not in tr]
    if missing:
        return (f"read-backs unchecked: traced before {THE_CODE} 4.2 (no {' or '.join(missing)}), whose capture lists the "
                f"read-backs that decide what the graph can't carry; trace the run again")
    if tr["data_dependent"]:
        return f"data-dependent: not measured ({_events(tr['data_dependent'])})"
    if tr["carried_scalars"]:
        return (f"carried Python number: Delta_grad grows with the step, not measured ({_events(tr['carried_scalars'])}); "
                f"a program keeps such state as a device tensor")
    e = (row.get("output") or {}).get("e_bytes")
    none = [w for k, w in (("affected_share", "its affected share is 0"), ("u1_bits", "u(1) is 0"))
            if not tr.get(k, True)]
    if e != 0 and none:
        return f"update-free trace: E is {_size(e)}, but the traced step shows no update ({' and '.join(none)})"
    return None


def measure(row: dict, shape: dict) -> Measured:
    """A scored row's quantities: its `formalism` block (ingest's, `of_record`), else a v2.3 row's own fields, with
    stubs where it has none (`STUBS`, `V23`). Raises `Refused` for a row `refusal` refuses, and for one with a nonempty
    E whose prefix has no affected gates."""
    why = refusal(row)
    if why:
        raise Refused(why)
    m = complete(row.get("formalism") or {}, row, shape)
    e = (row.get("output") or {}).get("e_bytes")
    if e != 0 and not any(p.affected_gates for p in m.parts):
        raise Refused(f"update-free prefix: E is {_size(e)}, but no step through checkpoint {row.get('checkpoint_step')} "
                      f"reads theta' (affected gates 0)")
    return m


def of_record(record: dict, through: int | None = None) -> dict:
    """The `formalism` block ingest writes for a run record's prefix through step `through`: one part per step
    structure (its steps' gates by training_slowdown.gates, their tokens, the gates of their token blocks, its affected gates and
    Δ_grad), and the record's own `formalism` fields over these. In a record with traced structures
    (`grade.with_circuit`), a traced structure's affected gates, token blocks' gates and Δ_grad are its measured
    shares of its counted gates and its measured Δ_grad, and its tokens are scaled by its token factor (`TRACED`), a
    step before the first update's too (its Δ_grad 0, measured); an untraced structure there has no affected gates and
    no Δ_grad (Fit fails at every k). Otherwise a structure's affected gates are the record's when it gives them
    (`affected_gates`), else its updated forward nodes' MACs at the bf16 GEMM price; its token blocks' gates are its
    forward and backward (`_token_log`); its Δ_grad is its `delta_grad_bits`, else 4.0's `cut_bits`. The setup, and the
    tail of a whole run, are parts of their own, so |C| is the prefix's whole count."""
    own = dict(record.get("formalism") or {})
    labels = dict(own.get("labels", {}))
    steps = [s for s in record["steps"] if through is None or s["step"] <= through]
    groups: dict[object, list[dict]] = {}
    for name in ("setup", *(("tail",) if through is None else ())):
        if record.get(name, {}).get("by"):
            groups[name] = [record[name]]
    for s in steps:
        groups.setdefault(("step", s["struct"]), []).append(s)
    traced = any("affected_share" in (st or {}) for st in record["structures"].values())
    if traced:
        labels |= {k: TRACED[k] for k in ("tokens", "token_gates", "affected_gates", "delta_grad_bits")}
    rows = (own.get("trace") or {}).get("products") if traced else None
    traced_steps, shapes, counts = 0.0, set(), set()
    parts = []
    for gid, ss in groups.items():
        st = record["structures"].get(gid[1] if isinstance(gid, tuple) else ss[0].get("struct")) or {}
        n = len(ss)
        log = {"by": {}, "ops": {}}
        for s in ss:
            for k, v in s.get("by", {}).items():
                log["by"][k] = log["by"].get(k, 0) + v
        for row in st.get("log", []):
            key = " ".join(str(x) for x in row[:4])
            log["ops"][key] = [row[4] * n, row[5] * n]
        gates, tokens = of_log(log)["gates"], sum(s.get("tokens", 0) for s in ss)
        token_gates = of_log(_token_log(log))["gates"]
        partial = grad = None
        if traced:
            affected, dg = st.get("affected_share", 0.0) * gates, st.get("delta_grad_bits")
            partial, grad = st.get("partial_cut_bits"), st.get("grad_cut_bits")
            if rows and st.get("affected_share"):
                traced_steps += tokens / rows["tokens"]
                shapes.add(str(gid[1] if isinstance(gid, tuple) else gid).split("+traced")[0])
                counts |= {s.get("tokens", 0) for s in ss}
            tokens *= st.get("token_factor", 1.0)
            token_gates = st["sigma"] * gates if "sigma" in st else token_gates
        else:
            if "affected_gates" in st:
                affected = st["affected_gates"] * n
            else:
                affected = sum(u[2] for u in st.get("updated", [])) * n * per_mac("gemm/bf16")
                labels.setdefault("affected_gates",
                                  "stub: forward MACs with an update-dependent operand (no backward), bf16 GEMM")
            dg = st.get("delta_grad_bits", st.get("cut_bits"))
            if "delta_grad_bits" not in st and dg is not None:
                labels.setdefault("delta_grad_bits", f"measured: {THE_CODE} 4.0's cut (gradient side to the new values)")
        parts.append({"gates": gates, "tokens": tokens, "token_gates": token_gates, "affected_gates": affected,
                      "steps": n, "delta_grad_bits": dg, "partial_bits": partial, "grad_bits": grad})
    labels.setdefault("gates", "counted MACs in gates (training_slowdown.gates)")
    labels.setdefault("tokens", "measured: the steps' token counts")
    labels.setdefault("token_gates", "counted: the steps' forward and backward in gates (training_slowdown.gates)")
    if rows and "products" not in own:
        own["products"] = products_of_trace(rows, traced_steps)
        labels.setdefault("products", TRACED_PRODUCTS)
        if len(shapes) > 1 or counts != {rows["tokens"]}:
            own["products"]["stand_in"] = {"structures": sorted(shapes), "step_tokens": sorted(counts),
                                           "traced_tokens": rows["tokens"], "why": STAND_IN}
            labels["products"] = f"{labels['products']}; {STAND_IN}"
    return own | {"parts": own.get("parts") or parts, "labels": labels}
