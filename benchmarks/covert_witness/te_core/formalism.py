"""A record's slowdown ρ at a policy (c, k), by the training formalism v2: its
theorems Fit and Slowdown per record, the attack side beside them, and the quantities Security's S = W*/H reads (te_core.frontier). |C|, c, w and d are Boolean gates
(te_core.gates), every other quantity bits.

Measured per record (`Measured`), each with its source in `labels` ("measured: ...", "counted: ...", "stub: ...",
"assumption (A1): ..." or "assumption (A1'): ..." for a profile the claim takes from an assumption, or "upper bound
(witness): ..." for a value a witness bounds from above):

- `parts`: one per distinct step structure, each with its gates |C_j|, its token blocks (one per token per step,
  `tokens`), the gates in them (`token_gates`: each token's forward and backward; the optimizer is in no token block),
  the affected gates among those, |A ∩ ∪T| (`affected_gates`: the gates that depend on θ'), and the gradient and
  optimizer state its blocks pass between them (`delta_grad_bits`, Δ_grad; None: unmeasured);
- `tau_bits` τ, the inputs outside θ a token block reads; `delta_bits` Δ, the bits that describe θ' given θ, by a
  decoder of `decoder_gates` d gates (`decoder`; None: unmeasured, so Δ never counts and Fit fails); `delta_t_bits`
  Δ_t, the weights in the form the update rule reads and writes: a straight-through run's fp32 latents at 32 bits a
  value (None: Δ, for every other run; the coordinator's 00:47Z ruling);
- `a_bits` a, the cut cost: the computed bits a block reads to do any of a token's affected work without computing that
  token from its inputs;
- `u`, A1's update profile: u(x) the update bits a block must read to do a share x of one token's affected work, as
  points (x, bits) from (0, 0) to (1, u(1)), linear between them (two points at one x are a jump), u(1) (`u1_bits`) the
  per-token update; `u_in`, A1''s: the update bits a block must read to do the first share x, in dependency order,
  while computing the token from its inputs. Update bits are values of θ', or of an encoding of it, that a block reads,
  disjoint from the cut values A1 charges as a·m. For an evenly spread update both are x·u(1) (`even`); with frozen
  layers above changed ones, their share t of a token's affected work (`tail_share`), u is 0 up to t and u_in jumps to
  the changed layers' full update at once (`tail`): the frozen work is always charged (Daniel, 1:58 PM PST). Where Δ
  counts, both are at most Δ (`capped`): a block can read the update code instead ("a shorter update encoding lowers
  u"), a witness the record exhibits (the checkpoint's, for every step of its prefix).

At (c, k), with s = c/w, w = token_gates / tokens (each part's for Fit; the record's for Slowdown) and
σ = token_gates / |C|:

- **Fit** (construction, checked per part): the training's three kinds of block, compute Δ + sτ ≤ k, reduction
  2Δ_grad ≤ k and update application Δ_t + Δ_grad ≤ k (Δ counts only when its decoder fits, d < c, and so does Δ_t;
  a part with no token blocks has only the compute kind). Where every part fits, the record's ρ is c/(c − d) (`rho_fit`).
- **Slowdown** (the counting lemma at B = A \\ R; stated, not yet proved in Lean): its token form with the largest
  per-token affected count α_max where token blocks vary, ρ ≥ σ(1 − x0)·(s/s*)·(ᾱ/α_max), ᾱ the mean per-token
  affected count (for identical token blocks the last factor is 1). The claim takes the assumptions' upper bounds,
  never best-known values: s* ≤ max_x x(k − u(x))/a (A1) and x0 ≤ max{x : u_in(x) ≤ k} (A1'). Where Fit fails, the
  record's ρ is max(1, that bound).
- **The attack side** (tightness, never the claim): the best-known x0 and s* from exhibited blocks. `x0_known` is 1
  where the compute block fits (it computes s tokens from their inputs), else 0; `s_star_known` is the frozen tail's
  block's, which is handed each token's state at the tail (a + τ bits) and does the tail's share t of its affected
  work reading no update bits, t·min(⌊k/(a + τ)⌋, ⌊c/(t·α_max)⌋), and 0 where there is no tail or x0_known is 1.

Where Fit holds, Δ + sτ ≤ k bounds u_in by k through the cap, so x0's bound is 1 and Slowdown's bound 0: the two never
disagree.

**The context charge** (`Measured.context`, te_core.rescore's `--charge-context`; a sensitivity for A1, off by default,
the coordinator's ask of 3 Nov). A1 charges a block a per token it serves. The charge adds, for each token whose
affected work a block does without computing the token from its inputs, at each layer the block works on for it, the
keys and values of every earlier token of the sequence at that layer that the block doesn't compute itself:
κ = 2·d_kv·16 bits per earlier token per layer, each counted once per block. Only K/V that depend on θ' are charged,
at the layers ℓ ≥ ℓ_kv (`context`). A block whose tokens are the span [q, q + m) of a sequence, doing a share x of each
one's affected work and some of it at such a layer, reads u(x) + a·m + κ·n·q ≤ k (n ≥ 1 such layers). With a ≤ κ a
cut token is cheaper than one earlier token's K/V at one layer, so its best span starts at the sequence start and
every token p it serves has a(p + 1) ≤ k − u(x). With L = ctx positions, h the share of a token's affected work at no
such layer (`free_share`) and h' = min(h, 1 − x0), Slowdown's bound becomes

    ρ ≥ σ·s·(ᾱ/α_max)·(1/L)·Σ_{p<L} [h'/max(S0, S1(p)) + (1 − x0 − h')/S1(p)],
    S1(p) = max{x(k − u(x))/a : u(x) + a(p + 1) ≤ k},   S0 = max{x(k − u(x))/a : x ≤ h},

which is the token form with s* the harmonic mean (1 − x0)/((1/L)·Σ_{p<L}[...]) (`s_star`; v2's A1 value is
`s_star_a1`). S1(p) is v2's s* while p + 1 ≤ (k − u(x*))/a at v2's optimum x*, so a record whose optimal block holds
L tokens keeps v2's ρ. S1(p) is 0 once a(p + 1) > k, so a charged share left there makes ρ infinite (s* 0). The
choices favourable to the block:
- its span starts at the sequence start;
- a token in its span counts as computed by it at every layer (its K/V are never charged), and K/V that depend on θ
  alone are free;
- u(x) and the charge are minimised separately;
- h is an upper bound (`context`) and lies outside x0;
- tokens are fractional, as in v2's room.
A block that computes a token from its inputs (x0) is not charged, and the attack side is v2's.

**The one bound** (`bound`, and `at` with `reading` A): the counting lemma over kinds K of block work, each a lower
bound on its own, ρ ≥ max_K (|B_K|/|C|)/(cap_K(k)/c): each kind's term, the max, the kind it binds on and what each
kind rests on (`RESTS_ON`). Where Fit fails ρ is max(1, that max); where Fit holds it is c/(c − d), and every term is at
most that. The kinds:
- fresh_products: products of two fresh operands, the Loomis-Whitney cap amortized over the partition, a theorem for
  the run's circuit given the standard algorithm and the widths it passes: te_core.io_bound's bound with Λ proved
  (`fresh_products`) on the run's products (`Measured.products`, which `products_of` takes from steps captured by
  capture_step or capture_fn, and `products_of_trace` from the product rows a recorded program's trace keeps from
  te_core 4.8 on (te_core.capture.products), each at the k it classifies them at, a trace's from te_core 4.12.1 on at
  every k of the console's dial too, `at`), its figure without P-cell, a premise on the partition (`P_CELL`; the
  whole-cell figure is the sensitivity fresh_cell). At a fixed k the term is linear in c, so the console's dial reads
  it as c times a number for each k it has products at (`fresh_per_c`, in te_core.panels' rho_inputs). io_bound's cap has the
  chain order's constant (its premise P-chain), which is the reference circuit's order by construction (`ORDER`:
  te_core.gates prices a GEMM MAC as a dot cell adding into one running 32-bit sum), so the term is measured on every
  run's products. Λ proved reads the step's whole-making system (io_bound's `whole_chain_of`: every product with both
  operands affected, a class by its chain length, at its least fixed point, solved exactly in feeding order where no
  cycle runs through it), each subset of the products with its own factor from the classes that feed it; the figures
  are the best subset's (io_bound's `bound`, whose subsets also drop the products fed by the classes of largest
  made-whole share), the dropped products still in |C|, and `inputs` names how many it dropped, their share of the
  counted MACs, the rule that dropped them and the classes only they read (`subset`). Before that G drops, never |C|,
  every product with an operand that reads an embedding's rows before any op makes them token-specific (tokens that
  share an id share its row, so a block that pays for it gets their products free), named the same way (`gathered`),
  and each token-indexed product's share at the positions sequences can share (`SHARED_PREFIX`, `shared_prefix`).
  Where every subset is fed by a cycle that diverges the term is unmeasured (`FRESH_VOID`), never 0, and so it is
  where an op breaks P-match,
  on a trace whose product rows predate what the term reads (`FRESH_RETRACE`). Products that come without one (a model's, a hand-written set) take io_bound's
  one-class rule, named in `inputs` (`ONE_CLASS`). It is unmeasured, with io_bound's chain figure in `inputs`,
  also for a circuit that declares a tree (none does), and on a record with no product rows. A record with an op log
  and no product
  rows (a ledger row traced before 4.8) has its log's
  (`products_of_ops`: dims, widths, θ or attention's where the log's class says so), but a log doesn't keep whether an
  operand is fresh (`LOG_UNKNOWN`), so the term is unmeasured there with the share of MACs whose freshness is unknown;
- token: A1's cap on a block's affected token work under reading A (below), A1's values from a witness family with
  matrix-product and weight-gradient tiles; unmeasured (never ∞) for a record without an op log or entries;
- decoding: a block reading the update code runs the whole decoder, so it holds at most c − d affected gates: the
  token kind with such run blocks at c − d gates (assumption (D)).
A kind's term is None where its inputs are unmeasured, with the reason (`unmeasured`), and the max is over the measured
ones (None, and so ρ where Fit fails, where none is). Beside the kinds, never in the max, the sensitivities
(`SENSITIVITIES`), each labelled with what it rests on:
- reduction: where a step's token work fills b ≥ 2 blocks by gates (b = ⌈N·w/c⌉), its b − 1 partial gradients at
  their passed width (`Part.partial_bits`) are read by blocks of k bits each, c·Σ steps·(b − 1)·partial/(k|C|)
  (assumption (R));
- update: there, the blocks applying the update read the gradient's cut (`Part.grad_bits`), the weights read once and θ
  free, c·Σ steps·γ/(k|C|) (assumption (U));
- token_form: Slowdown's token form, A1 under reading B;
- decoding_form: (D) under reading B, a block reading the update code holding at most max(c − d, min(c, α_max(x0·s +
  s*))) affected gates;
- per_gate: the token kind with each gate at the best witness that holds it, not closed under unions (below);
- fresh_cell: the fresh-products term with P-cell, every block holding whole dot cells, a premise on the partition.
(R) and (U) hold only for partitions split by tokens, not by layers or tensors (a block holding only frozen layers'
work passes no partial), reading B isn't A1's approved reading, per_gate holds only for blocks that are no union of
witnesses, and P-cell only for partitions that cut no dot cell, so they aren't lower bounds on their own; `kinds`
refuses them. Each result is labelled "te_core VERSION, reading A". `published` is the console's ρ (coordinator
08:26Z), under reading A where a row's token kind is measured (its trace's entries and an op log), else reading B's
token form, a target: Fit's c/(c − d) where it holds; else the max of the measured terms where it exceeds 1, partial
where a kind is unmeasured; else None where a kind is unmeasured; else 1, binding none (coordinator 12:20Z). Each
comes with the kind it binds on and what that rests on, short (`RESTS_ON_CELL`), and the reduction, update and
fresh_cell sensitivities.

**Reading A** (A1's approved reading): every block I reads |in(I) \\ θ| ≥ u_I + Σ_T a_T(e_T(I)), u_I the
update bits I's work reads, counted once, and a_T(e) the bits of token T's computed state where I's work on T begins,
the earlier tokens' keys and values that work reads and doesn't compute included. A1's values are the least reads any
checked witness achieves (a witness block is a set of the trace's ops, te_core.capture.circuit.check counts its gates
and reads), and the default packer's witnesses are of two sorts. Runs of one traced token's path in segments (the
trace's `entries`, which `check` counts the same): a run block holding a gate at segment s and position p holds at
most M·W_ij of a run i..j ∋ s, W_ij the run's gates a token and M (`_span_tokens`) the most tokens such a block holds:
at most C0 = its gates over W_ij, and reads U + A·M + its context ≤ k, U the run's update cut (once), A its frontier
and token ids at ⌈log2 V⌉ a token, and the context kf a position before and kb a position after p in p's sequence
outside the block's span, each read once a block. A block of c gates reads the run's update cut; where Δ counts
(d < c), one reading the code reads min(cut, Δ) and has c − d gates (c for the token kind). And matrix-product tiles
of the record's op log (te_core.tiles; `Measured.ops`): a box of one product's MACs reading only the operand
coordinates it uses, weight-gradient tiles among them; a matmul is one op in the trace, so a tile's reads are counted
from its op's dims and passed widths rather than by `check`. `packer` gives the token kind's term from both, closed
under unions: a block's work is a union of witnesses, each part reading at most the block's k bits, so it holds at
most its gates, and at most the best run block (entry costs add over tokens, so a union of runs holds no more) plus
every product's best tile at k (an array the block reads once may serve every product it is an operand of) plus
every gate no tile places. The term is c·A/(|C|·that cap), A the affected gates, and ∞ where no block holds one.
Each gate at its own best witness instead (`per_gate`, a sensitivity) is no lower bound on its own: a block holding a
product's gate for its few operand bits also holds the best run or another product's tile. The choices favourable to
the block: tokens are fractional; a block's tokens beyond p's sequence cost
A each, as if it computed their sequences' context; values computed from θ and token data alone are free (so a tile
takes either operand free); every inner coordinate may begin a sum.

Where the quantities come from (`measure`): a row's `formalism` block, which ingest writes from the run record
(`of_record`) or a re-score attaches (te_core.rescore), else the stubs below from a v2.3 row's own fields. te_core
4.0 measures them on the record's traced step (te_core.capture.circuit, `TRACED` its labels): each traced structure's
σ, affected share, token factor and Δ_grad (`TRACED_STRUCTURE`), and the record's τ (`tau_ids` token positions read
per consumed token at ⌈log2 V⌉ bits, plus `tau_other_bits`), a, u_in, u and t (`of_trace`, with every measured field
under `trace`). Δ stays 8 |E|, the submitted code: the trace's own encoding of θ' (`trace.update_cut_bits`) is the
carried state, θ' in full. A run record may also give its own block, {"parts": [...], "tau_bits", "a_bits", "u",
"u_in", "u1_bits", "tail_share", "delta_bits", "delta_t_bits", "decoder_gates", "ops", "products", "labels"}, which
`of_record` passes through field by field. Reading A's tiles read the block's `ops`, else the row's `macs_log`
(te_core.tiles). The trace's Δ_grad leaves θ' out of its carried state (a latent the step quantizes is θ' as the step reads it), so
Δ_t + Δ_grad counts the latents once.

**The decoder** (`decoder`; seed-mezo's 04:15Z finding): d is the block's `decoder_gates` where it gives one,
else the gates the record's program reports for the checkpoint's E (`output.decoder_gates`, ingest's from the
submission's `notes.decoder`), else the witness's: the harness's decoder for E, `prepare.rebuild_macs(θ, E)` fp32
multiply-adds (`output.rebuild_macs`) at F32Fma_v4's gates each. d is 0 only where no decoder exists (E stores θ'
directly: rebuild_macs 0, or a full or model E), and unmeasured (Fit refused) where the row has none of these.

**Refusals** (`refusal`, `measure` raising `Refused`; the same finding): a traced row's measured quantities are refused,
never scored, with the reason, when
- its trace lists no read-backs: it was traced before te_core 4.2, whose capture lists them (unchecked);
- its capture lists a read-back that decides what the graph can't carry (`trace.data_dependent`, te_core.capture.script:
  a Python clip or branch on a read-back loss): the update the trace shows is the stand-in's, not the run's;
- a Python number is folded over steps (`trace.carried_scalars`, an EMA of read-back estimates kept as a float): the
  trace carries every earlier step's read-back, so its Δ_grad grows with the step;
- E is not empty (Δ > 0, or E's size unknown) but the trace shows no update: its affected share is 0, or u(1) is 0.
Any row with a nonempty E whose prefix has no affected gates (no step through its checkpoint reads θ') is refused too.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import io_bound, policy, tiles
from .gates import gates, of_log, per_mac

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
    #: reading A's inputs: the trace's `entries` (te_core.capture.circuit) and `id_bits`, ⌈log2 V⌉ (None: traced before
    #: te_core 4.4, or not traced)
    entries: dict | None = None
    #: the record's ops as reading A's tiles read them (te_core.tiles' `of_log` or `of_graph`; None: no op log)
    ops: list | None = None
    #: the fresh-products kind's inputs, the run's products of two fresh operands as te_core.io_bound reads them
    #: (`products_of` or `products_of_trace`: {"k", "order", "products"}; None: no captured step and no trace product
    #: rows)
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

    def to_json(self) -> dict:
        return asdict(self) | {"parts": [asdict(p) for p in self.parts], "u": [list(p) for p in self.u],
                               "u_in": [list(p) for p in self.u_in], "u1_bits": self.u1_bits}


def from_json(d: dict) -> Measured:
    return Measured(tuple(Part(**p) for p in d["parts"]), d["tau_bits"], d["a_bits"], tuple(map(tuple, d["u"])),
                    tuple(map(tuple, d["u_in"])), d.get("delta_bits"), d.get("decoder_gates"), d.get("labels", {}),
                    d.get("tail_share", 0.0), d.get("context"), d.get("delta_t_bits"), d.get("entries"), d.get("ops"),
                    d.get("products"))


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
    """The largest x in [0, 1] with u(x) ≤ k (u non-decreasing): A1''s bound on x0, read on u_in."""
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


def _span_tokens(A: float, U: float, kf: float, kb: float, L: int | None, C0: float, k: float) -> np.ndarray:
    """Per position p of a sequence of L (one value where L is None: no context), the most tokens M of one run's work a
    block covering p holds (reading A's default packer, module docstring): it reads U once and A a token, the run's
    context for each position of p's sequence outside its span (kf each earlier one, kb each later one), and holds at
    most C0 tokens by gates. M = f + x with p's own share f ≤ 1 and x other tokens at A each (whole sequences, whose
    context the block computes, fractional), or M = 1 + n + x with n positions of p's sequence in the span, the
    cheaper side first (A − max(kf, kb) a token, then A − min(kf, kb)): reads are convex in M, so the most M under k
    lies on the last piece that fits. 0 where no block fits."""
    if L is None or not (kf or kb):
        top = (k - U) / A if A > 0 else (math.inf if U <= k else -1.0)
        return np.array([min(max(top, 0.0) if U <= k else 0.0, C0)])
    p = np.arange(L, dtype=float)
    k1, k2 = max(kf, kb), min(kf, kb)
    n1 = p if kf >= kb else L - 1 - p
    n2 = L - 1 - n1
    r0 = U + p * kf + (L - 1 - p) * kb
    with np.errstate(divide="ignore", invalid="ignore"):
        lone = np.where(r0 <= k, (k - r0) / A if A > 0 else math.inf, 0.0)
        r1 = r0 + A
        r2 = r1 + n1 * (A - k1)
        r3 = r2 + n2 * (A - k2)
        span = np.where(r3 <= k, L + ((k - r3) / A if A > 0 else math.inf),
                        np.where(r2 <= k, 1 + n1 + (k - r2) / (A - k2),
                                 np.where(r1 <= k, 1 + (k - r1) / (A - k1), 0.0)))
    return np.minimum(np.maximum(lone, np.nan_to_num(span, nan=0.0)), C0)


def _block_kinds(m: Measured, c: float, decoder: bool) -> list[tuple[float, float | None]]:
    """A run block's kinds, (its gates, the code's Δ bits it reads instead of the run's update cut; None: the cut): one
    not reading the update code has c gates; one reading it (Δ counted, d < c) has c − d (`decoder`; c without it)."""
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    kinds: list[tuple[float, float | None]] = [(c, None)]
    if m.delta_bits is not None and d < c:
        kinds.append((c - d if decoder else c, m.delta_bits))
    return kinds


def _run_caps(m: Measured, c: float, k: float, kinds: list[tuple[float, float | None]]) -> np.ndarray:
    """cap[p, s], the most affected gates of any run block (module docstring) of `kinds` (`_block_kinds`) holding a
    gate of segment s at position p (one row where the trace has no context): the max over runs i..j ∋ s and kinds of
    `_span_tokens`' M times the run's gates a token, a block reading min(cut, Δ) where it reads the code. 0 where no
    run block holds the gate."""
    e = m.entries
    W = np.asarray(e["gates"], dtype=float)
    n, L = e["segments"], e.get("positions")
    kf = np.asarray(e.get("context_fwd_bits") or [0.0] * n, dtype=float)
    kb = np.asarray(e.get("context_bwd_bits") or [0.0] * n, dtype=float)
    cw, cf, cb = (np.concatenate([[0.0], np.cumsum(x)]) for x in (W, kf, kb))
    cap = np.zeros((L if L and (kf.any() or kb.any()) else 1, n))
    for i in range(n):
        for j in range(i, n):
            w = cw[j + 1] - cw[i]
            if w <= 0:
                continue
            a = e["frontier_bits"][i][j - i] + e["ids"][i][j - i] * e["id_bits"]
            for size, delta in kinds:
                u = e["u_bits"][i][j - i] if delta is None else min(e["u_bits"][i][j - i], delta)
                t = _span_tokens(a, u, cf[j + 1] - cf[i], cb[j + 1] - cb[i], L if len(cap) > 1 else None,
                                 size / w, k)
                cap[:, i:j + 1] = np.maximum(cap[:, i:j + 1], (t * w)[:, None])
    return cap


def _antitone(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray) -> float:
    """The least E[min(X, Y)] over every joint law of X (values `a`, weights `wa`) and Y (`b`, `wb`), each law's
    weights normalised to 1: min is supermodular, so the antitone coupling, X's least values against Y's greatest,
    ∫_0^1 min(F_X⁻¹(u), F_Y⁻¹(1 − u)) du."""
    ia, ib = np.argsort(a), np.argsort(b)[::-1]
    a, b = a[ia], b[ib]
    ea, eb = np.cumsum(wa[ia]), np.cumsum(wb[ib])
    ea, eb = ea / ea[-1], eb / eb[-1]
    edges = np.unique(np.clip(np.concatenate([[0.0], ea, eb]), 0.0, 1.0))
    width = np.diff(edges)
    mid = (edges[:-1] + edges[1:]) / 2
    v = np.minimum(a[np.minimum(np.searchsorted(ea, mid), len(a) - 1)],
                   b[np.minimum(np.searchsorted(eb, mid), len(b) - 1)])
    keep = width > 0
    return float((v[keep] * width[keep]).sum())


#: the trace's segment phases (te_core.capture.circuit) as the op log's
SEGMENT_PHASE = {0: "fwd", 1: "bwd"}


def packer(m: Measured, c: float, k: float, decoder: bool = True) -> float | None:
    """The token kind under reading A (module docstring): c·A/(|C|·cap), A the affected gates and cap the most any
    block holds, its work a union of the default packer's witnesses, each part reading at most the block's k bits: per
    block kind (`_block_kinds`, with `decoder` one reading the update code at c − d gates), at most its gates and at
    most its kind's best run block (`_run_caps`: a union of run blocks holds no more than the best, since a token's
    entry costs add and the union reads every run's update cut) plus the tiles beside it (te_core.tiles.held, the log
    scaled to |C|). None (unmeasured) where the record has no op log or no entries: A1's values then come from runs
    alone, a family without matrix-product tiles. Infinite where no block holds an affected gate."""
    e, ops = m.entries, m.ops
    if e is None or ops is None:
        return None
    affected = sum(p.affected_gates for p in m.parts)
    if not m.gates or not affected:
        return 0.0
    logged = sum(op["gates"] for op in ops)
    beside = m.gates / logged * tiles.held(ops, k) if logged else 0.0
    cap = max(min(size, float(_run_caps(m, c, k, [(size, delta)]).max(initial=0.0)) + beside)
              for size, delta in _block_kinds(m, c, decoder))
    return c * affected / m.gates / cap if cap > 0 else math.inf


def per_gate(m: Measured, c: float, k: float, decoder: bool = False) -> float | None:
    """`packer`'s term with each gate at the best witness that holds it, not closed under unions (`SENSITIVITIES`):
    (c/|C|)·Σ_g 1/cap(g) over the affected gates g of token blocks, cap(g) the most affected gates of any of the
    default packer's blocks that holds g, a run block's (`_run_caps`) or a tile's (te_core.tiles.caps, at c gates: a
    tile with either operand free covers one reading the code). Which op of the log an affected gate at a segment and
    position is isn't in the trace, so per phase the term takes the joint law of the two caps least favourable to the
    bound (`_antitone`): the run caps over the phase's affected gates (spread by its segments' gates and the
    sequence's positions) against the tile caps of the phase's A·W_φ/W best-capped gates of the log (scaled to |C|;
    gates beyond the log's at c). None where `packer` is; infinite where some affected gate is in no block."""
    e, ops = m.entries, m.ops
    if e is None or ops is None:
        return None
    W = np.asarray(e["gates"], dtype=float)
    total, logged = W.sum(), sum(op["gates"] for op in ops)
    affected = sum(p.affected_gates for p in m.parts)
    if not total or not affected:
        return 0.0
    run = _run_caps(m, c, k, _block_kinds(m, c, decoder))
    with np.errstate(divide="ignore"):
        inv = 1 / run
    phase = np.asarray(e.get("phase") or [0] * e["segments"])
    scale = m.gates / logged if logged else 0.0
    out = 0.0
    for p in np.unique(phase[W > 0]):
        on = (phase == p) & (W > 0)
        share = W[on].sum() / total
        a = inv[:, on].ravel()
        wa = np.broadcast_to(W[on], inv[:, on].shape).ravel()
        name = SEGMENT_PHASE.get(int(p))
        tile = sorted(tiles.caps(ops, name, k, c), key=lambda gc: -gc[1]) if name and scale else []
        need, b, wb = affected * share, [], []
        for g, cap in tile:
            if need <= 0:
                break
            b.append(1 / cap if cap > 0 else math.inf)
            wb.append(min(g * scale, need))
            need -= wb[-1]
        if need > 0:
            b.append(1 / c)
            wb.append(need)
        out += share * _antitone(a, wa, np.asarray(b), np.asarray(wb))
    return c * affected / m.gates * out


def _blocks(p: Part, c: float) -> float:
    """The fewest blocks a step's token work of part `p` fills by gates, ⌈N·w/c⌉ (N its tokens a step)."""
    return math.ceil(p.token_gates / p.steps / c) if p.tokens else 0


#: whether the fresh-products term takes P-cell (a block holds whole dot cells), a premise on the partition, which the
#: attacker chooses: off, the term is io_bound's proved figure without it (`rho_no_cell`) and the whole-cell figure is
#: the sensitivity `fresh_cell`; on only if Daniel takes the whole-cell option on his premises card
P_CELL = False
#: what every string a panel publishes calls this code, which names no module, function or constant of it
#: (tests/test_formalism.py's code-name check)
THE_CODE = "training's slowdown code"
#: the premises on the reference circuit in words, by name, as published strings state them
PREMISES = {"P-width": "every value a block reads costs its full width",
            "P-chain": "each output is summed along one chain",
            "P-match": "elementwise steps don't fan out",
            "P-cell": "blocks hold whole multiply cells"}
P_WIDTH, P_CHAIN, P_MATCH, P_CELL_WORDS = (f"{PREMISES[x]} ({x})" for x in ("P-width", "P-chain", "P-match", "P-cell"))
_CELL = (f"given that {P_CELL_WORDS}, a premise on the partition" if P_CELL else
         f"without assuming that {P_CELL_WORDS}, a premise on the partition, which the attacker chooses (the figure "
         f"with it is published beside the term)")
#: the one bound's kinds (`bound`), each a lower bound on its own, in the order ties bind: the one resting on less first
KINDS = ("fresh_products", "token", "decoding")
#: what each kind rests on
RESTS_ON = {
    "fresh_products": "theorem for this circuit: the Loomis-Whitney cap on the multiplications a block does of "
                      "products whose two operands both depend on the update (fresh operands), amortized over the "
                      "partition, on the best subset of those products (every product still counted in |C|), each "
                      "operand and partial sum at its passed width, given these premises on the reference circuit: "
                      f"{P_WIDTH}, extended to every value a block reads while making operands whole (attention's "
                      f"and the optimizer's included); {P_MATCH}: an entry's fan-out is at most its fan-in, "
                      f"checked on the traced step; {P_CHAIN}, a chain of whole multiply cells, as {THE_CODE} prices "
                      "a matrix product; products share arrays only as a layer's forward, input-gradient and weight-gradient "
                      f"products do (P-share); and the standard multiplication algorithm; {_CELL}",
    "token": "assumption (A1, reading A) with its witness family, closed under unions: the runs of one traced token's "
             "path in segments (the trace's entries) and the matrix-product tiles of the record's op log, "
             "weight-gradient tiles among them; a block's work is a union of them, each part reading "
             "at most its k bits, so it holds at most the best run block plus every product's tile at k plus every "
             "gate no tile places; a block reading the update code at c gates",
    "decoding": "assumption (D): a block reading the update code runs the whole decoder, so it holds at most c - d "
                "gates; with A1 (reading A) and the token kind's witness family",
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
    "token_form": "sensitivity, assumption (A1, reading B): Slowdown's token form, s* the largest x(k - u(x))/a over x "
                  "and x0 from the input profile, u capped at Delta for a block reading the update code; reading B "
                  "isn't A1's approved reading",
    "decoding_form": "sensitivity, assumption (D) with A1 (reading B): a block reading the update code runs the whole "
                     "decoder, so it holds at most max(c - d, min(c, α (x0 s + s*))) affected gates, α the most "
                     "affected gates a token has in any part, x0 and s* from the uncapped profiles; reading B isn't "
                     "A1's approved reading",
    "per_gate": "sensitivity, assumption (A1, reading A) with its witness family, not closed under unions: each gate "
                "at the best witness that holds it; a block holding one gate of a product beside another product's "
                "tile or the best run holds more, for a few bits of that gate's own operands",
    "fresh_cell": f"sensitivity, assuming that {P_CELL_WORDS}: the fresh-products term's figure where every block "
                  "does, a premise on the partition, which the attacker chooses; the term is the figure without it",
}
#: the orders a circuit may declare its products' output entries summed in, each with its cap's constant; io_bound has
#: the chain's
ORDERS = ("chain", "tree")
#: the reference circuit's order: te_core.gates prices a GEMM MAC as a tensor-core dot cell over its K, each adding
#: into the running sum from a zero 32-bit word, so each output entry is summed along one chain of whole cells
#: (te_core.io_bound's docstring). P-chain is a premise on the reference circuit, true of it as te_core prices it, as
#: P-width is; a run's products take it whatever the recorded graph shows (a GEMM is one op there), and only a circuit
#: that declares another summation takes another (none does). A split-K kernel only multiplies the chains, which
#: io_bound's `outputs` counts
ORDER = "chain"
#: the first te_core whose capture writes a traced step's product rows (te_core.capture.products' `CAPTURE`)
PRODUCTS_FROM = "4.8"
#: why the fresh-products kind is unmeasured on a record with no product rows
FRESH_UNMEASURED = (f"unmeasured: the record has no product rows (a step captured on its own has them, and a recorded "
                    f"program's trace has them from {THE_CODE} {PRODUCTS_FROM} on)")
#: why the fresh-products kind is unmeasured where every subset of the counted products voids the proved Λ
FRESH_VOID = ("unmeasured: every counted product is fed by a cycle of the step's whole-making system that has no fixed "
              "point at this k (the cycle's chains, r = c k/(K^2 b), make more operand entries whole than they read, "
              "and that compounds), so Λ has no proved value on any subset; a void is never a measured 0")
#: the first te_core whose capture writes everything the term reads: the whole-making system's rows (each product's
#: `fed` and `P`, the trace's `p_match`) and which operands read a gather's rows (`gathered`)
SYSTEM_FROM = "4.12.1"
#: the positions at the start of each sequence that G leaves out of every token-indexed product (`products_of_trace`):
#: position 0 reads only its own token, so sequences that share their first token (a BOS, or the stream's own where
#: windows are cut from it) compute it identically through every layer. A longer shared prefix needs the batch's
#: tokens, which no trace keeps
SHARED_PREFIX = 1
#: the first te_core whose capture checks a step for matrix products written with elementwise ops (te_core.capture.
#: products' `elementwise_products`): a multiplication that broadcasts both its operands (an outer product, or a matrix
#: product once summed), or a gather, a multiplication and a sum by index, which no product row is, so the whole-making
#: system misses their classes. 4.12.7's check lost a broadcast at any op but a view, so a map, a copy or a
#: multiplication by a value outside the update hid it, and it took an embedding's rows for a broadcast
ELEMENTWISE_FROM = "4.12.9"


def _release(v) -> tuple[int, ...]:
    """A te_core version as a tuple (none: (0,))."""
    return tuple(int(x) for x in str(v).split(".")) if v else (0,)
#: why the fresh-products kind is unmeasured on a trace captured before `ELEMENTWISE_FROM`
FRESH_UNCHECKED = (f"unmeasured: the trace was captured before {THE_CODE} {ELEMENTWISE_FROM}'s check for matrix "
                   "products written as elementwise steps (a multiplication that broadcasts both its operands, "
                   "whatever the broadcast passes through first, or gathered rows multiplied and added by index), so it "
                   "may hold products the bound can't see; retrace the run with it or later")
#: why the fresh-products kind is unmeasured on a trace captured before `SYSTEM_FROM`
FRESH_RETRACE = (f"unmeasured: the trace's product rows predate {THE_CODE} {SYSTEM_FROM}, so they don't say which "
                 "chains make each operand whole or which products read an embedding's rows before any op makes them "
                 f"token-specific, which G drops; retrace the run with {THE_CODE} {ELEMENTWISE_FROM} or later")
#: what Λ's whole-making factor is on products that come with no whole-making system (a model's, a hand-written set):
#: io_bound's one-class rule at each subset's least k_whole, which misses any shorter chain nested below their operands
ONE_CLASS = ("one-class rule: every product's operands made whole along the shortest chain that makes any of them "
             "whole (no whole-making system given)")
#: an op log's class (te_core.tiles' ops: a macs_log's role, or a traced step's weight class) as what it says of a
#: product: its non-token operand is θ (the product is never fresh), or it is attention's, which io_bound doesn't count
LOG_KIND = {"theta": "theta", "registered": "theta"}
#: what an op log doesn't keep, which decides whether a product's operands are fresh (`products_of_ops`): a weight
#: decoded from a code or merged from a low-rank update, or an activation read from an embedding's rows, is passed
#: below its op's dtype, and the log's updated class says only that an operand depends on the run's state
LOG_UNKNOWN = ("the record's capture keeps each product's dims, its op's dtype and its op log's class, not whether "
               "its operands depend on the update or the widths and rates they're passed at, which a captured step's "
               "product rows say")


def products_of_ops(ops: list[dict]) -> dict:
    """A record's matrix products from its op log (te_core.tiles' ops: a ledger row's macs_log by `tiles.of_log`, or a
    traced step's by `tiles.of_graph`), for a capture that keeps no step `io_bound.rows_of` reads: each product
    (b, m, K, n) of an op tiles places, its MACs and output entries, its operands' widths as the log gives them and
    its partial sums' (`io_bound.ACC_BITS`), and its kind where the log decides it (`LOG_KIND`: θ where its non-token
    operand is θ, attention for a scaled-dot-product op), else unknown (`LOG_UNKNOWN`). The order is the reference
    circuit's (`ORDER`), which a log, keeping a matmul as one op, doesn't show. No product is counted (`products` is
    empty): a log never says both operands are fresh."""
    rows = []
    for op in ops:
        if not op["products"]:
            continue
        total = sum(math.prod(p) for p in op["products"])
        name = op["op"].removesuffix("_backward")
        kind = "attention" if name in tiles.SDPA else LOG_KIND.get(op["class"], "unknown")
        for b, m, K, n in op["products"]:
            macs = op["macs"] * b * m * K * n / total
            rows.append({"phase": op["phase"], "class": op["class"], "op": op["op"], "dims": [b, m, K, n],
                         "macs": macs, "outputs": macs / K, "widths": op["widths"], "b_c": io_bound.ACC_BITS,
                         "gates_per_mac": op["gates"] / op["macs"], "kind": kind})
    macs = {kind: sum(r["macs"] for r in rows if r["kind"] == kind) for kind in ("theta", "attention", "unknown")}
    total = sum(macs.values())
    return {"k": None, "order": ORDER, "source": "op log", "products": [], "rows": rows, "macs": macs,
            "unknown_share": macs["unknown"] / total if total else 0.0}


def products_of(segments, k: float | None = None, gemm_dtype: str | None = "bfloat16", order: str | None = None) -> dict:
    """`Measured.products` for a run of captured steps, `segments` [(steps, StepGraph)] as te_core.capture.capture_run
    takes them, each captured by capture_step or capture_fn: every product te_core.io_bound.rows_of counts at k (kind
    "fresh"; k the console's by default), its MACs and output entries times its steps (the bound's G and Q add over
    steps, so its blocks do too), the order its entries are summed in (`ORDERS`; None: the reference circuit's,
    `ORDER`), and the steps' whole-making system and P-match check from their product rows (te_core.capture.products,
    `system_of`), G without the products that read a gather's rows as `products_of_trace` drops them (`gathered`), and
    the matrix products the steps write with elementwise ops (`elementwise_products`, times their steps), which
    `fresh_products` refuses to price."""
    from te_core.capture import products

    if order not in (None, *ORDERS):
        raise ValueError(f"order {order!r}: one of {ORDERS}, or None for the reference circuit's ({ORDER})")
    k = float(policy.CONSOLE[1] if k is None else k)
    out, rows, broken, ew = [], [], [], {}
    fresh = dropped = macs = dropped_macs = 0
    for n, g in segments:
        block = products.rows(g, k, gemm_dtype)
        for x in block["elementwise_products"]:
            c, w = ew.get(x["op"], (0, 0))
            ew[x["op"]] = (c + n * x["count"], w + n * x["macs"])
        gathered = products.gathered_gemms(g)
        for r in io_bound.rows_of(g, gemm_dtype, k):
            if r.kind == "fresh":
                p = r.product()
                fresh += 1
                macs += n * p["macs"]
                if (r.node, r.part) in gathered:
                    dropped += 1
                    dropped_macs += n * p["macs"]
                    continue
                out.append(p | {"macs": n * p["macs"], "outputs": n * p["outputs"]})
        rows += block["rows"]
        broken += block["p_match"]
    return {"k": k, "order": order or ORDER, "products": out, "system": system_of(rows), "p_match": broken,
            "gathered": {"dropped": dropped, "of": fresh, "dropped_macs_share": dropped_macs / macs if macs else 0.0},
            "elementwise_products": [{"op": op, "count": c, "macs": w} for op, (c, w) in sorted(ew.items())]}


def product_kind(row: dict) -> str:
    """A trace's product row's kind (te_core.capture.products' rows), as te_core.io_bound.rows_of classifies a GEMM:
    unaffected, attention (two token-indexed dims among m, k and n), self (both operands one array), theta (one
    operand doesn't depend on θ′), fresh (both do and both are paid at their passed width, `io_bound.is_paid` on the
    row's rates and widths) or expansion. The row's own `paid` isn't read: before te_core 4.10 a capture compared an
    upcast operand's rate with the cast's width, so a trace's rates and widths decide."""
    if not any(row["affected"]):
        return "unaffected"
    if sum(x in row["token_dims"] for x in ("m", "k", "n")) >= 2:
        return "attention"
    if row["self"]:
        return "self"
    if not all(row["affected"]):
        return "theta"
    return "fresh" if all(io_bound.is_paid(r, w) for r, w in zip(row["rates"], row["widths"])) else "expansion"


def products_of_trace(block: dict, steps: float = 1.0) -> dict:
    """`Measured.products` from a trace's product rows (te_core.capture.products' `rows`, the trace's `products`) over
    `steps` traced steps: its fresh rows (`product_kind`) as exactly the products te_core.io_bound's bound reads, each
    row's MACs and output entries times its count and `steps`, at the k the capture classified them at, and each kind's
    MACs. Their order is the reference circuit's (`ORDER`): the trace's ``unknown`` says only that the recorded graph
    doesn't show it (`trace_order`).

    Tokens that share an id share its row, so a block that pays for a row gets every such token's products of it free:
    the bound holds on any subset of the products, so G drops (never |C|) every fresh product with an operand that
    reads a gather's rows before any op makes them token-specific (the row's `gathered`), and `gathered` says how many
    it drops of how many and their share of the fresh MACs (None: rows captured before `SYSTEM_FROM`).

    Sequences that share a prefix compute it identically through every layer (causal attention reads only the prefix
    there), so a block that pays for one sequence's prefix gets the others' free. G keeps each token-indexed fresh
    product's MACs times 1 − `SHARED_PREFIX`/n, n the sequence length (the token-indexed m, k or n of the products with
    two of them, attention's, affected or not), and its output entries whole, which only lowers the bound;
    `shared_prefix` says so (None: no such product, so no position reads another's tokens and a repeated token is
    `gathered`'s). Each product keeps its operands' `fed`,
    which `fresh_products` reads to give each subset its own whole-making factor, and `elementwise_products` is the
    trace's (None: captured before `ELEMENTWISE_FROM`, whose check an earlier list may not have passed)."""
    out, kinds = [], dict.fromkeys(io_bound.KINDS, 0.0)
    flagged = all("gathered" in r for r in block["rows"])
    seq = max((r["dims"][d] for r in block["rows"] if sum(x in r["token_dims"] for x in ("m", "k", "n")) >= 2
               for d in ("m", "k", "n") if d in r["token_dims"]), default=None)
    keep = 1 - SHARED_PREFIX / seq if seq else 1.0
    fresh = dropped = macs = dropped_macs = trimmed = 0.0
    for r in block["rows"]:
        n = r["count"] * steps
        kinds[product_kind(r)] += n * r["macs"]
        if product_kind(r) != "fresh":
            continue
        fresh += r["count"]
        macs += n * r["macs"]
        if any(r.get("gathered", ())):
            dropped += r["count"]
            dropped_macs += n * r["macs"]
            continue
        scale = keep if r["token_dims"] else 1.0
        trimmed += (1 - scale) * n * r["macs"]
        out.append({"macs": scale * n * r["macs"], "outputs": n * r["outputs"], "k_dim": r["dims"]["k"],
                    "k_whole": r["k_whole"], "b_a": r["widths"][0], "b_b": r["widths"][1], "b_c": r["b_c"],
                    "gates_per_mac": r["gates_per_mac"], "cell_gates": r["cell_gates"]}
                   | ({"fed": [list(x) for x in r["fed"]]} if "fed" in r else {}))
    order = ORDER if block["order"] == "unknown" else block["order"]
    gathered = {"dropped": int(dropped), "of": int(fresh),
                "dropped_macs_share": dropped_macs / macs if macs else 0.0} if flagged else None
    shared = {"length": SHARED_PREFIX, "sequence": seq, "kept": keep,
              "trimmed_macs_share": trimmed / macs if macs else 0.0} if seq else None
    got = {"k": float(block["k"]), "order": order, "trace_order": block["order"], "capture": block["capture"],
           "products": out, "kinds_macs": kinds, "system": system_of(block["rows"]), "p_match": block.get("p_match"),
           "gathered": gathered, "shared_prefix": shared,
           "elementwise_products": block.get("elementwise_products") if _release(block.get("capture")) >=
           _release(ELEMENTWISE_FROM) else None}
    if "at" in block:
        rest = {x: v for x, v in block.items() if x != "at"}
        got["at"] = {x: products_of_trace(rest | b, steps) for x, b in block["at"].items()}
    return got


def classified_at(p: dict, k: float) -> dict | None:
    """`Measured.products` `p` as classified at k: itself where it was classified at k, else its block at k (`at`: a
    trace's rows at each k of the console's dial, te_core.capture.products.rows_on_dial) with `p`'s stand_in; None
    where it has neither."""
    if math.isclose(p["k"], k, rel_tol=1e-12):
        return p
    return next((b | {x: p[x] for x in ("stand_in",) if x in p} for b in p.get("at", {}).values()
                 if math.isclose(b["k"], k, rel_tol=1e-12)), None)


def system_of(rows: list[dict]) -> dict | None:
    """The step's whole-making system (te_core.io_bound's `chain_classes`) from a trace's product rows: a class for
    each chain length K among the rows with both operands affected, with c (`io_bound.C_SELF` where it holds a product
    of an array with itself, else `C_PAIR`), P and the classes feeding it ([K, A] pairs), the counted rows' feeds, and
    the least width of a class's operand; None for rows captured before `SYSTEM_FROM`, which have no `fed` or no
    `gathered`. The counted rows are the fresh ones that read no gathered rows (`products_of_trace`)."""
    if any("fed" not in r or "gathered" not in r for r in rows):
        return None
    classes: dict = {}
    counted: dict = {}
    b = math.inf
    for r in rows:
        if all(r["affected"]):
            e = classes.setdefault(r["dims"]["k"], {"k": r["dims"]["k"], "c": io_bound.C_PAIR, "P": 1.0, "feeds": {}})
            e["c"] = max(e["c"], io_bound.C_SELF if r["self"] else io_bound.C_PAIR)
            e["P"] = max(e["P"], r["P"])
            for K, w in r["fed"]:
                e["feeds"][K] = max(e["feeds"].get(K, 0.0), w)
            b = min(b, *r["widths"])
        if product_kind(r) == "fresh" and not any(r["gathered"]):
            for K, w in r["fed"]:
                counted[K] = max(counted.get(K, 0.0), w)
    return {"classes": [e | {"feeds": [list(x) for x in sorted(e["feeds"].items())]} for _, e in sorted(classes.items())],
            "counted": [list(x) for x in sorted(counted.items())], "b": b if math.isfinite(b) else 16}


def whole_chain(system: dict, k: float) -> dict:
    """te_core.io_bound's `whole_chain_of` on a system `system_of` gives."""
    return io_bound.whole_chain_of({"classes": {e["k"]: {"c": e["c"], "P": e["P"], "feeds": dict(e["feeds"])}
                                                for e in system["classes"]},
                                    "counted": dict(system["counted"]), "b": system["b"]}, k)


def traced_products(p: dict | None, block: dict | None) -> dict | None:
    """Products ingest took from a trace's rows (`p`, `products_of_trace` of `block`) as this te_core takes them from
    the same rows over the same traced steps (their kinds' MACs over the rows' own), whatever te_core ingested the
    record, so the rows are read by this te_core's rules: an ingest before 4.12 kept no whole-making system, which
    `fresh_products` reads as products of no trace (`ONE_CLASS`), 4.12's kept one that counts the products reading
    gathered rows, which it would read as this te_core's, 4.12.1's kept them at the console's k only, though the trace
    keeps rows at each k of the dial (`at`, which `classified_at` reads), te_core 4.8's kept the trace's order unknown,
    which it never measures, and before 4.10 an upcast operand's product was an expansion. Taken again, rows captured
    before `SYSTEM_FROM` say they need a retrace (`FRESH_RETRACE`), rows captured before `ELEMENTWISE_FROM` that no
    capture looked at for products written elementwise say so (`FRESH_UNCHECKED`), and the rest are measured on their
    system, at each k their trace keeps rows at. Products a record has of its own (no `kinds_macs`), and a block with
    no trace rows, stand."""
    if p is None or block is None or "kinds_macs" not in p:
        return p
    per = sum(r["count"] * r["macs"] for r in block["rows"])
    return products_of_trace(block, sum(p["kinds_macs"].values()) / per if per else 1.0) | {
        k: p[k] for k in ("stand_in",) if k in p}


def fresh_products(m: Measured, c: float, k: float) -> tuple[float | None, str | None, dict]:
    """The fresh-products kind's term at (c, k), te_core.io_bound's bound with Λ proved (`io_bound.bound`) on the run's
    products (`Measured.products`) and the record's |C|, its figure without P-cell (`rho_no_cell`; with it where
    `P_CELL`); its reason where unmeasured (None: measured); and what it read, the whole-cell figure among it
    (`fresh_cell`). The products are io_bound's at the k `rows_of` classified them at, or a trace's at each k of the
    console's dial (`classified_at`), summed in the reference circuit's chains (`ORDER`, io_bound's premise P-chain)
    unless their circuit declares another order, so a term at another k, or for a circuit that declares a tree, is
    unmeasured, with io_bound's chain figure in what it read. A
    record with an op log and no product rows has its log's (`products_of_ops`), none of which a log can count, so the
    term is unmeasured with what the log leaves unknown. The products' `stand_in`, where a run's traced steps differ in
    shape (`of_record`), is in what it read, and so is `subset` where io_bound's best subset for the term dropped
    products (how many, of how many, their share of the counted MACs and the subset's least whole-making chain, and
    on a trace's rows the rule that kept it, its e and the classes only the dropped products read); where no subset has
    a proved Λ the term is unmeasured (`FRESH_VOID`). `gathered` is what G dropped before that, the products that read
    an embedding's rows before any op makes them token-specific, and `shared_prefix` what it trimmed at the sequences'
    first positions (`products_of_trace`). Products that come with a whole-making system are unmeasured where the step
    writes matrix products with elementwise ops (they are products, never elementwise work, and the system misses
    their classes) or where its capture didn't look for them (`FRESH_UNCHECKED`)."""
    p = m.products
    if p is None and m.ops is not None:
        p = products_of_ops(m.ops)
        got = {"source": p["source"], "order": p["order"], "rows": len(p["rows"]), "macs": p["macs"],
               "unknown_share": p["unknown_share"]}
        return None, (f"unmeasured: {LOG_UNKNOWN}; {p['unknown_share']:.0%} of its products' MACs have operands of "
                      f"unknown freshness (the rest a θ operand or attention's); a trace from {THE_CODE} "
                      f"{PRODUCTS_FROM} on keeps them"), got
    if p is None:
        return None, FRESH_UNMEASURED, {}
    p = classified_at(p, k) or p
    order = p.get("order") or ORDER
    got = {"k_classified": p["k"], "order": order, "products": len(p["products"])} | {
        x: p[x] for x in ("capture", "trace_order", "kinds_macs", "stand_in", "gathered", "shared_prefix") if x in p}
    if not math.isclose(p["k"], k, rel_tol=1e-12):
        return None, (f"unmeasured: the products were classified at k = {p['k']:g}, not at this k, and keep none at "
                      f"it (a trace from {THE_CODE} {SYSTEM_FROM} on keeps them at each k of the console's dial); "
                      f"take them again at it from the run's captured step or its trace"), got
    if "system" not in p:
        got["whole_making"] = ONE_CLASS
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates)
    elif p["system"] is None:
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates, math.inf)
        return None, FRESH_RETRACE, got
    elif p.get("elementwise_products") is None:
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates, math.inf)
        return None, FRESH_UNCHECKED, got
    elif p["elementwise_products"]:
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates, math.inf)
        got["elementwise_products"] = p["elementwise_products"]
        ops = ", ".join(f"{x['op']} ({x['count']}×, {x['macs']:.3g} multiplications)" for x in p["elementwise_products"])
        return None, (f"unmeasured: the traced step writes matrix products as elementwise steps (a multiplication "
                      f"that broadcasts both its operands, as an outer product or a matrix product summed afterwards "
                      f"does, or gathered rows multiplied and added by index): {ops}. They are products, which the "
                      f"bound never prices as elementwise work, and no product row says how their outputs are made "
                      f"whole; write them as matrix products (x @ W, or a sparse matrix's product) and trace the run "
                      f"again"), got
    elif p.get("p_match"):
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates, math.inf)
        got["p_match"] = p["p_match"]
        ops = ", ".join(f"{x['op']} (fan-out {x['fan_out']:.3g}, {x['count']}×)" for x in p["p_match"])
        return None, (f"unmeasured: the traced step breaks the premise that {P_MATCH}: {ops} hand one entry to "
                      f"several on the way into a product's operand"), got
    else:
        w = whole_chain(p["system"], k)
        got["whole_making"] = {"e": w["e"], "whole_chain": w["whole_chain"] if math.isfinite(w["whole_chain"]) else None,
                               "diverged": w["diverged"]}
        own = all("fed" in x for x in p["products"])
        got["io_bound"] = io_bound.bound(p["products"], c, k, m.gates, w["whole_chain"],
                                         made_whole=w["mu"] if own else None)
    if order == "chain":
        proved = got["io_bound"]["proved"]
        got["fresh_cell"] = proved["rho"]
        sub = proved["subset" if P_CELL else "subset_no_cell"]
        if sub is not None and sub["dropped"]:
            got["subset"] = {"dropped": sub["dropped"], "of": sub["dropped"] + len(sub["kept"]),
                             "dropped_macs_share": sub["dropped_macs_share"], "k_whole_min": sub["k_whole_min"]} | {
                x: sub[x] for x in ("rule", "threshold", "e", "dropped_fed") if x in sub}
        term = proved["rho" if P_CELL else "rho_no_cell"]
        return term, FRESH_VOID if term is None else None, got
    return None, (f"unmeasured: the circuit declares its products summed in a {order}, and the bound has the chain "
                  f"constant only (its premise that {P_CHAIN}); the chain figure is in the inputs, never the bound"), got


def fresh_per_c(m: Measured) -> dict[str, list[float] | None] | None:
    """The fresh-products term as the console's dial reads it: for each k the run's products are classified at (by
    its integer as a string; a trace's at each k of the console's dial, `classified_at`), the term over c without
    P-cell and with it, [rho_no_cell/c, rho/c] (`fresh_products`' term and fresh_cell where `P_CELL` is off): at a
    fixed k io_bound's blocks don't move with c, so the term at (c, k) is c times it exactly. None at a k where the
    term is unmeasured; None for a run with no product rows."""
    p = m.products
    if p is None:
        return None
    out = {}
    for k in sorted({p["k"], *(b["k"] for b in p.get("at", {}).values())}):
        term, why, got = fresh_products(m, 1.0, k)
        proved = got["io_bound"]["proved"] if why is None else None
        out[f"{k:.0f}"] = None if proved is None else [proved["rho_no_cell"], proved["rho"]]
    return out


def bound(m: Measured, c: float, k: float, kinds: tuple[str, ...] = KINDS) -> dict:
    """The counting lemma over the kinds of block work that are lower bounds on their own (module docstring) at (c, k):
    each kind's term (None: unmeasured, its reason in `unmeasured`), ρ ≥ their max over `kinds`, the kind that max
    binds on (`binding`; None where no kind is measured) and what each kind rests on (`rests_on`); and beside them the
    sensitivities (`SENSITIVITIES`), never in the max, reduction and update each the sum over the parts that measure
    it (None where none does; `unmeasured` names it where a part doesn't). `inputs` holds what each term reads beyond
    `m`'s fields."""
    off = [kind for kind in kinds if kind in SENSITIVITIES]
    if off:
        raise ValueError(f"kinds {off!r}: sensitivities, never in the bound's max (they hold only for partitions split "
                         f"by tokens, under reading B, or for blocks that are no union of witnesses); bound() returns "
                         f"them under 'sensitivities'")
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError(f"kinds {kinds!r}: some of {KINDS}")
    from . import VERSION

    total = m.gates
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    code = m.delta_bits is not None and d < c
    share = sum(p.affected_gates for p in m.parts) / total if total else 0.0
    terms, unmeasured = {}, {}
    terms["fresh_products"], why, fresh = fresh_products(m, c, k)
    if why is not None:
        unmeasured["fresh_products"] = why
    terms["token"] = packer(m, c, k, decoder=False)
    terms["decoding"] = packer(m, c, k, decoder=True) if code else terms["token"]
    if terms["token"] is None:
        why = (f"unmeasured: no entries (the run was traced before {THE_CODE} 4.4, or not traced); trace the run "
               "again" if m.entries is None else
               "unmeasured: no op log, so the family has no matrix-product tiles, and A1's values from runs alone "
               "overclaim")
        unmeasured |= {"token": why, "decoding": why, "per_gate": why}
    logged = sum(op["gates"] for op in m.ops or ())
    tok = {"share": share, "delta_bits": m.delta_bits if code else None, "decoder_gates": m.decoder_gates,
           "segments": (m.entries or {}).get("segments"), "positions": (m.entries or {}).get("positions"),
           "ops": None if m.ops is None else len(m.ops), "unplaced": None if m.ops is None else tiles.unplaced(m.ops),
           "tiles_gates": total / logged * tiles.held(m.ops, k) if logged else None}
    u, u_in = (capped(m.u, m.delta_bits), capped(m.u_in, m.delta_bits)) if code else (m.u, m.u_in)
    x0, s_star = bottom_share(u_in, k), room(u, k) / m.a_bits
    if m.context is not None:
        s_star = context_s_star(m.context, u, k, m.a_bits, x0, s_star)
    x0r, sr = bottom_share(m.u_in, k), room(m.u, k) / m.a_bits
    if m.context is not None:
        sr = context_s_star(m.context, m.u, k, m.a_bits, x0r, sr)
    cap = max(c - d if code else 0.0, min(c, m.alpha_max * (x0r * c / m.w + sr)))
    sens = {"token_form": _token_form(m.sigma, x0, c / m.w, s_star, m.alpha_ratio),
            "decoding_form": share * c / cap if cap > 0 else (math.inf if share else 0.0), "per_gate": per_gate(m, c, k),
            "fresh_cell": fresh.get("fresh_cell")}
    if "fresh_products" in unmeasured:
        unmeasured["fresh_cell"] = unmeasured["fresh_products"]
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
    inputs = {"fresh_products": fresh, "token": tok, "decoding": tok, "per_gate": tok,
              "token_form": {"sigma": m.sigma, "x0": x0, "s": c / m.w, "s_star": s_star, "alpha_ratio": m.alpha_ratio},
              "decoding_form": {"share": share, "delta_bits": tok["delta_bits"], "decoder_gates": m.decoder_gates,
                                "x0": x0r, "s_star": sr, "alpha_max": m.alpha_max, "cap": cap},
              "reduction": {"blocks": blocks, "steps": steps, "partial_bits": [p.partial_bits for p in m.parts]},
              "update": {"blocks": blocks, "steps": steps, "grad_bits": [p.grad_bits for p in m.parts]}}
    on = [kind for kind in KINDS if kind in kinds and terms[kind] is not None]
    top = max(on, key=lambda kind: terms[kind]) if on else None
    return {"version": VERSION, "label": f"{THE_CODE} {VERSION}, reading A", "terms": terms,
            "kinds": [kind for kind in KINDS if kind in kinds], "max": terms[top] if top else None, "binding": top,
            "rests_on": dict(RESTS_ON), "unmeasured": unmeasured,
            "sensitivities": {name: sens[name] for name in SENSITIVITIES}, "inputs": inputs,
            "labels": dict(SENSITIVITIES)}


def at(m: Measured, c: float, k: float, reading: str | None = None, kinds: tuple[str, ...] = KINDS) -> dict:
    """ρ at (c, k), with each part's s, w and Fit, Slowdown's bound and its terms, and the attack side (module
    docstring); with `m.context`, s* is the context charge's (`context_s_star`) and v2's is `s_star_a1`. An unmeasured
    decoder (d None) fits no block. With `reading` A, ρ where Fit fails is the one bound's, max(1, `bound`'s max over
    `kinds`; None where no kind is measured), and `bound` gives its terms and label; without it, Slowdown's token form
    alone."""
    if reading not in (None, "A"):
        raise ValueError(f"reading {reading!r}: None (Slowdown's token form) or A (the one bound); reading B's token "
                         f"form is a sensitivity of bound() (token_form), never the bound")
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    delta, delta_t = (m.delta_bits, m.delta_t) if d < c else (None, None)
    u, u_in = (m.u, m.u_in) if delta is None else (capped(m.u, delta), capped(m.u_in, delta))
    rho_fit = c / (c - d) if d < c else math.inf
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
    sigma, s = m.sigma, c / m.w
    ratio = m.alpha_ratio
    x0, s_star = bottom_share(u_in, k), room(u, k) / m.a_bits
    charged = {}
    if m.context is not None:
        charged = {"s_star_a1": s_star}
        s_star = context_s_star(m.context, u, k, m.a_bits, x0, s_star)
    slow = _token_form(sigma, x0, s, s_star, ratio)
    fit = all(fits)
    x0_known = 1.0 if all(p["fit"]["compute"] for p in parts) else 0.0
    t = m.tail_share
    tail_tokens = min(math.floor(k / (m.a_bits + m.tau_bits)),
                      math.floor(c / (t * m.alpha_max)) if t * m.alpha_max else math.inf) if 0 < t < 1 else 0
    s_star_known = t * tail_tokens if x0_known < 1 else 0.0
    one = {} if reading is None else {"bound": bound(m, c, k, kinds)}
    low = one["bound"]["max"] if one else slow
    rho = rho_fit if fit else None if low is None else max(1.0, low)
    return {"c": c, "k": k, "sigma": sigma, "w": m.w, "s": s, "x0": x0, "s_star": s_star, "alpha_ratio": ratio,
            "slowdown": slow, "delta_counted": delta is not None, "parts": parts, "fit": fit, "rho_fit": rho_fit,
            "rho": rho, "x0_known": x0_known, "s_star_known": s_star_known} | charged | one


#: what a published ρ rests on, by its binding kind, short (a console cell on every row; `RESTS_ON` has each kind's in
#: full): Fit's c/(c − d), a kind of the one bound (the measured kind its max takes, where that exceeds 1), `none`
#: where every kind is measured and none exceeds 1 (ρ is the floor's 1, not any kind's), reading B's token form (a
#: target, not a bound), or `unmeasured` (ρ None) where Fit fails, no measured term exceeds 1 and a kind is unmeasured
RESTS_ON_CELL = {
    "fit": "Fit's construction",
    "fresh_products": f"theorem for this circuit (Loomis-Whitney), given that {PREMISES['P-width']} (P-width, "
                      f"extended to what makes operands whole), {P_MATCH} and {P_CHAIN}, "
                      f"{'given' if P_CELL else 'without assuming'} that {P_CELL_WORDS}",
    "token": "assumption A1 (reading A): a block's token work is a union of runs of one token's path and "
             "matrix-product tiles",
    "decoding": "assumption D: a block reading the update code runs the whole decoder; with A1 (reading A)",
    "target": "target: A1 at reading B, not a bound",
    "none": "no kind exceeds 1",
    "unmeasured": "unmeasured",
}
#: the sensitivities a published ρ carries beside it (the coordinator's 08:28Z: reduction and update; fresh_cell, the
#: fresh-products term with P-cell), on every console row, so the others (`SENSITIVITIES`) stay in `bound`
CELL_SENSITIVITIES = ("reduction", "update", "fresh_cell")
#: whether a row whose token kind is measured (its trace's `entries` and an op log) publishes reading A (`published`)
READING_A = True


def published(m: Measured, c: float | None = None, k: float | None = None, reading: str | None = None) -> dict:
    """The published ρ at (c, k), by default the console's policy (te_core.policy's `CONSOLE`, one full session of the
    reference model) (coordinator 08:26Z), `at`'s: under reading A (`reading`, else where `READING_A` and
    the row's token kind is measured): Fit's c/(c − d) where it holds; else the max of the measured terms where it
    exceeds 1, binding that kind, `partial` where a kind is unmeasured (ρ is then only a lower bound); else, where a
    kind is unmeasured, None (binding `unmeasured`), with `why` saying so, never 1, which would read as no slowdown
    (coordinator 12:20Z); else 1, binding `none` (every kind measured, none above 1: the floor sets that 1, not any
    kind). Under reading B, Slowdown's token form, a target. Beside it what it rests on (`RESTS_ON_CELL`), the terms
    its max takes (None: unmeasured; under reading B the token form), the bound's `CELL_SENSITIVITIES`, and
    `fresh_subset`, the products the fresh-products term's best subset dropped (`fresh_products`' `subset`; None where
    it dropped none or the term is unmeasured). This is the one place that rule lives: nothing downstream relabels its
    output."""
    if (c is None) != (k is None):
        raise ValueError("published: give both c and k, or neither for the console's policy")
    if c is None:
        c, k = policy.CONSOLE
    if reading is None:
        reading = "A" if READING_A and m.entries is not None and m.ops is not None else "B"
    if reading not in ("A", "B"):
        raise ValueError(f"reading {reading!r}: A (the one bound) or B (Slowdown's token form, a target)")
    from . import VERSION

    v = at(m, c, k, "A") if reading == "A" else at(m, c, k)
    b = v["bound"] if reading == "A" else bound(m, c, k)
    rho, partial, why = v["rho"], False, None
    off = [kind for kind in b["kinds"] if b["terms"][kind] is None]
    if v["fit"]:
        binding = "fit"
    elif reading == "B":
        binding = "target"
    elif b["binding"] is not None and b["max"] > 1:
        rho, binding, partial = b["max"], b["binding"], bool(off)
    elif off:
        rho, binding = None, "unmeasured"
        why = (f"the measured kinds give at most 1; {', '.join(kind.replace('_', ' ') for kind in off)} unmeasured"
               if b["binding"] is not None else "no kind is measured")
    else:
        rho, binding = 1.0, "none"
    terms = dict(b["terms"]) if reading == "A" else {"token_form": v["slowdown"]}
    return {"reading": reading, "label": f"{THE_CODE} {VERSION}, reading {reading}", "rho": rho,
            "binding_kind": binding, "rests_on": RESTS_ON_CELL["unmeasured" if rho is None else binding],
            "partial": partial, "why": why,
            "terms": terms,
            "sensitivities": {name: b["sensitivities"][name] for name in CELL_SENSITIVITIES},
            "fresh_subset": b["inputs"]["fresh_products"].get("subset")}


def rhos(m: Measured, cells: dict[str, tuple[float, float]]) -> tuple[dict[str, float], dict[str, bool], dict]:
    """ρ in each cell of `cells` (key -> (c, k)), whether Fit holds there for the training, and Slowdown's terms there
    (x0, s* and the attack side's)."""
    rho, fit, terms = {}, {}, {}
    for key, (c, k) in cells.items():
        v = at(m, c, k)
        rho[key], fit[key] = v["rho"], v["fit"]
        terms[key] = {n: v[n] for n in ("x0", "s_star", "x0_known", "s_star_known")}
    return rho, fit, terms


def fit_k(m: Measured, c: float, w_run: float) -> tuple[float | None, float | None]:
    """The least k at which Fit holds at c, for running θ' (one block per s = c/w_run tokens of the served model,
    `w_run` its gates per token) and for the training (each part's three kinds of block); None: at no k, or
    unmeasured (Δ, or the decoder)."""
    if m.delta_bits is None or m.decoder_gates is None or m.decoder_gates >= c:
        return None, None
    run = m.delta_bits + c / w_run * m.tau_bits
    need = []
    for p in m.parts:
        st = c * p.tokens / p.token_gates * m.tau_bits if p.tokens else 0.0
        dg = p.delta_grad_bits
        if dg is None and p.tokens:
            return run, None
        need.append(max((m.delta_bits + st, *((2 * dg, m.delta_t + dg) if dg is not None else ()))))
    return run, max(need)


# ---- measurement -------------------------------------------------------------------------------------------------------

STUBS = {
    "tau_bits": "stub: one token id per token block, ceil(log2 V) bits (its target is the next token's id)",
    "a_bits": "upper bound (witness): the model's width x 16, one token's bf16 residual stream",
    "u": "assumption (A1): evenly spread, u(x) = x * u(1)",
    "u_in": "assumption (A1'): evenly spread, the input profile equal to u",
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
#: formats whose E stores the updated tensors themselves (`prepare.rebuild_macs` 0)
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
    "u_in": "assumption (A1'), from the trace: the narrowest cut from the carried values to the update values one "
            "token's ops read so far, in program order",
    "u": "assumption (A1), from the trace: 0 up to t, then the input profile",
    "tail_share": "measured (trace): the share of a token's affected work in ops that read no update value",
}
TRACED_STRUCTURE = ("sigma", "affected_share", "token_factor", "delta_grad_bits", "partial_cut_bits", "grad_cut_bits")
ENTRIES = ("assumption (A1, reading A), from the trace: one token's path in segments, what a run of them reads (its "
           "frontier, token positions, update cut and context per other position), the least any of the default "
           "packer's blocks reads")
OPS = "counted: the row's op log as matrix products, each op's gates at its kind's price, reading A's tiles"
PRODUCTS = ("measured: the captured steps' products of two fresh operands, each at its passed widths and gates a MAC, "
            "at the block's k")
TRACED_PRODUCTS = ("measured (trace): the traced step's products of two fresh operands, each at its passed widths and "
                   "gates a MAC, times the traced steps' tokens over the traced step's")
#: the products' approximation where a run's traced steps differ in shape (their structures, or their tokens)
STAND_IN = ("approximated: the run's traced steps differ in shape, so the dominant traced step's products stand in "
            "for all of them, scaled by tokens")
#: the capture's lists of read-backs that decide what the graph can't carry (te_core.capture.script), which `refusal`
#: reads under a row's `trace`
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
    """The record-level `formalism` fields of a traced step's quantities (te_core.capture.circuit.measure), with every
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
    """A part from a block, a te_core 4.0 block's `dep_gates` read as its affected gates."""
    p = dict(p)
    if "dep_gates" in p:
        p.setdefault("affected_gates", p.pop("dep_gates"))
    if "token_gates" not in p:
        p["token_gates"] = p["gates"]
        labels.setdefault("token_gates", STUBS["token_gates"])
    return Part(**p)


def complete(f: dict, row: dict, shape: dict) -> Measured:
    """`f`'s quantities (a `formalism` block, possibly partial), each missing one stubbed (`STUBS`) from the row and the
    start's shape, and a trace's products this te_core's of its rows (`traced_products`); `labels` names every
    quantity's source."""
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
        raise ValueError("no update profile and no update code: rho is unmeasured")
    u, u_in = tail(u1, t)
    for name, default, a, how in (("u", u, "A1", "done reading no update bits, the changed update even over the rest"),
                                  ("u_in", u_in, "A1'", "u_in jumps to the changed layers' full update at once")):
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
    entries = f.get("entries") or (f.get("trace") or {}).get("entries")
    if entries is not None:
        entries = entries | {"id_bits": math.ceil(math.log2(shape["vocab"]))}
        labels.setdefault("entries", ENTRIES)
    ops, log = f.get("ops"), row.get("macs_log") or {}
    if ops is None and log.get("ops"):
        ops = tiles.of_log(log)
        labels.setdefault("ops", OPS)
    products = traced_products(f.get("products"), (f.get("trace") or {}).get("products"))
    if products is not None:
        labels.setdefault("products", PRODUCTS)
    return Measured(parts, out["tau_bits"], out["a_bits"], out["u"], out["u_in"], out["delta_bits"],
                    out["decoder_gates"], labels, t, f.get("context"), f.get("delta_t_bits"), entries, ops,
                    products)


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
        return (f"read-backs unchecked: traced before te_core 4.2 (no {' or '.join(missing)}), whose capture lists the "
                f"read-backs that decide what the graph can't carry; trace the run again")
    if tr["data_dependent"]:
        return f"data-dependent: not measured ({_events(tr['data_dependent'])})"
    if tr["carried_scalars"]:
        return (f"carried Python number: Delta_grad grows with the step, not measured ({_events(tr['carried_scalars'])}); "
                f"a program keeps such state as a device tensor")
    e = (row.get("output") or {}).get("e_bytes")
    none = [w for k, w in (("affected_share", "its affected share is 0"), ("u1_bits", "u(1) is 0")) if not tr.get(k)]
    if e != 0 and none:
        return f"update-free trace: E is {_size(e)}, but the traced step shows no update ({' and '.join(none)})"
    return None


def measure(row: dict, shape: dict) -> Measured:
    """A ledger row's quantities: its `formalism` block (ingest's, `of_record`), else a v2.3 row's own fields, with
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
    structure (its steps' gates by te_core.gates, their tokens, the gates of their token blocks, its affected gates and
    Δ_grad), and the record's own `formalism` fields over these. In a record with traced structures
    (`annotate.with_structures`), a traced structure's affected gates, token blocks' gates and Δ_grad are its measured
    shares of its counted gates and its measured Δ_grad, and its tokens are scaled by its token factor (`TRACED`), a
    step before the first update's too (its Δ_grad 0, measured); an untraced structure there has no affected gates and
    no Δ_grad (Fit fails at every k). Otherwise a structure's affected gates are the record's when it gives them
    (`affected_gates`), else its updated forward nodes' MACs at the bf16 GEMM price; its token blocks' gates are its
    forward and backward (`_token_log`); its Δ_grad is its `delta_grad_bits`, else 4.0's `cut_bits`. The setup, and the
    tail of a whole run, are parts of their own, so |C| is the prefix's whole count, as `ledger._macs_log`'s."""
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
                labels.setdefault("delta_grad_bits", "measured: te_core 4.0's cut (gradient side to the new values)")
        parts.append({"gates": gates, "tokens": tokens, "token_gates": token_gates, "affected_gates": affected,
                      "steps": n, "delta_grad_bits": dg, "partial_bits": partial, "grad_bits": grad})
    labels.setdefault("gates", "counted MACs in gates (te_core.gates)")
    labels.setdefault("tokens", "measured: the steps' token counts")
    labels.setdefault("token_gates", "counted: the steps' forward and backward in gates (te_core.gates)")
    if rows and "products" not in own:
        own["products"] = products_of_trace(rows, traced_steps)
        labels.setdefault("products", TRACED_PRODUCTS)
        if len(shapes) > 1 or counts != {rows["tokens"]}:
            own["products"]["stand_in"] = {"structures": sorted(shapes), "step_tokens": sorted(counts),
                                           "traced_tokens": rows["tokens"], "why": STAND_IN}
            labels["products"] = f"{labels['products']}; {STAND_IN}"
    return own | {"parts": own.get("parts") or parts, "labels": labels}
