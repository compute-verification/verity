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

Where the quantities come from (`measure`): a row's `formalism` block, which ingest writes from the run record
(`of_record`) or a re-score attaches (te_core.rescore), else the stubs below from a v2.3 row's own fields. te_core
4.0 measures them on the record's traced step (te_core.capture.circuit, `TRACED` its labels): each traced structure's
σ, affected share, token factor and Δ_grad (`TRACED_STRUCTURE`), and the record's τ (`tau_ids` token positions read
per consumed token at ⌈log2 V⌉ bits, plus `tau_other_bits`), a, u_in, u and t (`of_trace`, with every measured field
under `trace`). Δ stays 8 |E|, the submitted code: the trace's own encoding of θ' (`trace.update_cut_bits`) is the
carried state, θ' in full. A run record may also give its own block, {"parts": [...], "tau_bits", "a_bits", "u",
"u_in", "u1_bits", "tail_share", "delta_bits", "delta_t_bits", "decoder_gates", "labels"}, which `of_record` passes
<<<<<<< HEAD
through field by field. The trace's Δ_grad leaves θ' out of its carried state (a latent the step quantizes is θ' as the
step reads it), so Δ_t + Δ_grad counts the latents once.
=======
through field by field.

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
>>>>>>> 6a0d0067ee
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

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
                    d.get("tail_share", 0.0), d.get("context"), d.get("delta_t_bits"))


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


def at(m: Measured, c: float, k: float) -> dict:
    """ρ at (c, k), with each part's s, w and Fit, Slowdown's bound and its terms, and the attack side (module
    docstring); with `m.context`, s* is the context charge's (`context_s_star`) and v2's is `s_star_a1`. An unmeasured
    decoder (d None) fits no block."""
    d = m.decoder_gates if m.decoder_gates is not None else math.inf
    delta, delta_t = (m.delta_bits, m.delta_t) if d < c else (None, None)
    u, u_in = (m.u, m.u_in) if delta is None else (capped(m.u, delta), capped(m.u_in, delta))
    rho_fit = c / (c - d) if d < c else math.inf
    parts, fits = [], []
    for p in m.parts:
        w = p.token_gates / p.tokens if p.tokens else math.inf
        s = c / w
        dg = p.delta_grad_bits
        kinds = {"compute": delta is not None and delta + s * m.tau_bits <= k}
        if p.tokens or dg is not None:
            kinds["reduction"] = dg is not None and 2 * dg <= k
            kinds["update"] = delta_t is not None and dg is not None and delta_t + dg <= k
        fits.append(all(kinds.values()))
        parts.append({"w": w, "s": s, "fit": kinds, "fit_training": fits[-1]})
    sigma, s = m.sigma, c / m.w
    ratio = m.alpha_ratio
    x0, s_star = bottom_share(u_in, k), room(u, k) / m.a_bits
    charged = {}
    if m.context is not None:
        charged = {"s_star_a1": s_star}
        s_star = context_s_star(m.context, u, k, m.a_bits, x0, s_star)
    bound = _token_form(sigma, x0, s, s_star, ratio)
    fit = all(fits)
    x0_known = 1.0 if all(p["fit"]["compute"] for p in parts) else 0.0
    t = m.tail_share
    tail_tokens = min(math.floor(k / (m.a_bits + m.tau_bits)),
                      math.floor(c / (t * m.alpha_max)) if t * m.alpha_max else math.inf) if 0 < t < 1 else 0
    s_star_known = t * tail_tokens if x0_known < 1 else 0.0
    return {"c": c, "k": k, "sigma": sigma, "w": m.w, "s": s, "x0": x0, "s_star": s_star, "alpha_ratio": ratio,
            "slowdown": bound, "delta_counted": delta is not None, "parts": parts, "fit": fit, "rho_fit": rho_fit,
            "rho": rho_fit if fit else max(1.0, bound), "x0_known": x0_known, "s_star_known": s_star_known} | charged


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
    "a_bits": "upper bound (witness): d_model x 16, one token's bf16 residual stream",
    "u": "assumption (A1): evenly spread, u(x) = x * u(1)",
    "u_in": "assumption (A1'): evenly spread, u_in = u",
    "u1_bits": "upper bound (witness): u(1) = Delta, every token's path reads the whole update code",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "delta_grad_bits": "unmeasured: training Fit unchecked",
    "token_gates": "stub: every gate in a token block (sigma 1)",
}
#: a rebuild multiply-add's circuit Definition (`decoder`)
DECODER_DEFINITION = "F32Fma_v4"
DECODER = {
    "notes": "reported: the record's program's decoder gates for this checkpoint's E (the submission's notes.decoder)",
    "witness": f"upper bound (witness): prepare.rebuild_macs(theta, E), the harness's decoder for E, in fp32 "
               f"multiply-adds at {DECODER_DEFINITION}'s gates each",
    "none": "measured: no decoder, E stores theta' directly (prepare.rebuild_macs 0, or a full or model E)",
    "unmeasured": "unmeasured: no decoder cost for E (no notes.decoder, no rebuild_macs), so Fit is refused",
}
#: formats whose E stores the updated tensors themselves (`prepare.rebuild_macs` 0)
STORED = ("full", "model")
V23 = {
    "gates": "v2.3 macs_log in gates (te_core.gates; no dtypes, so its assumptions)",
    "tokens": "stub: forward MACs over the start's forward MACs per token (a v2.3 row has no token count)",
    "token_gates": "v2.3 macs_log's forward and backward in gates (the optimizer in no token block)",
    "affected_gates": "stub: forward products that read an updated weight (v2.3's count; understates A)",
}
TAIL = "assumption ({a}): frozen layers above changed ones, their share t of a token's affected work, {how}"
TRACED = {
    "tokens": "measured (trace): the steps' token counts times the traced step's consumed tokens over its count (the "
              "union of the positions its token-index ops read, per storage)",
    "token_gates": "measured (trace): sigma |C_j|, sigma the traced step's token ops (forward and backward, the "
                   "optimizer in none) over its gates",
    "affected_gates": "measured (trace): the traced step's token ops that depend on theta' over its gates, times "
                      "|C_j|; 0 for a step before the first update",
    "delta_grad_bits": "upper bound (witness, trace): the gradient's narrowest cut at 32 bits or more an element, plus "
                       "the carried state the token ops don't read",
    "tau_bits": "measured (trace): the token inputs' positions the step reads per consumed token at ceil(log2 V) bits, "
                "and its other inputs",
    "a_bits": "upper bound (witness, trace): the narrowest cut across the forward's affected path, per consumed token",
    "u_in": "assumption (A1'), from the trace: the narrowest cut from the carried values to the update values one "
            "token's ops read so far, in program order",
    "u": "assumption (A1), from the trace: 0 up to t, then u_in",
    "tail_share": "measured (trace): the share of a token's affected work in ops that read no update value",
}
TRACED_STRUCTURE = ("sigma", "affected_share", "token_factor", "delta_grad_bits")
#: the capture's lists of read-backs that decide what the graph can't carry (te_core.capture.script), which `refusal`
#: reads under a row's `trace`
READ_BACKS = ("data_dependent", "carried_scalars")
CONTEXT = ("charged (--charge-context): every earlier token's keys and values at each layer from l_kv up, "
           "2 d_model x 16 bits; h an upper bound from the start's shape and the run's changed tensors "
           "(formalism.context)")
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
    start's shape; `labels` names every quantity's source."""
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
    return Measured(parts, out["tau_bits"], out["a_bits"], out["u"], out["u_in"], out["delta_bits"],
                    out["decoder_gates"], labels, t, f.get("context"), f.get("delta_t_bits"))


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
    shares of its counted gates and its measured Δ_grad, and its tokens are scaled by its token factor (`TRACED`); any
    other structure there (a step before the first update) has no affected gates. Otherwise a structure's affected
    gates are the record's when it gives them (`affected_gates`), else its updated forward nodes' MACs at the bf16 GEMM
    price; its token blocks' gates are its forward and backward (`_token_log`); its Δ_grad is its `delta_grad_bits`,
    else 4.0's `cut_bits`. The setup, and the tail of a whole run, are parts of their own, so |C| is the prefix's whole
    count, as `ledger._macs_log`'s."""
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
        if traced:
            affected, dg = st.get("affected_share", 0.0) * gates, st.get("delta_grad_bits")
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
                      "steps": n, "delta_grad_bits": dg})
    labels.setdefault("gates", "counted MACs in gates (te_core.gates)")
    labels.setdefault("tokens", "measured: the steps' token counts")
    labels.setdefault("token_gates", "counted: the steps' forward and backward in gates (te_core.gates)")
    return own | {"parts": own.get("parts") or parts, "labels": labels}
