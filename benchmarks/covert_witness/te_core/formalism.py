"""A record's slowdown ρ at a policy (c, k), by the training formalism v2 (Daniel, 2 Nov 2026, 2:09 and 2:12 PM PST;
note:training-20261102T2215Z-report-training-formalism-v2): its theorems Fit and Slowdown per record, the attack side
beside them, and the quantities Security's S = W*/H reads (te_core.frontier). |C|, c, w and d are Boolean gates
(te_core.gates), every other quantity bits.

Measured per record (`Measured`), each with its source in `labels` ("measured: ...", "counted: ...", "stub: ...",
"assumption (A1): ..." or "assumption (A1'): ..." for a profile the claim takes from an assumption, or "upper bound
(witness): ..." for a value a witness bounds from above):

- `parts`: one per distinct step structure, each with its gates |C_j|, its token blocks (one per token per step,
  `tokens`), the gates in them (`token_gates`: each token's forward and backward; the optimizer is in no token block),
  the affected gates among those, |A ∩ ∪T| (`affected_gates`: the gates that depend on θ'), and the gradient and
  optimizer state its blocks pass between them (`delta_grad_bits`, Δ_grad; None: unmeasured);
- `tau_bits` τ, the inputs outside θ a token block reads; `delta_bits` Δ, the bits that describe θ' given θ, by a
  decoder of `decoder_gates` d gates (None: unmeasured, taken as 0, a label); `delta_t_bits` Δ_t, the weights in the
  form the update rule reads and writes: a straight-through run's fp32 latents at 32 bits a value (None: Δ, for every
  other run; the coordinator's 00:47Z ruling);
- `a_bits` a, the cut cost: the computed bits a block reads to do any of a token's affected work without computing that
  token from its inputs;
- `u`, A1's update profile: u(x) the update bits a block must read to do a share x of one token's affected work, as
  points (x, bits) from (0, 0) to (1, u(1)), linear between them (two points at one x are a jump), u(1) (`u1_bits`) the
  per-token update; `u_in`, A1''s: the update bits a block must read to do the first share x, in dependency order,
  while computing the token from its inputs. Update bits are values of θ', or of an encoding of it, that a block reads,
  disjoint from the cut values A1 charges as a·m. For an evenly spread update both are x·u(1) (`even`); with frozen
  layers above changed ones, their share t of a token's affected work (`tail_share`), u is 0 up to t and u_in jumps to
  the changed layers' full update at once (`tail`): the frozen work is always charged (Daniel, 1:58 PM PST).

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

A record where Fit holds and Slowdown's bound exceeds c/(c − d) is flagged (`fit-over-slowdown`: the fitting
partition would refute A1 or A1', or a measurement is wrong).

Where the quantities come from (`measure`): a row's `formalism` block, which ingest writes from the run record
(`of_record`) or a re-score attaches (te_core.rescore), else the stubs below from a v2.3 row's own fields. te_core
4.0's capture gives `cut_bits` per step structure, read as Δ_grad; the rest (A with backward gates, τ, a, u and u_in
from the traced circuit, Δ's decoder) a run record gives as its own `formalism` block, {"parts": [...], "tau_bits",
"a_bits", "u", "u_in", "u1_bits", "tail_share", "delta_bits", "delta_t_bits", "decoder_gates", "labels"}, which
`of_record` passes through field by field.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .gates import of_log, per_mac

ACT_BITS = 16
FIT_OVER_SLOWDOWN = "fit-over-slowdown"


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
    def delta_grad_bits(self) -> float | None:
        """The largest Δ_grad of the parts that measure one (None: none does)."""
        return max((p.delta_grad_bits for p in self.parts if p.delta_grad_bits is not None), default=None)

    def to_json(self) -> dict:
        return asdict(self) | {"parts": [asdict(p) for p in self.parts], "u": [list(p) for p in self.u],
                               "u_in": [list(p) for p in self.u_in], "u1_bits": self.u1_bits}


def from_json(d: dict) -> Measured:
    return Measured(tuple(Part(**p) for p in d["parts"]), d["tau_bits"], d["a_bits"], tuple(map(tuple, d["u"])),
                    tuple(map(tuple, d["u_in"])), d.get("delta_bits"), d.get("decoder_gates"), d.get("labels", {}),
                    d.get("tail_share", 0.0), d.get("delta_t_bits"))


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


def room(u, k: float) -> float:
    """max over x in (0, 1] of x·(k − u(x)), A1's a·s*: on each linear piece a concave quadratic, so its ends and its
    vertex."""
    best = 0.0
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        xs = [x0, x1]
        if x1 > x0 and u1 > u0:
            m = (u1 - u0) / (x1 - x0)
            v = (k - u0 + m * x0) / (2 * m)
            if x0 < v < x1:
                xs.append(v)
        best = max(best, *(x * (k - u_at(u, x)) for x in xs if x > 0))
    return best


# ---- the theorems at (c, k) --------------------------------------------------------------------------------------------

def _token_form(sigma: float, x0: float, s: float, s_star: float, ratio: float) -> float:
    lead = sigma * (1 - x0) * s * ratio
    if lead == 0:
        return 0.0
    return lead / s_star if s_star > 0 else math.inf


def at(m: Measured, c: float, k: float) -> dict:
    """ρ at (c, k), with each part's s, w and Fit, Slowdown's bound and its terms, and the attack side (module
    docstring)."""
    d = m.decoder_gates or 0.0
    delta, delta_t = (m.delta_bits, m.delta_t) if d < c else (None, None)
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
    ratio = m.alpha_mean / m.alpha_max if m.alpha_max else 0.0
    x0, s_star = bottom_share(m.u_in, k), room(m.u, k) / m.a_bits
    bound = _token_form(sigma, x0, s, s_star, ratio)
    fit = all(fits)
    x0_known = 1.0 if all(p["fit"]["compute"] for p in parts) else 0.0
    t = m.tail_share
    tail_tokens = min(math.floor(k / (m.a_bits + m.tau_bits)),
                      math.floor(c / (t * m.alpha_max)) if t * m.alpha_max else math.inf) if 0 < t < 1 else 0
    s_star_known = t * tail_tokens if x0_known < 1 else 0.0
    return {"c": c, "k": k, "sigma": sigma, "w": m.w, "s": s, "x0": x0, "s_star": s_star, "alpha_ratio": ratio,
            "slowdown": bound, "delta_counted": delta is not None, "parts": parts, "fit": fit, "rho_fit": rho_fit,
            "rho": rho_fit if fit else max(1.0, bound), "x0_known": x0_known, "s_star_known": s_star_known,
            "flags": [FIT_OVER_SLOWDOWN] if fit and bound > rho_fit else []}


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
    unmeasured."""
    if m.delta_bits is None or (m.decoder_gates or 0.0) >= c:
        return None, None
    run = m.delta_bits + c / w_run * m.tau_bits
    need = []
    for p in m.parts:
        st = c * p.tokens / p.token_gates * m.tau_bits if p.tokens else 0.0
        dg = p.delta_grad_bits
        if dg is None and p.tokens:
            return run, None
        need.append(max(m.delta_bits + st, *((2 * dg, m.delta_t + dg) if dg is not None else ())))
    return run, max(need)


# ---- measurement -------------------------------------------------------------------------------------------------------

STUBS = {
    "tau_bits": "stub: one token id per token block, ceil(log2 V) bits (its target is the next token's id)",
    "a_bits": "upper bound (witness): d_model x 16, one token's bf16 residual stream",
    "u": "assumption (A1): evenly spread, u(x) = x * u(1)",
    "u_in": "assumption (A1'): evenly spread, u_in = u",
    "u1_bits": "upper bound (witness): u(1) = Delta, every token's path reads the whole update code",
    "decoder_gates": "stub: the decoder's gates d unmeasured, taken as 0 (rho 1 where Fit holds)",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "delta_grad_bits": "unmeasured: training Fit unchecked",
    "token_gates": "stub: every gate in a token block (sigma 1)",
}
V23 = {
    "gates": "v2.3 macs_log in gates (te_core.gates; no dtypes, so its assumptions)",
    "tokens": "stub: forward MACs over the start's forward MACs per token (a v2.3 row has no token count)",
    "token_gates": "v2.3 macs_log's forward and backward in gates (the optimizer in no token block)",
    "affected_gates": "stub: forward products that read an updated weight (v2.3's count; understates A)",
}
TAIL = "assumption ({a}): frozen layers above changed ones, their share t of a token's affected work, {how}"


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
    e = (row.get("output") or {}).get("e_bytes")
    defaults = {"tau_bits": math.ceil(math.log2(shape["vocab"])), "a_bits": shape["d_model"] * ACT_BITS,
                "delta_bits": 8 * e if e is not None else None, "decoder_gates": None}
    for name, default in defaults.items():
        out[name] = f.get(name, default)
        if name not in f:
            labels.setdefault(name, STUBS[name] if default is not None or name == "decoder_gates"
                              else "unmeasured: no update code")
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
    return Measured(parts, out["tau_bits"], out["a_bits"], out["u"], out["u_in"], out["delta_bits"],
                    out["decoder_gates"], labels, t, f.get("delta_t_bits"))


def measure(row: dict, shape: dict) -> Measured:
    """A ledger row's quantities: its `formalism` block (ingest's, `of_record`), else a v2.3 row's own fields, with
    stubs where it has none (`STUBS`, `V23`)."""
    return complete(row.get("formalism") or {}, row, shape)


def of_record(record: dict, through: int | None = None) -> dict:
    """The `formalism` block ingest writes for a run record's prefix through step `through`: one part per step
    structure (its steps' gates by te_core.gates, their tokens, the gates of their token blocks, its affected gates and
    Δ_grad), and the record's own `formalism` fields over these. A structure's affected gates are the record's when it
    gives them (`affected_gates`), else its updated forward nodes' MACs (te_core 4.0's capture: every forward MAC with
    an update-dependent operand; the backward's are not counted yet), at the bf16 GEMM price; its token blocks' gates
    are its forward and backward (`_token_log`); its Δ_grad is its `delta_grad_bits`, else 4.0's `cut_bits` (the
    narrowest cut between the step's gradient side and the new values it carries). The setup, and the tail of a whole
    run, are parts of their own, so |C| is the prefix's whole count, as `ledger._macs_log`'s."""
    own = dict(record.get("formalism") or {})
    labels = dict(own.get("labels", {}))
    steps = [s for s in record["steps"] if through is None or s["step"] <= through]
    groups: dict[object, list[dict]] = {}
    for name in ("setup", *(("tail",) if through is None else ())):
        if record.get(name, {}).get("by"):
            groups[name] = [record[name]]
    for s in steps:
        groups.setdefault(("step", s["struct"]), []).append(s)
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
        if "affected_gates" in st:
            affected = st["affected_gates"] * n
        else:
            affected = sum(u[2] for u in st.get("updated", [])) * n * per_mac("gemm/bf16")
            labels.setdefault("affected_gates",
                              "stub: forward MACs with an update-dependent operand (no backward), bf16 GEMM")
        dg = st.get("delta_grad_bits", st.get("cut_bits"))
        if "delta_grad_bits" not in st and dg is not None:
            labels.setdefault("delta_grad_bits", "measured: te_core 4.0's cut (gradient side to the new values)")
        parts.append({"gates": of_log(log)["gates"], "tokens": sum(s.get("tokens", 0) for s in ss),
                      "token_gates": of_log(_token_log(log))["gates"], "affected_gates": affected, "steps": n,
                      "delta_grad_bits": dg})
    labels.setdefault("gates", "counted MACs in gates (te_core.gates)")
    labels.setdefault("tokens", "measured: the steps' token counts")
    labels.setdefault("token_gates", "counted: the steps' forward and backward in gates (te_core.gates)")
    return own | {"parts": own.get("parts") or parts, "labels": labels}
