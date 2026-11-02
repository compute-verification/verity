"""A record's slowdown ρ at a policy (c, k), by the training formalism v1 (Daniel, 2 Nov 2026;
note:training-20261102T1817Z-report-training-formalism-v1, with its 10:58 AM PST corrections): its theorems Fit and
Slowdown per record, and the quantities Security's S = W*/H reads (te_core.frontier). |C|, c and w are Boolean gates
(te_core.gates), every other quantity bits.

Measured per record (`Measured`), each with its source in `labels` ("measured: ...", "counted: ...", "stub: ...", or
"upper bound (witness): ..." for a value a witness bounds from above):

- `parts`: one per distinct step structure, each with its gates |C_j|, its token blocks (one per token per step,
  `tokens`), its dependent gates in token blocks |Dep ∩ ∪T| (`dep_gates`) and the gradient and optimizer state its
  blocks pass between them (`delta_grad_bits`, Δ_grad; None: unmeasured);
- `tau_bits` τ, the unregistered input bits a token block reads; `delta_bits` Δ, Fit's total update, the bits that
  describe θ' given θ, by a decoder of `decoder_gates` (None: not measured);
- A1's parameters, Slowdown's: `a_bits` a, the cut cost, the bits that hand a block one token's state mid-path;
  `u`, the update profile from the token's start, u(x) the update bits a block reads to do the first share x of a
  token's dependent work, as points (x, bits) from (0, 0) to (1, u(1)), linear between them, with u(1) (`u1_bits`) the
  per-token update; and `tail_share` t, the share of that work in a frozen tail above the highest update (0: none),
  which a block can do reading no update bits, so the whole profile is 0 up to t, then u (`with_tail`). Each is the
  best-known value from witnesses (an encoding or activation code a run exhibits), an upper bound, never a measured
  truth. u(1) and Δ are separate fields: they are equal for an update every token's path reads in full (a dense
  model), not in general.

At (c, k), per part: w = |C_j| / tokens, s = c / w, φ = dep_gates / |C_j|, and

- **Fit** (checked): Δ + sτ ≤ k (running the updated model), and for training also 2Δ_grad + sτ ≤ k; the part's
  ρ is 1 when both hold. Δ counts only when its decoder fits in c gates (an unmeasured decoder is assumed to, a label).
- **Slowdown** (a target, until Lean proves it): ρ ≳ φ(1 − x₀)·s·a / max_x x(k − u(x)), the room max_x x(k − u(x))
  over the whole profile. Its frozen-tail case has two readings (`READINGS`, the coordinator's 20:18Z; Daniel decides
  which the theorem means, so every score carries both and neither is the headline), which agree when t = 0:
  - `literal-x0`: x₀ the largest x with u(x) ≤ k on the whole profile, so x₀ ≥ t and (1 − x₀) exempts the tail's work
    from any charge: ρ ≈ φ(1 − t)·s·a / (t·k) for u(1) ≫ k;
  - `tail-charged`: x₀ the share a block can do from the token's start within k (the from-start `u`, the tail not in
    it), so the tail's work is charged through a (a block doing only the tail must be handed each token's state after
    the update, A1's a·m term): ρ ≈ φ(1 − k/u(1))·s·a / (t·k).

The part's ρ is 1 when training fits, else max(1, Slowdown's); a part where both fit and Slowdown's bound exceeds 1 is
flagged (`fit-over-slowdown`: the fitting partition would refute A1, or the measurement is wrong). The record's ρ is the
parts' |C_j|-weighted mean, so |C|ρ is the sum of their blocks' gates.

Where the quantities come from (`measure`): a row's `formalism` block, which ingest writes from the run record
(`of_record`) or a re-score attaches (te_core.rescore), else the stubs below from a v2.3 row's own fields. te_core
4.0's capture gives `cut_bits` per step structure, read as Δ_grad; the rest of what the formalism measures (Dep with
backward gates, τ, a and u from the traced circuit, Δ's decoder) a run record gives as its own `formalism` block,
{"parts": [...], "tau_bits", "a_bits", "u", "u1_bits", "tail_share", "delta_bits", "decoder_gates", "labels"}, which
`of_record` passes through field by field.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .gates import of_log, per_mac

ACT_BITS = 16
FIT_OVER_SLOWDOWN = "fit-over-slowdown"
LITERAL, TAIL_CHARGED = "literal-x0", "tail-charged"
READINGS = (LITERAL, TAIL_CHARGED)


@dataclass(frozen=True)
class Part:
    gates: float
    tokens: float
    dep_gates: float
    steps: int = 1
    delta_grad_bits: float | None = None


@dataclass(frozen=True)
class Measured:
    parts: tuple[Part, ...]
    tau_bits: float
    a_bits: float
    u: tuple[tuple[float, float], ...]
    delta_bits: float | None
    decoder_gates: float | None = None
    labels: dict = field(default_factory=dict)
    tail_share: float = 0.0

    @property
    def gates(self) -> float:
        return sum(p.gates for p in self.parts)

    @property
    def tokens(self) -> float:
        return sum(p.tokens for p in self.parts)

    @property
    def phi(self) -> float:
        return sum(p.dep_gates for p in self.parts) / self.gates if self.gates else 0.0

    @property
    def w(self) -> float:
        return self.gates / self.tokens if self.tokens else math.inf

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
                               "u1_bits": self.u1_bits}


def from_json(d: dict) -> Measured:
    return Measured(tuple(Part(**p) for p in d["parts"]), d["tau_bits"], d["a_bits"], tuple(map(tuple, d["u"])),
                    d.get("delta_bits"), d.get("decoder_gates"), d.get("labels", {}), d.get("tail_share", 0.0))


def even(u1: float) -> tuple[tuple[float, float], ...]:
    """The evenly spread profile u(x) = x·u(1)."""
    return ((0.0, 0.0), (1.0, float(u1)))


def with_tail(u, t: float) -> tuple[tuple[float, float], ...]:
    """The whole profile: the least of the from-start profile `u` and a frozen tail's, whose block does only the tail's
    share t of a token's dependent work (the frozen layers above the highest update) and reads no update bits, so 0 up
    to t, then u."""
    if not 0 < t < 1:
        return tuple(map(tuple, u))
    return ((0.0, 0.0), (t, 0.0), (t, u_at(u, t)), *((x, b) for x, b in u if x > t))


# ---- the update profile ------------------------------------------------------------------------------------------------

def u_at(u, x: float) -> float:
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if x <= x1:
            return u0 if x1 == x0 else u0 + (u1 - u0) * (x - x0) / (x1 - x0)
    return u[-1][1]


def bottom_share(u, k: float) -> float:
    """x₀: the largest x in [0, 1] with u(x) ≤ k (u non-decreasing)."""
    if u[-1][1] <= k:
        return 1.0
    for (x0, u0), (x1, u1) in zip(u, u[1:]):
        if u1 > k:
            return x0 if u1 == u0 or u0 > k else x0 + (x1 - x0) * (k - u0) / (u1 - u0)
    return 0.0


def room(u, k: float) -> float:
    """max over x in (0, 1] of x·(k − u(x)): on each linear piece a concave quadratic, so its ends and its vertex."""
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

def at(m: Measured, c: float, k: float, reading: str) -> dict:
    """ρ at (c, k) under one of `READINGS`, with each part's s, w, Fit and Slowdown's bound (module docstring)."""
    if reading not in READINGS:
        raise ValueError(f"reading {reading!r}: one of {', '.join(READINGS)}")
    delta = m.delta_bits if m.decoder_gates is None or m.decoder_gates <= c else None
    whole = with_tail(m.u, m.tail_share)
    x0, r = bottom_share(whole if reading == LITERAL else m.u, k), room(whole, k)
    parts, blocks, flags = [], 0.0, []
    for p in m.parts:
        w = p.gates / p.tokens if p.tokens else math.inf
        s = c / w
        phi = p.dep_gates / p.gates if p.gates else 0.0
        fit_inf = delta is not None and delta + s * m.tau_bits <= k
        fit_train = fit_inf and p.delta_grad_bits is not None and 2 * p.delta_grad_bits + s * m.tau_bits <= k
        bound = phi * (1 - x0) * s * m.a_bits / r if r > 0 else (0.0 if phi * (1 - x0) == 0 else math.inf)
        rho = 1.0 if fit_train else max(1.0, bound)
        if fit_train and bound > 1:
            flags.append(FIT_OVER_SLOWDOWN)
        parts.append({"w": w, "s": s, "phi": phi, "fit": fit_inf, "fit_training": fit_train, "slowdown": bound,
                      "rho": rho})
        blocks += p.gates * rho
    return {"c": c, "k": k, "reading": reading, "x0": x0, "room": r, "delta_counted": delta is not None, "parts": parts,
            "rho": blocks / m.gates if m.gates else 1.0, "flags": sorted(set(flags))}


def rhos(m: Measured, cells: dict[str, tuple[float, float]],
         readings=READINGS) -> tuple[dict[str, dict[str, float]], dict[str, bool]]:
    """ρ in each cell of `cells` (key -> (c, k)) under each reading, and whether Fit holds there for the training in
    every part (ρ = 1 by Fit, the same under both readings)."""
    rho: dict[str, dict[str, float]] = {name: {} for name in readings}
    fit = {}
    for key, (c, k) in cells.items():
        for name in readings:
            v = at(m, c, k, name)
            rho[name][key] = v["rho"]
        fit[key] = all(p["fit_training"] for p in v["parts"])
    return rho, fit


def fit_k(m: Measured, c: float) -> tuple[float | None, float | None]:
    """The least k at which Fit holds at c, for running the updated model and for the training (None: at no k, or
    unmeasured)."""
    if m.delta_bits is None or m.decoder_gates is not None and m.decoder_gates > c:
        return None, None
    s_tau = [c * p.tokens / p.gates * m.tau_bits if p.gates else 0.0 for p in m.parts]
    grads = [p.delta_grad_bits for p in m.parts]
    run = m.delta_bits + max(s_tau)
    train = None if None in grads else max(max(m.delta_bits, 2 * g) + st for g, st in zip(grads, s_tau))
    return run, train


# ---- measurement -------------------------------------------------------------------------------------------------------

STUBS = {
    "tau_bits": "stub: one token id per token block, ceil(log2 V) bits (its target is the next token's id)",
    "a_bits": "upper bound (witness): d_model x 16, one token's bf16 residual stream",
    "u": "upper bound (witness): evenly spread, u(x) = x * u(1)",
    "u1_bits": "upper bound (witness): u(1) = Delta, every token's path reads the whole update code",
    "decoder_gates": "stub: the decoder is assumed to fit in c",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "delta_grad_bits": "unmeasured: training Fit unchecked",
}
V23 = {
    "gates": "v2.3 macs_log in gates (te_core.gates; no dtypes, so its assumptions)",
    "tokens": "stub: forward MACs over the start's forward MACs per token (a v2.3 row has no token count)",
    "dep_gates": "stub: forward products that read an updated weight (v2.3's count, retired; understates Dep)",
}


def _forward(log: dict) -> dict:
    return {"ops": {k: v for k, v in log.get("ops", {}).items() if k.startswith("fwd ")},
            "by": {k: v for k, v in log.get("by", {}).items() if k.startswith("fwd/")}}


def _v23_part(row: dict, shape: dict) -> Part:
    log = row["macs_log"]
    fwd = sum(_forward(log)["by"].values())
    upd = {"ops": {k: v for k, v in log.get("ops", {}).items() if k.startswith("fwd updated ")},
           "by": {"fwd/updated": log.get("by", {}).get("fwd/updated", 0)}}
    return Part(of_log(log)["gates"], fwd / shape["forward_macs_per_token"], of_log(upd)["gates"],
                (row.get("annotation") or {}).get("steps", 1))


def complete(f: dict, row: dict, shape: dict) -> Measured:
    """`f`'s quantities (a `formalism` block, possibly partial), each missing one stubbed (`STUBS`) from the row and the
    start's shape; `labels` names every quantity's source."""
    labels = dict(f.get("labels", {}))
    if f.get("parts"):
        parts = tuple(Part(**p) for p in f["parts"])
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
    if f.get("u"):
        out["u"] = tuple(map(tuple, f["u"]))
        if f.get("u1_bits") is not None and not math.isclose(out["u"][-1][1], f["u1_bits"]):
            raise ValueError(f"u(1) is {f['u1_bits']} but the profile ends at {out['u'][-1][1]}")
        labels.setdefault("u", "upper bound (witness): the record's profile")
        labels.setdefault("u1_bits", labels["u"])
    else:
        u1 = f["u1_bits"] if f.get("u1_bits") is not None else out["delta_bits"]
        if u1 is None:
            raise ValueError("no update profile and no update code: rho is unmeasured")
        out["u"] = even(u1)
        labels.setdefault("u", STUBS["u"])
        if f.get("u1_bits") is None:
            labels.setdefault("u1_bits", STUBS["u1_bits"])
    if any(p.delta_grad_bits is None for p in parts):
        labels.setdefault("delta_grad_bits", STUBS["delta_grad_bits"])
    t = float(f.get("tail_share") or 0.0)
    if not 0 <= t < 1:
        raise ValueError(f"tail share {t} is not in [0, 1)")
    if t:
        labels.setdefault("tail_share", "upper bound (witness): the record's frozen tail, read with u")
    return Measured(parts, out["tau_bits"], out["a_bits"], out["u"], out["delta_bits"], out["decoder_gates"], labels, t)


def measure(row: dict, shape: dict) -> Measured:
    """A ledger row's quantities: its `formalism` block (ingest's, `of_record`), else a v2.3 row's own fields, with
    stubs where it has none (`STUBS`, `V23`)."""
    return complete(row.get("formalism") or {}, row, shape)


def of_record(record: dict, through: int | None = None) -> dict:
    """The `formalism` block ingest writes for a run record's prefix through step `through`: one part per step
    structure (its steps' gates by te_core.gates, their tokens, its dependent gates and Δ_grad), and the record's own
    `formalism` fields over these. A structure's dependent gates are the record's when it gives them (`dep_gates`), else
    its updated forward nodes' MACs (te_core 4.0's capture: every forward MAC with an update-dependent operand; the
    backward's are not counted yet), at the bf16 GEMM price; its Δ_grad is its `delta_grad_bits`, else 4.0's
    `cut_bits` (the narrowest cut between the step's gradient side and the new values it carries). The setup, and the
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
        if "dep_gates" in st:
            dep = st["dep_gates"] * n
        else:
            dep = sum(u[2] for u in st.get("updated", [])) * n * per_mac("gemm/bf16")
            labels.setdefault("dep_gates", "stub: forward MACs with an update-dependent operand (no backward), bf16 GEMM")
        dg = st.get("delta_grad_bits", st.get("cut_bits"))
        if "delta_grad_bits" not in st and dg is not None:
            labels.setdefault("delta_grad_bits", "measured: te_core 4.0's cut (gradient side to the new values)")
        parts.append({"gates": of_log(log)["gates"], "tokens": sum(s.get("tokens", 0) for s in ss), "dep_gates": dep,
                      "steps": n, "delta_grad_bits": dg})
    labels.setdefault("gates", "counted MACs in gates (te_core.gates)")
    labels.setdefault("tokens", "measured: the steps' token counts")
    return own | {"parts": own.get("parts") or parts, "labels": labels}
