"""ε against a fixed verification budget B: what each lever costs, what it buys, and the ε-optimal allocation (BUDGET,
cmp-13; Derived).

    uv run benchmarks/pouw/budget.py [--md OUT] [--json OUT] [--verify small|large] [--fixed slot|work]

B is the verification overhead as a fraction of the plain workload's compute: GPU-seconds a plain GPU-second, with CPU
time converted at the price ratio (a vCPU-second costs `VCPU_USD_H / gpu_usd_h` GPU-seconds). Its parts:

- the PoUW transform, γ − 1 (honest over plain int8, the machine model's γ on the family's measured kernel);
- commitment hashing and draws (COMMIT-GPU r5's fractions of plain prefill time);
- proving the drawn units: a session's fixed cost a batch of b rounds, plus each draw's marginal (PRF-57);
- verifying a session (PRF-57's Lean verify, or the Lean verifier profile's large sessions with `--verify large`).

ε is NOTATION's ε[Π] (note:20261018T0206Z-draft-notation) on the protocol decisions 101 and 102 leave: ε_skip and ε_machine
(route + cert) under the floor credit, at q* = 1.22 with operands free (decision 89), from the machine model; plus the
sampled term. With a seed a matmul (decision 101) the draws of a round land in independent seeds, so s draws a round
detect a deficient unit at s(1 − p) and a wrong unit at s, and the cheater's best is the larger of σ_w/(1 − p) and 1 a
unit: ε_samp = ℓ_s/(s·S)·max(σ_w/(1 − p), 1). There is no steering term and no closure. Full checking's luck (Λ_full, one
σ a round) is not a term here: under decision 101 it is the s = ∞ limit, 0.

The levers the optimum moves are verified filler (extra lane instructions a word, `FILL`; decision 90 divides every term
by the design's own floor program, filler included), draws a round s, rounds a session b, and checking P·V's served
output (on, since it costs only an opening and the epilogue in a drawn P·V unit's proof). `levers` reports the rest
(hashing leaf size, unit size, sessions shared across a site's GPUs, the operand floor, P·V's noise, credited subset,
L and noise bits) with what each moves; the run prints the allocation at 2×, 5× and 10× each family's B today and every
lever's marginal Δε per unit of B there. Every constant carries its source and whether it is measured, modeled or assumed.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

LN_INV_DELTA = 41 * math.log(2)   # ℓ_s at δ_s = 2⁻⁴¹, the machine model's luck budget (deficit 0.002368 at p = ¾)
ROUNDS = 48_000                   # S: one week of 12.6 s rounds (Theorem S; SPEC)
ROUND_S = 12.6
P_PASS = 0.75                     # a deficient unit's pass probability at 1-bit coins (Theorem S)
DEFICIT = LN_INV_DELTA / ((1 - P_PASS) * ROUNDS)   # d̄: full checking's deficit; the machine model's luck = σ_w·d̄

#: measured price: Nebius cpu-d3, $6.34/h for 192 vCPU, $8.45/h for 256, $4.23/h for 128 (machines.d, vy-nebius-cpu-1..3)
VCPU_USD_H = 0.033

#: PRF-57 (note:proofs/20261017T0552Z-report-proof-service-cost-model), C-Flock --zk on a 32-thread Xeon 6776P: process
#: wall 14 s at 16 whole 4K dot products a session and 42 s at 256, so 12.13 s fixed and 0.1167 s a 4K DP (measured,
#: fitted on two points); the 32-thread slot is held for the wall (`slot`), or only for the prove time, 1.8 s fixed and
#: 0.037 s a 4K DP plus ≈ 10 s on one thread (`work`, modeled). Sessions cap at 256 4K DPs (11 GiB, measured).
THREADS = 32
FIXED = {"slot": 12.13 * THREADS, "work": 1.8 * THREADS + 10.3}
PER_DP = {"slot": 0.1167 * THREADS, "work": 0.037 * THREADS}
SESSION_CAP = 256
#: verifying a session: Lean verify 1.0–1.3 s flat on one thread (PRF-57, measured, small sessions), or 159–174 CPU-s
#: flat in B for the large bf16 sessions (note:proofs/20261017T1150Z-finding-lean-verifier-profile, measured)
VERIFY = {"small": 1.2, "large": 167.0}
#: a draw's mean unit in 4K-DP equivalents: whole dot products drawn by work, proof cost linear in length, Llama-3.1-8B
#: prefill at 2K (E_w[k] = Σ n·k² / Σ n·k = 6,627 over qkv, o, gate/up at k 4,096, down at 14,336, QKᵀ at 128 and P·V at
#: the causal keys; derived from the shapes)
UNIT_4K = 1.62
#: a drawn unit's proof grows with the filler it recomputes (p = 0 needs the check to recompute it): (1 + y/16) at y
#: extra lane instructions a word, as if the receipt arm were a quarter of the proof at fx1's 4 a word (assumed)
FILL_PROOF = 1 / 16

#: of plain prefill time (COMMIT-GPU r5, r20261016-183051-a379; measured on sm_120 at 8 × 4K, applied to every family as
#: assumed): SHA-512 operand tile leaves at R = 128 (+6.0%; R = 64 is +10.1%), receipts at S = 128, simulated draws
HASH_LEAVES = {64: 0.101, 128: 0.060}
HASH_RECEIPTS = 0.016
DRAWS = 0.019


def hash_leaves(r: int) -> float:
    """SHA-512 tile-leaf hashing at leaf size R: a + b/R through COMMIT-GPU's two points (modeled past R = 128)."""
    b = (HASH_LEAVES[64] - HASH_LEAVES[128]) * 128
    return HASH_LEAVES[128] - b / 128 + b / r


FILL_DATA = Path(__file__).with_name("budget_fill.json")
#: the filler sweep's grid, extra lane instructions a word on top of the design's own fillers
FILL_GRID = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64)
FILL_MIXES = {"int": {"ALU": 1.0, "IMAD": 1.0}, "ffma": {"FFMA": 1.0}}
#: FFMA filler's p = 0 holds only once a bit-exact FFMA check exists (decision 99); integer filler (LOP3 and IMAD
#: alternating) is exact today. A device whose two sweeps agree keeps only the integer one.
CONDITIONAL = {"ffma": "p = 0 needs a bit-exact FFMA check (decision 99)"}


@dataclass(frozen=True)
class Family:
    """A GPU family's headline design, its price, and what eps_budget says of its P·V. Its filler sweep is
    `budget_fill.json` (`sweep`), the machine model's `eps --credit floor --operands free --q 1.22` on the design with
    `Design.with_fill` stacked on its own fillers; the sweep's ε_skip is the stored [skip] over the filled ♭ (decision
    90). γ is honest over plain int8 a MMA on the measured kernel; past y = 0 the honest time is the machine model's
    (modeled)."""
    name: str
    device: str
    design: str
    design_name: str
    vector: str
    gpu: str
    gpu_usd_h: float
    price_source: str
    #: ε_skip's piece only P·V's served output needs (its served epilogue and the tail's 4th IMMA, `eps_budget.floor_skip`
    #: served free against checked, 8B 2K, the binding cell on every family), a fraction of ε_skip
    served: float
    #: --operands floor: ε_machine's route term that φ_del closes (a new named assumption, decision 89: Daniel's)
    route: float
    #: P·V uncredited: ε_skip's most over the cells without P·V, and P·V's share of ♭ at the binding cell (eps_budget)
    skip_no_pv: float
    pv_share: float
    #: fillers the headline already carries, (lane instructions a word, class mix)
    pre_fill: tuple[tuple[float, tuple[tuple[str, float], ...]], ...] = ()
    #: P·V noised as `pv` instead of `peel` (sm_120 only, the one device with pv designs): ε_skip's ratio and the
    #: whole-model γ's change, dither-c against dither-c-peel in eps_budget (8B, floor credit; pv's γ is the TS model's)
    pv_noise: tuple[float, float] | None = None
    #: the next larger L's design on this device (machine model, measured kernels): (the headline's L, the larger L,
    #: its core ε as a fraction, its γ)
    larger_l: tuple[int, int, float, float] | None = None

    @functools.cached_property
    def fill(self) -> dict[str, list[tuple[float, float, float, float, float]]]:
        """Per filler mix, (y, γ, ε_skip, ε_machine, full checking's ε_luck) at each grid point, ε as fractions."""
        return {m: [tuple(p) for p in pts] for m, pts in _fill_data()["families"][self.name]["curves"].items()}

    def mixes(self) -> list[str]:
        return list(self.fill)


@functools.cache
def _fill_data() -> dict:
    return json.loads(FILL_DATA.read_text())


FAMILIES = (
    Family("sm_120", "sm_120", "H1 + fx1 (dither-c-peel + IMAD fill 1, VF3r-16)", "dither-c-peel", "VF3r-16",
           "RTX PRO 6000", 1.80, "vy-nebius-1/2: $14.40/h for 8 GPUs with their host (measured price)",
           served=0.1737, route=0.0012505, skip_no_pv=0.001321, pv_share=0.0184, pre_fill=((1.0, (("IMAD", 1.0),)),),
           pv_noise=(0.341 / 0.417, -0.010)),
    Family("H200", "sm_90", "LI-CP-W redux L32 (DC-90-ri, decision 98)", "dither-2s-licpw-redux-l32", "DC-90-ri",
           "H200", 5.40, "vy-dc-h200: $5.40/h, one GPU with its host (measured price)",
           served=0.1045, route=0.0, skip_no_pv=0.0, pv_share=0.0185),
    Family("B200", "sm_100", "LI-CP-W redux heavy L32 (DC-100-ri)", "dither-2s-licpw-redux-heavy-l32", "DC-100-ri",
           "B200", 7.00, "vy-dc-b200: $7.00/h, one GPU with its host (measured price)",
           served=0.2065, route=0.004308, skip_no_pv=0.0, pv_share=0.0185),
    Family("L40S", "sm_89", "dither-2s L16 (ADA-89)", "dither-2s-l16", "", "L40S", 1.90,
           "vy-dc-l40s: $1.90/h, one GPU with its host (measured price)",
           served=0.2186, route=0.000226, skip_no_pv=0.0, pv_share=0.0185, larger_l=(16, 32, 0.126726, 1.5446)),
    Family("A100", "sm_80", "dither-2s L16 (AMPERE-80)", "dither-2s-l16", "", "A100 SXM4 80 GB", 1.79,
           "vy-dc-a100: RunPod $1.79/h (note:20261017T1852Z-draft-rt-dc, measured price)",
           served=0.2184, route=0.0, skip_no_pv=0.0, pv_share=0.0185, larger_l=(16, 32, 0.014775, 2.9618)),
)


def sweep(out: Path = FILL_DATA) -> dict:
    """Regenerate `budget_fill.json` from the machine model (on a tree whose machine model has every family's design and
    vector; VF3r-16 is on next since 9c9567c396). Each point is one `eps` LP, ≈ 0.3 s."""
    from verity_machine_model import cli
    from verity_machine_model import eps as E
    from verity_machine_model.model import DATA, device

    fams = {}
    for f in FAMILIES:
        data = DATA if f.device == "sm_120" else device(f.device)
        pi = E.machine(f.vector, data)
        base = E.design(f.design_name, data)
        flat0 = cli._eps_record(base, pi, 1.22, data, "free", "floor", None, None, "full")["baseline"]["cycles_per_mma"]
        d0 = base
        for x, mix in f.pre_fill:
            d0 = d0.with_fill(x, dict(mix))
        curves = {}
        for name, mix in FILL_MIXES.items():
            pts = []
            for y in FILL_GRID:
                r = cli._eps_record(d0.with_fill(y, mix) if y else d0, pi, 1.22, data, "free", "floor", None, None,
                                    "full")
                flat = r["baseline"]["cycles_per_mma"]
                pts.append([y, round(r["gamma"], 5), round(r["terms"]["skip"] * flat0 / flat, 7),
                            round(r["terms"]["machine"], 7), round(r["terms"]["luck"], 7)])
            curves[name] = pts
        if curves["ffma"] == curves["int"]:
            del curves["ffma"]
        fams[f.name] = {"device": f.device, "design": f.design_name, "vector": pi.vector, "curves": curves}
    doc = {"source": "budget.py --sweep: machine model eps, --credit floor --operands free --q 1.22, filler stacked on "
                     "each design's own (Design.with_fill); point = [y, gamma, eps_skip, eps_machine, eps_luck]",
           "families": fams}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


S_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
B_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
#: today: s = 2 batched 2 rounds a session (SAMPLING, note:20261017T0608Z-draft-eps1-sampling, after PRF-57), no filler
#: beyond the design's own, P·V's served output free, R = 128, one GPU a site
TODAY = {"y": 0.0, "s": 2, "b": 2, "served": False}


@dataclass(frozen=True)
class Config:
    mix: str
    y: float
    s: int
    b: int
    served: bool


@dataclass(frozen=True)
class Point:
    """ε by term and B by part, each a fraction (ε_* of the credit, B_* of the plain workload's compute)."""
    config: Config
    eps: float
    eps_skip: float
    eps_served: float      # removed from ε_skip by checking P·V's served output (≤ 0)
    eps_machine: float
    eps_samp: float
    B: float
    B_transform: float
    B_hash: float
    B_prove: float
    B_verify: float
    gamma: float
    sigma_w: float


def eps_samp(s: float, sigma_w: float, p: float = P_PASS) -> float:
    """ε_samp under decision 101: s independent draws a round over S rounds, the worse of deficient units (pass at p)
    and wrong units (never pass), each worth its share of the credit."""
    return LN_INV_DELTA / (s * ROUNDS) * max(sigma_w / (1 - p), 1.0)


def rate(f: Family) -> float:
    """vCPU-seconds a GPU-second costs: the price ratio."""
    return f.gpu_usd_h / VCPU_USD_H


def point(f: Family, c: Config, *, verify: str = "small", fixed: str = "slot", leaf: int = 128, unit: float = UNIT_4K,
          gpus: int = 1) -> Point:
    y, gamma, skip, machine, luck = next(p for p in f.fill[c.mix] if p[0] == c.y)
    sigma_w = luck / DEFICIT
    served = -f.served * skip if c.served else 0.0
    samp = eps_samp(c.s, sigma_w)
    cpu = rate(f) * ROUND_S
    prove = (FIXED[fixed] / (c.b * gpus) + c.s * unit * PER_DP[fixed] * (1 + FILL_PROOF * y)) / cpu
    ver = VERIFY[verify] / (c.b * gpus) / cpu
    hashing = hash_leaves(leaf) + HASH_RECEIPTS + DRAWS
    return Point(c, skip + served + machine + samp, skip, served, machine, samp, gamma - 1 + hashing + prove + ver,
                 gamma - 1, hashing, prove, ver, gamma, sigma_w)


def feasible(c: Config, unit: float = UNIT_4K) -> bool:
    return c.s * c.b * unit <= SESSION_CAP


def today(f: Family, **kw) -> Point:
    return point(f, Config("int", **TODAY), **kw)


def configs(f: Family, mixes: list[str] | None = None, unit: float = UNIT_4K):
    for mix in mixes or f.mixes():
        for (y, *_), s, b in itertools.product(f.fill[mix], S_GRID, B_GRID):
            c = Config(mix, y, s, b, True)
            if feasible(c, unit):
                yield c


def optimum(f: Family, budget: float, mixes: list[str] | None = None, **kw) -> Point:
    """The least ε whose B fits the budget, over the grid (exhaustive: the filler curves aren't convex)."""
    unit = kw.get("unit", UNIT_4K)
    best = None
    for c in configs(f, mixes, unit):
        p = point(f, c, **kw)
        if p.B <= budget and (best is None or (p.eps, p.B) < (best.eps, best.B)):
            best = p
    if best is None:
        raise ValueError(f"{f.name}: no configuration fits B = {budget:.3f}")
    return best


def _fit_b(c: Config, unit: float) -> Config:
    """c with the most rounds a session that still fits the session cap."""
    fits = [b for b in B_GRID if feasible(Config(c.mix, c.y, c.s, b, c.served), unit)]
    return Config(c.mix, c.y, c.s, max(fits), c.served) if fits else c


def marginals(f: Family, p: Point, **kw) -> list[tuple[str, float, float]]:
    """Each optimum lever's marginal at p, (lever, Δε, ΔB) for its cheapest step that lowers ε (filler y up to the next
    grid point that does, draws s one grid step up with b refit to the session cap) and its step down (Δε ≥ 0, ΔB < 0).
    −Δε/ΔB is the marginal; at the optimum the up steps' marginals sit below the down steps', up to the grid."""
    unit, c, out = kw.get("unit", UNIT_4K), p.config, []
    ys = [q[0] for q in f.fill[c.mix]]
    i = ys.index(c.y)
    for y in ys[i + 1:]:
        q = point(f, Config(c.mix, y, c.s, c.b, c.served), **kw)
        if q.eps < p.eps:
            out.append((f"filler y {c.y:g} → {y:g}", q.eps - p.eps, q.B - p.B))
            break
    if i:
        q = point(f, Config(c.mix, ys[i - 1], c.s, c.b, c.served), **kw)
        out.append((f"filler y {c.y:g} → {ys[i - 1]:g}", q.eps - p.eps, q.B - p.B))
    j = S_GRID.index(c.s)
    for k, label in ((j + 1, "up"), (j - 1, "down")):
        if 0 <= k < len(S_GRID):
            n = _fit_b(Config(c.mix, c.y, S_GRID[k], c.b if label == "up" else B_GRID[-1], c.served), unit)
            if feasible(n, unit):
                q = point(f, n, **kw)
                out.append((f"draws s {c.s} → {n.s} (b {n.b})", q.eps - p.eps, q.B - p.B))
    return out


def unconditional(f: Family) -> list[str]:
    return [m for m in f.mixes() if m not in CONDITIONAL]


def levers(f: Family, **kw) -> list[dict]:
    """Every lever's first step from today: Δε and ΔB, with what it rests on. A lever with ΔB ≤ 0 and Δε < 0 needs no
    budget; one with Δε = 0 frees budget for the others."""
    t = today(f, **kw)
    c = t.config
    rows = []

    def row(name, q: Point | None, d_eps: float, d_b: float, rests: str):
        rows.append({"lever": name, "d_eps": d_eps, "d_B": d_b, "rests_on": rests})

    q = point(f, Config(c.mix, c.y, c.s, c.b, True), **kw)
    row("check P·V's served output in its own unit's proof", q, q.eps - t.eps, q.B - t.B,
        "eps_budget's served-checked reading (the cheater pays the 4th IMMA and the epilogue); the proof adds one "
        "opening and the epilogue, priced 0 here (modeled)")
    ys = [p[0] for p in f.fill[c.mix]]
    q = point(f, Config(c.mix, ys[1], c.s, c.b, c.served), **kw)
    row(f"verified filler, y = 0 → {ys[1]:g} a word", q, q.eps - t.eps, q.B - t.B,
        "the machine model's sweep (honest time modeled past the measured kernel); FILL-OPT (decision 94)")
    q = point(f, Config(c.mix, c.y, 4, c.b, c.served), **kw)
    row("draws a round s = 2 → 4", q, q.eps - t.eps, q.B - t.B, "PRF-57's marginal cost a 4K DP; decision 101")
    q = point(f, Config(c.mix, c.y, c.s, 32, c.served), **kw)
    row("rounds a session b = 2 → 32", q, q.eps - t.eps, q.B - t.B,
        "PRF-57's session fixed cost; S holds when a batch covers its rounds (SAMPLING)")
    q = point(f, c, **{**kw, "unit": 1.0})
    row("units cut at 4K (ū 1.62 → 1.0)", q, q.eps - t.eps, q.B - t.B,
        "needs the running sum pinned at each cut (an opening a cut, not priced)")
    q = point(f, c, **{**kw, "leaf": 256})
    row("SHA-512 leaves R = 128 → 256", q, q.eps - t.eps, q.B - t.B,
        "COMMIT-GPU's two points extrapolated (modeled)")
    q = point(f, c, **{**kw, "gpus": 8})
    row("one session for a site's 8 GPUs (PRF-48)", q, q.eps - t.eps, q.B - t.B,
        "PRF-48's batched site round; a deployment's shape, not the protocol's")
    row("operand floor (--operands floor)", None, -f.route, 0.0,
        "a new named assumption on φ_del (decision 89: Daniel's)")
    if f.pv_noise:
        ratio, d_gamma = f.pv_noise
        row("P·V noised as pv, not peel", None, (ratio - 1) * t.eps_skip, d_gamma,
            "eps_budget (dither-c against dither-c-peel); pv's kernel is the TS model's, unmeasured; not additive with "
            "the served check (pv has no peel)")
    row("P·V uncredited", None, f.skip_no_pv - t.eps_skip, 0.0,
        f"forgoes {100 * f.pv_share:.2f}% of the credit at 8B 2K (19% at 8B 32K); changes what ε is a fraction of")
    if f.larger_l:
        small, big, core, gamma = f.larger_l
        row(f"L = {small} → {big}", None, core - (t.eps_skip + t.eps_machine), gamma - t.gamma,
            "the device's own designs (machine model, measured kernels); no design has a smaller L")
    two_bit = 7 / 16
    row("2-bit coins (p ≤ 7/16)", None, eps_samp(c.s, t.sigma_w, two_bit) - t.eps_samp, 0.0,
        "moves only ε_samp's deficient branch, which binds only at σ_w > 1 − p; FRONTIER: perplexity +2.32/+2.83%, "
        "fails")
    return rows


def allocation(f: Family, mults=(1, 2, 5, 10), mixes: list[str] | None = None, **kw) -> list[tuple[float, Point, list]]:
    t = today(f, **kw)
    out = []
    for m in mults:
        p = optimum(f, m * t.B, mixes, **kw)
        out.append((m, p, marginals(f, p, **kw)))
    return out


def _pct(x: float, nd: int = 3) -> str:
    return f"{100 * x:.{nd}f}%"


def render(fams=FAMILIES, **kw) -> str:
    lines = ["# ε against the verification budget B (budget.py)", "",
             "B is a fraction of the plain workload's compute (CPU at the price ratio); ε by NOTATION's terms on "
             "decisions 101 and 102. The allocation is the grid's least ε within the budget. Marginals are −Δε per +0.1 "
             "of B (10% of the workload's compute), in percentage points, for each optimum lever's next step.", ""]
    for f in fams:
        t = today(f, **kw)
        lines += [f"## {f.name}: {f.design}", "",
                  f"GPU {f.gpu} at ${f.gpu_usd_h:.2f}/h ({f.price_source}); a GPU-second is {rate(f):.0f} vCPU-seconds.",
                  "", "| budget | config (mix, y, s, b, served) | ε | ε_skip | served | ε_machine | ε_samp | B | γ − 1 "
                  "| hash + draws | prove | verify |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        alloc = allocation(f, **kw)[1:]
        rows = [("today", t, [])] + [(f"{m}× today", p, mg) for m, p, mg in alloc]
        conditional = [m for m, p, _ in alloc if p.config.mix in CONDITIONAL]
        if conditional:
            rows += [(f"{m}× today, integer filler", p, mg)
                     for m, p, mg in allocation(f, conditional, unconditional(f), **kw)]
        for label, p, _ in rows:
            c = p.config
            cond = f" ({CONDITIONAL[c.mix]})" if c.mix in CONDITIONAL else ""
            lines.append(f"| {label} | {c.mix}{cond}, {c.y:g}, {c.s}, {c.b}, {'on' if c.served else 'off'} | "
                         f"{_pct(p.eps)} | {_pct(p.eps_skip)} | {_pct(p.eps_served)} | {_pct(p.eps_machine)} | "
                         f"{_pct(p.eps_samp, 4)} | {p.B:.3f} | {p.B_transform:.3f} | {p.B_hash:.3f} | "
                         f"{p.B_prove:.4f} | {p.B_verify:.5f} |")
        lines += ["", "Marginals at each optimum (−Δε pp per 0.1 of B; a step down's is the ε each 0.1 of B it frees "
                  "costs):", ""]
        for label, p, mg in rows[1:]:
            cells = ", ".join(f"{name} {-de / db * 0.1 * 100:.4f}" for name, de, db in mg if db)
            lines.append(f"- {label}: {cells}")
        lines += ["", "Each lever's first step from today:", "", "| lever | Δε | ΔB | −Δε pp per +0.1 B | rests on |",
                  "|---|---:|---:|---:|---|"]
        for r in levers(f, **kw):
            m = "free" if r["d_B"] <= 0 and r["d_eps"] < 0 else (
                "frees budget" if r["d_B"] < 0 else f"{-r['d_eps'] / r['d_B'] * 0.1 * 100:.4f}" if r["d_B"] > 0 else "—")
            lines.append(f"| {r['lever']} | {_pct(r['d_eps'], 4)} | {r['d_B']:+.4f} | {m} | {r['rests_on']} |")
        lines.append("")
    return "\n".join(lines)


def summary(fams=FAMILIES, **kw) -> dict:
    out = {}
    for f in fams:
        out[f.name] = {"today": asdict(today(f, **kw)),
                       "allocation": [{"mult": m, "point": asdict(p), "marginals": mg} for m, p, mg in allocation(f, **kw)],
                       "allocation_integer_filler": [{"mult": m, "point": asdict(p), "marginals": mg}
                                                     for m, p, mg in allocation(f, mixes=unconditional(f), **kw)],
                       "levers": levers(f, **kw)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--verify", choices=sorted(VERIFY), default="small")
    ap.add_argument("--fixed", choices=sorted(FIXED), default="slot")
    ap.add_argument("--sweep", action="store_true", help="regenerate budget_fill.json from the machine model first")
    a = ap.parse_args(argv)
    if a.sweep:
        sweep()
        _fill_data.cache_clear()
    kw = {"verify": a.verify, "fixed": a.fixed}
    text = render(**kw)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary(**kw), fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
