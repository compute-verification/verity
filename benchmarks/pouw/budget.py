"""ε against a fixed verification budget B: what each lever costs, what it buys, and the ε-optimal allocation (BUDGET,
cmp-13; Derived).

    uv run benchmarks/pouw/budget.py [--md OUT] [--json OUT] [--sweep] [--sampling-sweep TREE]

B is the verification overhead as a fraction of the plain workload's compute: GPU-seconds a plain GPU-second of the
family, with every other resource converted at its price. Its parts:

- the PoUW transform, γ − 1 (honest over plain int8, the machine model's γ on the family's measured kernel);
- commitment hashing, receipts and the noise's samples (COMMIT-GPU r5's fractions of plain prefill time);
- sampling, B_s: proving, staging and verifying the sampled proof blocks under proofs' fitted C-Flock cost model (cost-model/v1),
  as OPT2's planner spends it. `budget_sampling.json` is its frontier at δ_s (`sampling_sweep`): the least ε_sample at
  each spend over a GPU-week of one RTX PRO 6000, with the plan's samples and cut. The provers are RTX PRO 6000 workers,
  so a family's B_s buys B_s·gpu_usd_h/PROVER_USD_H of that spend.

ε is the headline's sum, core plus sampling (NOTATION's ε[Π], note:20261018T0206Z-draft-notation, with luck in the core
by decision 92(a)). Core is ε_skip (less the share that checking P·V's served output removes) and ε_machine (route +
cert), the machine model's under the floor credit at q* = 1.22 with operands free (decision 89), and ε_luck at the
protocol's rule. Sampling is OPT2's ε_sample, the units a plan's samples leave unproved. With a seed a matmul (decision
101) a drawn deficient unit is caught at 1 − p, so luck is ε_sample·σ_w/(1 − p): the machine model's full-checking
ε_luck (one σ a round) times S over the samples. The cheater's best mix of wrong and deficient units gains
ε_sample·max(σ_w/(1 − p), 1), which the sum bounds.

The levers the optimum moves are verified filler (extra lane instructions a word, `FILL`; decision 90 divides every term
by the design's own floor program, filler included), the sampling spend B_s (the rest of the budget: ε_sample only falls
with it) and checking P·V's served output (on, since it costs only an opening and the epilogue in a drawn P·V unit's
proof). The allocation is reported in cost-model/v1's two regimes: `plain` (fitted, one unit a session) and
`sessions/256` (multi-round sessions amortize a session's cost 256-fold, OPT2's estimate). `levers` reports the rest
(draws, batching, running sums committed anyway, hashing leaf size, the operand floor, P·V's noise, credited subset, L
and noise bits) with what each moves. The run prints the allocation at 1×, 2×, 5× and 10× each family's B today and
every lever's marginal Δε per unit of B there. Every constant carries its source and whether it is measured, modeled or
assumed.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

LN_INV_DELTA = 41 * math.log(2)   # ℓ_s at δ_s = 2⁻⁴¹, the machine model's luck budget (deficit 0.002368 at p = ¾)
ROUNDS = 48_000                   # S: one week of 12.6 s rounds (Theorem S; SPEC)
P_PASS = 0.75                     # a deficient unit's pass probability at 1-bit coins (Theorem S)
DEFICIT = LN_INV_DELTA / ((1 - P_PASS) * ROUNDS)   # d̄: full checking's deficit; the machine model's luck = σ_w·d̄

#: cost-model/v1 (art:551e804f4585f234f9c64b7e12655788e7af0fbe2eb75977fe1e913b8f2f22f2) is GPU-seconds of an RTX PRO
#: 6000 proving and CPU-seconds at 1/55 of one, so a family pays it at this price (vy-nebius-1/2, measured price; that
#: the provers are RTX PRO 6000 workers is assumed)
PROVER_USD_H = 1.80
SAMPLING_DATA = Path(__file__).with_name("budget_sampling.json")
#: cost-model/v1's regimes, (commitment factor, session-cost factor) in OPT2's `run`: `plain` as fitted; `sessions/256`
#: with a session's cost amortized 256-fold by multi-round sessions (estimate); `prefix, sessions/256` with the running
#: sums committed anyway too (COMMIT's prefix mode)
REGIMES = {"plain": (1.0, 1.0), "sessions/256": (1.0, 1 / 256), "prefix, sessions/256": (0.0, 1 / 256)}
ALLOCATED = ("plain", "sessions/256")
#: the sweep's spends B_s, a fraction of an RTX PRO 6000's week, 8 a decade
SPEND_GRID = tuple(round(10 ** (e / 8), 6) for e in range(-28, 21))
#: today's samples: 2 a round over a week (SAMPLING, note:20261017T0608Z-draft-eps1-sampling, after PRF-57)
TODAY_DRAWS = 2 * ROUNDS
#: a sampled proof block's proof grows with the filler it recomputes (p = 0 needs the check to recompute it): its ANDs by
#: (1 + y/16) at y extra lane instructions a word, as if the receipt arm were a quarter of the proof at fx1's 4 a word
#: (assumed); it scales the part of a sample's cost that grows with the unit's ANDs (`unit_share`)
FILL_PROOF = 1 / 16

#: of plain prefill time (COMMIT-GPU r5, r20261016-183051-a379; measured on sm_120 at 8 × 4K, applied to every family as
#: assumed): SHA-512 operand tile leaves at R = 128 (+6.0%; R = 64 is +10.1%), receipts at S = 128, simulated draws.
#: R = 128's is cost-model/v1's commitment floor too.
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
    #: --operands floor: ε_machine's route term that φ_del closes (a new named assumption)
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

    @property
    def prover(self) -> float:
        """Plain GPU-seconds of this family an RTX PRO 6000's second costs: the price ratio."""
        return PROVER_USD_H / self.gpu_usd_h


@functools.cache
def _fill_data() -> dict:
    return json.loads(FILL_DATA.read_text())


FAMILIES = (
    Family("sm_120", "sm_120", "H1 + fx1 (dither-c-peel + IMAD fill 1, VF3r-16)", "dither-c-peel", "VF3r-16",
           "RTX PRO 6000", 1.80, "vy-nebius-1/2: $14.40/h for 8 GPUs with their host (measured price)",
           served=0.1737, route=0.0012505, skip_no_pv=0.001321, pv_share=0.0184, pre_fill=((1.0, (("IMAD", 1.0),)),),
           pv_noise=(0.341 / 0.417, -0.010)),
    Family("H200", "sm_90", "LI-CP-W redux L32, ffill at p = 1 (DC-90-ri, decisions 98 and 99)",
           "dither-2s-licpw-redux-p1-l32", "DC-90-ri",
           "H200", 5.40, "vy-dc-h200: $5.40/h, one GPU with its host (measured price)",
           served=0.0979, route=0.0, skip_no_pv=0.0, pv_share=0.0185),
    Family("B200", "sm_100", "LI-CP-W redux L32, ♭ without LDTM, the fold the checks read (DC-100-ri)",
           "dither-2s-licpw-redux-ldtm0-l32", "DC-100-ri",
           "B200", 7.00, "vy-dc-b200: $7.00/h, one GPU with its host (measured price)",
           served=0.2061, route=0.0, skip_no_pv=0.0, pv_share=0.0185),
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
    from verity_machine_model import eps as E
    from verity_machine_model.commands import eps as cmd
    from verity_machine_model.model import DATA, device

    fams = {}
    for f in FAMILIES:
        data = DATA if f.device == "sm_120" else device(f.device)
        pi = E.machine(f.vector, data)
        base = E.design(f.design_name, data)
        flat0 = cmd._eps_record(base, pi, 1.22, data, "free", "floor", None, None, "full")["baseline"]["cycles_per_mma"]
        d0 = base
        for x, mix in f.pre_fill:
            d0 = d0.with_fill(x, dict(mix))
        curves = {}
        for name, mix in FILL_MIXES.items():
            pts = []
            for y in FILL_GRID:
                r = cmd._eps_record(d0.with_fill(y, mix) if y else d0, pi, 1.22, data, "free", "floor", None, None,
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


def _cut_label(cut: dict) -> str:
    """The linears' units: `coord` when every linear is proved by whole coordinates, else each K's chunk length."""
    units = {op.removeprefix("linear K="): names[0] for op, names in cut.items() if op.startswith("linear")}
    if all(u.startswith("S8Coord") for u in units.values()):
        return "coord"
    return " ".join(f"{k}:{u.split('L=')[1].rstrip('}') if 'L=' in u else 'coord'}" for k, u in units.items())


def sampling_sweep(tree: Path, out: Path = SAMPLING_DATA) -> dict:
    """Regenerate `budget_sampling.json` from OPT2's planner on `tree`, a checkout with
    `benchmarks/sampling_plan/int8_llama.py` whose B is the serve-time total (ASK-25), run with `scheme=None` so a spend
    is sampling's alone, the serve-time commitments being `budget.py`'s own terms. Per regime its frontier at δ_s, and at
    each spend of SPEND_GRID the least-ε plan within it (`at_budget`, the zero-work templates included): ε_sample, the
    samples a week, the share of the plan's cost that grows with a unit's ANDs (all but the sessions' fixed cost) and the
    linears' cut. ≈ 2 s a regime."""
    tree = Path(tree).resolve()
    sys.path[:0] = [str(tree), str(tree / "benchmarks" / "sampling_plan")]
    import int8_llama as P
    from inputs import I

    P.DELTA = math.exp(-LN_INV_DELTA)
    v = {k: i.value for k, i in I.items()}
    regimes = {}
    for name, (commit, statement) in REGIMES.items():
        _, w, costs, objective, cands, alts, _, plans = P.run(v, commit=commit, statement=statement, scheme=None)
        extra = P.zero_work_cost(costs, v)
        pts = []
        for bs in SPEND_GRID:
            p = P.at_budget(bs, plans, cands, costs, objective, w, extra)
            if p is None:
                continue
            s = P.summarize(p, w, extra, alts)
            sessions = sum(math.ceil(n / s["statement"][t]) for t, n in s["draws"].items())
            total = p.cost["total"] + extra
            pts.append([bs, float(f"{s['eps']:.7g}"), sum(s["draws"].values()),
                        round(1 - costs.statement_fixed * sessions / total, 5), _cut_label(s["cut"])])
        regimes[name] = {"commit": commit, "statement": statement, "points": pts}
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True)
    doc = {"source": "budget.py --sampling-sweep: OPT2's int8 Llama-3.1-8B row (benchmarks/sampling_plan/"
                     f"int8_llama.py at {head.stdout.strip() or 'unknown'}), cost-model/v1, δ_s = 2^-41, a week of one "
                     "RTX PRO 6000; point = [B_s, eps_sample, draws a week, unit_share, linears' cut]",
           "regimes": regimes}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


@functools.cache
def _sampling_data() -> dict:
    return json.loads(SAMPLING_DATA.read_text())


@functools.cache
def curve(regime: str) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...],
                                tuple[str, ...]]:
    """A regime's frontier as columns: spend, ε_sample, draws a week, unit share, cut."""
    return tuple(zip(*_sampling_data()["regimes"][regime]["points"]))


def _loglog(xs, ys, x: float) -> float:
    """Through the points in log-log, the end segments extended."""
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    t = math.log(x / xs[i - 1]) / math.log(xs[i] / xs[i - 1])
    return math.exp(math.log(ys[i - 1]) + t * (math.log(ys[i]) - math.log(ys[i - 1])))


def spend_for(draws: float, regime: str = "plain") -> float:
    """The spend (an RTX PRO 6000's week) at which the regime's plan draws `draws` units a week."""
    xs, _, ds, _, _ = curve(regime)
    return _loglog(ds, xs, draws)


@dataclass(frozen=True)
class Sample:
    eps: float           # OPT2's ε_sample at the plan
    draws: float         # units drawn a week
    spend: float         # the plan's spend, a fraction of an RTX PRO 6000's week, the filler's growth taken out
    cut: str


def sample(f: Family, b_s: float, y: float = 0.0, regime: str = "plain") -> Sample | None:
    """What a family's B_s buys: its spend on RTX PRO 6000 provers, shrunk by the filler's growth of each sampled proof block's
    ANDs; None below the sweep's least spend."""
    xs, es, ds, us, cuts = curve(regime)
    x0 = b_s / f.prover
    if x0 < xs[0]:
        return None
    u = us[min(max(bisect.bisect_right(xs, x0) - 1, 0), len(xs) - 1)]
    x = x0 / (1 + FILL_PROOF * y * u)
    if x < xs[0]:
        return None
    return Sample(_loglog(xs, es, x), _loglog(xs, ds, x), x, cuts[min(max(bisect.bisect_right(xs, x) - 1, 0),
                                                                     len(xs) - 1)])


@dataclass(frozen=True)
class Config:
    mix: str
    y: float
    served: bool
    regime: str = "plain"


@dataclass(frozen=True)
class Point:
    """ε by term and B by part, each a fraction (ε_* of the credit, B_* of the plain workload's compute)."""
    config: Config
    eps: float
    eps_skip: float
    eps_served: float      # removed from ε_skip by checking P·V's served output (≤ 0)
    eps_machine: float
    eps_luck: float
    eps_sample: float
    B: float
    B_transform: float
    B_hash: float
    B_sample: float
    draws: float
    cut: str
    gamma: float
    sigma_w: float

    @property
    def core(self) -> float:
        return self.eps_skip + self.eps_served + self.eps_machine + self.eps_luck


def point(f: Family, c: Config, b_s: float, *, leaf: int = 128, p: float = P_PASS) -> Point | None:
    """The configuration with B_s spent on sampling; None if B_s doesn't reach the sweep's least spend."""
    y, gamma, skip, machine, luck_full = next(q for q in f.fill[c.mix] if q[0] == c.y)
    s = sample(f, b_s, y, c.regime)
    if s is None:
        return None
    sigma_w = luck_full / DEFICIT
    luck = s.eps * sigma_w / (1 - p)
    served = -f.served * skip if c.served else 0.0
    hashing = hash_leaves(leaf) + HASH_RECEIPTS + DRAWS
    return Point(c, skip + served + machine + luck + s.eps, skip, served, machine, luck, s.eps,
                 gamma - 1 + hashing + b_s, gamma - 1, hashing, b_s, s.draws, s.cut, gamma, sigma_w)


def fixed_b(f: Family, c: Config, leaf: int = 128) -> float:
    """B without sampling: the transform and the hashing."""
    gamma = next(q for q in f.fill[c.mix] if q[0] == c.y)[1]
    return gamma - 1 + hash_leaves(leaf) + HASH_RECEIPTS + DRAWS


def today(f: Family, **kw) -> Point:
    """Today's protocol priced by cost-model/v1: the design's own filler, P·V's served output free, today's samples."""
    return point(f, Config("int", 0.0, False), f.prover * spend_for(TODAY_DRAWS), **kw)


def optimum(f: Family, budget: float, mixes: list[str] | None = None, regime: str = "plain", **kw) -> Point:
    """The least ε within the budget: per filler point (exhaustive, the curves aren't convex), the rest of the budget
    spent on sampling."""
    best = None
    for mix in mixes or f.mixes():
        for y, *_ in f.fill[mix]:
            c = Config(mix, y, True, regime)
            q = point(f, c, budget - fixed_b(f, c, kw.get("leaf", 128)), **kw)
            if q is not None and (best is None or (q.eps, q.B) < (best.eps, best.B)):
                best = q
    if best is None:
        raise ValueError(f"{f.name}: no configuration fits B = {budget:.3f}")
    return best


#: the sampling spend's step in `marginals`
STEP = 0.01


def marginals(f: Family, q: Point, **kw) -> list[tuple[str, float, float]]:
    """Each optimum lever's marginal at q, (lever, Δε, ΔB), each step with the other levers held: filler y up to the
    next grid point that lowers ε and one point down, and the sampling spend STEP up and down. −Δε/ΔB is the marginal;
    at the optimum the up steps' marginals sit below the down steps', up to the grid."""
    c, out = q.config, []
    ys = [p[0] for p in f.fill[c.mix]]
    i = ys.index(c.y)
    for y in ys[i + 1:]:
        n = point(f, Config(c.mix, y, c.served, c.regime), q.B_sample, **kw)
        if n is not None and n.eps < q.eps:
            out.append((f"filler y {c.y:g} → {y:g}", n.eps - q.eps, n.B - q.B))
            break
    if i:
        n = point(f, Config(c.mix, ys[i - 1], c.served, c.regime), q.B_sample, **kw)
        if n is not None:
            out.append((f"filler y {c.y:g} → {ys[i - 1]:g}", n.eps - q.eps, n.B - q.B))
    for d in (STEP, -STEP):
        n = point(f, c, q.B_sample + d, **kw)
        if n is not None:
            out.append((f"sampling B_s {q.B_sample:.3g} → {q.B_sample + d:.3g}", n.eps - q.eps, d))
    return out


def unconditional(f: Family) -> list[str]:
    return [m for m in f.mixes() if m not in CONDITIONAL]


def levers(f: Family, **kw) -> list[dict]:
    """Every lever's first step from today: Δε and ΔB, with what it rests on. A lever with ΔB ≤ 0 and Δε < 0 needs no
    budget; one with Δε = 0 frees budget for the others."""
    t = today(f, **kw)
    c = t.config
    rows = []

    def row(name, d_eps: float, d_b: float, rests: str):
        rows.append({"lever": name, "d_eps": d_eps, "d_B": d_b, "rests_on": rests})

    def step(q: Point | None, name: str, rests: str):
        row(name, q.eps - t.eps, q.B - t.B, rests)

    step(point(f, Config(c.mix, c.y, True), t.B_sample, **kw), "check P·V's served output in its own unit's proof",
         "eps_budget's served-checked reading (the cheater pays the 4th IMMA and the epilogue); the proof adds one "
         "opening and the epilogue, priced 0 here (modeled)")
    ys = [p[0] for p in f.fill[c.mix]]
    step(point(f, Config(c.mix, ys[1], c.served), t.B_sample, **kw), f"verified filler, y = 0 → {ys[1]:g} a word",
         "the machine model's sweep (honest time modeled past the measured kernel); FILL-OPT (decision 94)")
    step(point(f, c, f.prover * spend_for(2 * TODAY_DRAWS), **kw), "draws 2 → 4 a round",
         "cost-model/v1's sessions (A6's two, fitted: the session's fixed cost is the uncertain part, σ 0.5)")
    for regime, rests in (("sessions/256", "multi-round sessions amortizing a session 256-fold (OPT2's estimate); S "
                                           "holds when a session covers its rounds"),
                          ("prefix, sessions/256", "also the running sums committed anyway (COMMIT's prefix mode), so "
                                                   "chunks of 32 pay")):
        step(point(f, Config(c.mix, c.y, c.served, regime), f.prover * spend_for(TODAY_DRAWS, regime), **kw),
             f"today's draws in regime {regime}", rests)
    step(point(f, c, t.B_sample, leaf=256, **{k: v for k, v in kw.items() if k != "leaf"}),
         "SHA-512 leaves R = 128 → 256", "COMMIT-GPU's two points extrapolated (modeled)")
    row("operand floor (--operands floor)", -f.route, 0.0, "a new named assumption on φ_del (decision 89: Daniel's)")
    if f.pv_noise:
        ratio, d_gamma = f.pv_noise
        row("P·V noised as pv, not peel", (ratio - 1) * t.eps_skip, d_gamma,
            "eps_budget (dither-c against dither-c-peel); pv's kernel is the TS model's, unmeasured; not additive with "
            "the served check (pv has no peel)")
    row("P·V uncredited", f.skip_no_pv - t.eps_skip, 0.0,
        f"forgoes {100 * f.pv_share:.2f}% of the credit at 8B 2K (19% at 8B 32K); changes what ε is a fraction of")
    if f.larger_l:
        small, big, core, gamma = f.larger_l
        row(f"L = {small} → {big}", core - (t.eps_skip + t.eps_machine), gamma - t.gamma,
            "the device's own designs (machine model, measured kernels); no design has a smaller L")
    step(point(f, c, t.B_sample, p=7 / 16, **kw), "2-bit coins (p ≤ 7/16)",
         "moves only luck; FRONTIER: perplexity +2.32/+2.83%, fails")
    return rows


def allocation(f: Family, mults=(1, 2, 5, 10), mixes: list[str] | None = None, regime: str = "plain",
               **kw) -> list[tuple[float, Point, list]]:
    """The optimum at each multiple of today's B (today's in the plain regime, whatever `regime` allocates)."""
    t = today(f, **kw)
    out = []
    for m in mults:
        q = optimum(f, m * t.B, mixes, regime, **kw)
        out.append((m, q, marginals(f, q, **kw)))
    return out


def _pct(x: float, nd: int = 3) -> str:
    return f"{100 * x:.{nd}f}%"


def headline(fams=FAMILIES, mults=(1, 2, 5, 10), **kw) -> list[str]:
    """Per family, ε = core + sampling today and at each budget, in each allocated regime."""
    lines = ["| family | regime | today | " + " | ".join(f"{m}× today" for m in mults) + " |",
             "|---|---|---|" + "---|" * len(mults)]
    def cell(q: Point) -> str:
        return f"{_pct(q.eps)} = {_pct(q.core)} + {_pct(q.eps_sample, 4)} (B {q.B:.2f})"

    for f in fams:
        t = today(f, **kw)
        for regime in ALLOCATED:
            cells = [cell(q) for _, q, _ in allocation(f, mults, regime=regime, **kw)]
            first = cell(t) if regime == "plain" else "—"
            lines.append(f"| {f.name} | {regime} | {first} | " + " | ".join(cells) + " |")
    return lines


def render(fams=FAMILIES, **kw) -> str:
    lines = ["# ε against the verification budget B (budget.py)", "",
             "B is a fraction of the plain workload's compute (other resources at their price); ε is the headline's "
             "core (ε_skip + served + ε_machine + ε_luck) plus sampling, OPT2's ε_sample under cost-model/v1 at δ_s = "
             "2⁻⁴¹. The allocation is the least ε within the budget, the rest of it spent on sampling. Marginals are "
             "−Δε per +0.1 of B (10% of the workload's compute), in percentage points, for each optimum lever's next "
             "step.", "", "## Headline: ε = core + sampling", ""] + headline(fams, **kw) + [""]
    for f in fams:
        t = today(f, **kw)
        lines += [f"## {f.name}: {f.design}", "",
                  f"GPU {f.gpu} at ${f.gpu_usd_h:.2f}/h ({f.price_source}); a B_s of its compute buys "
                  f"{1 / f.prover:.3g}× as much of cost-model/v1's RTX PRO 6000 provers.", ""]
        for regime in ALLOCATED:
            lines += [f"Regime `{regime}`:", "",
                      "| budget | config (mix, y, served) | ε | core | ε_skip | served | ε_machine | ε_luck | ε_sample | B "
                      "| γ − 1 | hash + receipts + draws | B_s | draws a week | cut |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            alloc = allocation(f, regime=regime, **kw)
            rows = ([("today", t, [])] if regime == "plain" else []) + [(f"{m}× today", q, mg) for m, q, mg in alloc]
            conditional = [m for m, q, _ in alloc if q.config.mix in CONDITIONAL]
            if conditional:
                rows += [(f"{m}× today, integer filler", q, mg)
                         for m, q, mg in allocation(f, conditional, unconditional(f), regime, **kw)]
            for label, q, _ in rows:
                c = q.config
                cond = f" ({CONDITIONAL[c.mix]})" if c.mix in CONDITIONAL else ""
                lines.append(f"| {label} | {c.mix}{cond}, {c.y:g}, {'on' if c.served else 'off'} | {_pct(q.eps)} | "
                             f"{_pct(q.core)} | {_pct(q.eps_skip)} | {_pct(q.eps_served)} | {_pct(q.eps_machine)} | "
                             f"{_pct(q.eps_luck)} | {_pct(q.eps_sample, 4)} | {q.B:.3f} | {q.B_transform:.3f} | "
                             f"{q.B_hash:.3f} | {q.B_sample:.4f} | {q.draws:.3g} | {q.cut} |")
            lines += ["", "Marginals at each optimum (−Δε pp per 0.1 of B; a step down's is the ε each 0.1 of B it "
                      "frees costs):", ""]
            for label, q, mg in rows:
                if mg:
                    cells = ", ".join(f"{name} {-de / db * 0.1 * 100:.4f}" for name, de, db in mg if db)
                    lines.append(f"- {label}: {cells}")
            lines.append("")
        lines += ["Each lever's first step from today:", "", "| lever | Δε | ΔB | −Δε pp per +0.1 B | rests on |",
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
                       "allocation": {r: [{"mult": m, "point": asdict(q), "marginals": mg}
                                          for m, q, mg in allocation(f, regime=r, **kw)] for r in ALLOCATED},
                       "allocation_integer_filler": {r: [{"mult": m, "point": asdict(q), "marginals": mg} for m, q, mg
                                                         in allocation(f, mixes=unconditional(f), regime=r, **kw)]
                                                     for r in ALLOCATED},
                       "levers": levers(f, **kw)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--sweep", action="store_true", help="regenerate budget_fill.json from the machine model first")
    ap.add_argument("--sampling-sweep", metavar="TREE", type=Path,
                    help="regenerate budget_sampling.json from OPT2's planner on TREE first")
    a = ap.parse_args(argv)
    if a.sweep:
        sweep()
        _fill_data.cache_clear()
    if a.sampling_sweep:
        sampling_sweep(a.sampling_sweep)
        _sampling_data.cache_clear()
        curve.cache_clear()
    text = render()
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(summary(), fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
