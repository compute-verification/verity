"""The first certified PoUW accounting period: windows, draws, wall-clock and node-days, and what each lever buys.

    uv run benchmarks/pouw/period_cost.py [--json OUT] [--md OUT]

**The bound.** An accepted period of n windows (#2722's `ComposePeriod`) has ε(n) = ε_♭ + Λ·ln(1/δ_s)/(u·n), with Λ the
sharp split rate (`SplitRateExp`) at K_d draws a window, M rounds and credit ratio r:

    x = K_d/(r·M),  Λ = max(a/(M·(−ln(p + (1 − p)e^{−x}))), b/(M·(−ln(p0 + (1 − p0)e^{−x})))),

at the lucks a = 0.218, b = 0.998, p = 7/8, p0 = ½, δ_s = 2^-40 and ε_♭ = 0.571% (sm_120 sheet, H1 + fx1). u is the
period's admitted rows over its declared rows, so the first period at ε < 1% is n = ⌈Λ·ln(1/δ_s)/(u·(1% − ε_♭))⌉ windows.
At small x, Λ ≈ b·r/((1 − p0)·K_d): what counts is the period's total draws n·K_d, and M past 18 buys nothing.
A period's credit is (1 − ε(j))₊ of its accepted prefix's live words, at the realized prefix j (RT4-INT-8), so a prefix
earns nothing until ε(j) < 1 (`first_credit`: 202 windows today). A window that skips every drawn tile passes with
p0^K_d, which past 1 − b (K_d ≤ 8) would out-earn an honest window if credited without that discount (`all_skip_pays`).

**The served window** (r20261018-234409-19cd, SERVE-H1 on vy-nebius-2): 18 rounds, 16 served steps and 2 empty, 64 rows a
step; 2,304 GEMM calls and 1,548,288 units of 32×32 outputs (86,016 a step: qkv 384, o 256, gate_up 1,792 at K = 4,096
and down 256 at K = 14,336, a layer, 32 layers). 5 sequences fill 119 of its 1,152 row slots (u = 119/1152; 731,136 units
live). Its 485.5 s serve is 439.1 s of receipts (284 µs a unit, launch-bound torch, on a node at load 500 to 615), 1.7 s of
empty steps and about 45 s of eager int8 serving; the receipt trees add 12.6 s. The period counts 439.1 s a window,
and 32 times that at 4×8 units unless the receipts are fused into the kernel.

**A window's certification** is its K_d sampled proof blocks, proved by C-Flock (`--zk`, live coins) and checked by the Lean
verifier (`flock-verify`). A sampled proof block costs its outputs × K in lanes of the measured H1 tile (`PouwH1Tile_v1`, 4×8
outputs at K = 4,096: 46.7 M ANDs, `k_log` 27, the most `Stmt.InRange` admits): 32 at K = 4,096, 39.6 at the served mean
K. down's units need a narrower lane to fit a block (a 4×8 tile at K = 14,336 is 164 M ANDs). A statement holds at most
256 instances (`m = k_log + nbl ≤ 35`), each a lane, and no more than its prover's memory allows on one node. Its cost is
fixed parts (the prover's load of the unrolled circuit and its session's per-round work; Lean's parse of the 26.8 GB
circuit file and its teardown, and Lean's session check) plus a part a lane. `RATES` holds each measurement and its run.

**Node-days** are memory-packed: a node runs ⌊node GB / process GB⌋ provers (prover and session verifier) or Lean checks
at once, and a window's node-seconds are its prove seconds over the prove slots plus its Lean seconds over the Lean
slots. Process-days are the same seconds unpacked (INT-PERIOD's node-days). Windows certify in parallel on N nodes, but
one stream serves the period (#2722: disjoint periods, one stream each), and a window's sample comes only once it is
served, so the period takes max(serving, certifying its first n − 1 windows on N nodes) plus its last window's
certification: its statements in parallel, but no sooner than one statement's prove and check.

**One window long enough on its own** (`long_window`) is n = 1: Compose's `CircuitWork`, with no period statement and
no record check. It needs Λ·ln(1/δ_s)/u under the budget, and Λ falls with M only to `lam_limit`, so a K_d below
r·max(a/(1 − p), b/(1 − p0))·ln(1/δ_s)/(u·budget) never gets there. Its certification starts when its serving ends,
and it holds custody of M/18 served windows' values until its sample.

**Levers** (`LEVERS`): one change each, with its owner and a work class (1 a parameter, 2 one module, 3 a new path in one
component, 4 a new circuit or proof form), applied alone to today's configuration and in sequence along `route`. δ_s is
the claim's parameter and ε_♭ the machine model's: both are shown (`sensitivity`), neither is a lever here.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

A, B, P, P0 = 0.218, 0.998, 7 / 8, 0.5
DELTA_S_LOG2 = 40
EPS_FLAT = 0.00571
TARGET = 0.01
DAY = 86_400.0

ROUNDS, ROWS, SERVED_STEPS, LAYERS = 18, 64, 16, 32
R_SERVED = 2.87
U_TODAY = 119 / 1152
U_FULL = SERVED_STEPS / ROUNDS
T_WINDOW = 439.1
T_WINDOW_DITHER = ROUNDS * 0.06665
CUSTODY_GB = 1.05050112
UNIT_ROWS, UNIT_COLS = 32, 32
LANE_OUTPUTS, LANE_K = 4 * 8, 4096
SERVED_UNITS_PER_LAYER = {"qkv": (384, 4096), "o": (256, 4096), "gate_up": (1792, 4096), "down": (256, 14336)}
STATEMENT_LANES = 256
NODE_GB = 1007
NODES = 2
SPEND = "certifying nodes 2 → 8"


def k_factor(units: dict[str, tuple[int, int]] = SERVED_UNITS_PER_LAYER) -> float:
    """A uniformly sampled served unit's mean K over the measured tile's 4,096."""
    n = sum(c for c, _ in units.values())
    return sum(c * k for c, k in units.values()) / n / LANE_K


def lanes_per_unit(unit_outputs: int = UNIT_ROWS * UNIT_COLS, k: float | None = None) -> float:
    """A sampled proof block's lanes of the measured tile: its outputs over the tile's, times its K over 4,096."""
    return unit_outputs / LANE_OUTPUTS * (k_factor() if k is None else k)


@dataclass(frozen=True)
class Rates:
    """One statement's cost: fixed seconds and seconds a lane, and fixed GB and GB a lane, for the prover (with its
    session verifier) and for Lean."""
    name: str
    source: str
    prove_load_s: float
    prove_session_s: float
    prove_lane_s: float
    prove_gb: float
    prove_lane_gb: float
    lean_process_s: float
    lean_session_s: float
    lean_lane_s: float
    lean_gb: float
    lean_lane_gb: float


@dataclass(frozen=True)
class Pair:
    """One quantity of the K = 1,024 tile's statement, measured at 8 lanes (one instance, padded) and at 64."""
    run: str
    at8: float
    at64: float

    @property
    def lane(self) -> float:
        return (self.at64 - self.at8) / 56


ANDS_4096, ANDS_1024 = 46_744_981, 12_904_803
W2 = "r20261019-000025-2a3b"
W2_PROVE_LOAD_S, W2_SESSION_S, W2_PROVE_GB = 529.05 - 200.6, 200.6, 136.8 + 136.1
W2_LEAN_PARSE_S, W2_LEAN_SETUP_S, W2_LEAN_SESSION_S, W2_LEAN_TEARDOWN_S, W2_LEAN_GB = 1865.8, 0.9, 1175.7, 527.2, 110.2
K1024_SESSION_S = Pair("r20261019-230625-bc89", (41.65 + 31.44) / 2, (71.57 + 51.53) / 2)
K1024_PROVE_GB = Pair("r20261019-230625-bc89", 36.08 + 28.94, 41.49 + 30.85)
K1024_LEAN_SESSION_S = Pair("r20261019-224439-a8f6", (222.84 + 222.92) / 2, (224.01 + 222.56) / 2)
SESSION_SCALE = W2_SESSION_S / K1024_SESSION_S.at8
LEAN_SCALE = W2_LEAN_SESSION_S / K1024_LEAN_SESSION_S.at8


def today_rates() -> Rates:
    """K = 4,096's statement at 26 threads: W2's fixed parts, and a lane at K = 1,024's slope (today's main) scaled to
    K = 4,096 by the two sizes' 8-lane sessions (time; 5.49 for the prover, above the ANDs' 3.62) and by their ANDs
    (memory)."""
    prove_lane = K1024_SESSION_S.lane * SESSION_SCALE
    prove_lane_gb = K1024_PROVE_GB.lane * ANDS_4096 / ANDS_1024
    lean_lane = K1024_LEAN_SESSION_S.lane * LEAN_SCALE
    return Rates(
        "today",
        f"{W2} (vy-nebius-cpu-1, 26 threads, K = 4,096, 8 lanes): prove {W2_PROVE_LOAD_S + W2_SESSION_S:.2f} s = "
        f"{W2_PROVE_LOAD_S:.1f} s load + {W2_SESSION_S} s session, {W2_PROVE_GB:.1f} GB with its session verifier; Lean "
        f"{W2_LEAN_PARSE_S:,} s parse + {W2_LEAN_SETUP_S} s setup + {W2_LEAN_SESSION_S:,} s session + "
        f"{W2_LEAN_TEARDOWN_S} s teardown, {W2_LEAN_GB} GB. A lane: {K1024_SESSION_S.run} (26 threads, K = 1,024: "
        f"sessions {K1024_SESSION_S.at8:.2f} s at 8 lanes, {K1024_SESSION_S.at64:.2f} s at 64; prover and verifier "
        f"{K1024_PROVE_GB.at8:.2f} → {K1024_PROVE_GB.at64:.2f} GB) and {K1024_LEAN_SESSION_S.run} (Lean sessions "
        f"{K1024_LEAN_SESSION_S.at8:.2f} s and {K1024_LEAN_SESSION_S.at64:.2f} s, 23.5 GB at both), so "
        f"{prove_lane:.2f} s, {prove_lane_gb:.3f} GB and {lean_lane:.3f} s of Lean a lane at K = 4,096",
        prove_load_s=W2_PROVE_LOAD_S, prove_session_s=W2_SESSION_S - 8 * prove_lane, prove_lane_s=prove_lane,
        prove_gb=W2_PROVE_GB - 8 * prove_lane_gb, prove_lane_gb=prove_lane_gb,
        lean_process_s=W2_LEAN_PARSE_S + W2_LEAN_SETUP_S + W2_LEAN_TEARDOWN_S,
        lean_session_s=W2_LEAN_SESSION_S - 8 * lean_lane, lean_lane_s=lean_lane, lean_gb=W2_LEAN_GB, lean_lane_gb=0.0)


RATES = {
    "int-period": Rates(
        "int-period", "INT-PERIOD's per-draw rates (RT-INT-3): W2's statement over its 8 draws, one tile a draw",
        prove_load_s=0.0, prove_session_s=0.0, prove_lane_s=66.0, prove_gb=W2_PROVE_GB, prove_lane_gb=0.0,
        lean_process_s=0.0, lean_session_s=0.0, lean_lane_s=446.0, lean_gb=W2_LEAN_GB, lean_lane_gb=0.0),
    "today": today_rates(),
}


@dataclass(frozen=True)
class Config:
    """A period's shape, rates and fleet. `t_window_s` is an 18-round window's wall-clock at 32×32 units; a window of M
    rounds takes M/18 of it, and with `receipts_per_unit` (torch receipts, 284 µs a unit) its receipts scale with the
    units, 1,024/`unit_outputs` of them for each 32×32 unit."""
    name: str
    u: float = U_TODAY
    r: float = R_SERVED
    rounds: int = ROUNDS
    draws: int = 8
    t_window_s: float = T_WINDOW
    unit_outputs: int = UNIT_ROWS * UNIT_COLS
    k: float = k_factor()
    rates: str = "today"
    lean_windows_per_process: int = 1
    prove_windows_per_process: int = 1
    lean_session_s: float | None = None
    receipts_per_unit: bool = True
    nodes: int = NODES
    eps_flat: float = EPS_FLAT
    delta_s_log2: float = DELTA_S_LOG2


def lam(draws: int, r: float, rounds: int) -> float:
    """Λ, the sharp split rate (`SplitRateExp`) at equal rounds (w_max = 1/M) and ρ = 1/r."""
    w, x = 1.0 / rounds, draws / (r * rounds)
    fixed = -math.log1p((1 - P) * math.expm1(-x))
    folded = -math.log1p((1 - P0) * math.expm1(-x))
    return max(A * w / fixed, B * w / folded)


def lam_limit(draws: int, r: float) -> float:
    """Λ's many-rounds limit, which it falls to from above: r·max(a/(1 − p), b/(1 − p0))/K_d."""
    return r * max(A / (1 - P), B / (1 - P0)) / draws


def eps(n: int, c: Config) -> float:
    return c.eps_flat + lam(c.draws, c.r, c.rounds) * c.delta_s_log2 * math.log(2) / (c.u * n)


def windows(c: Config) -> int:
    """The least n with ε(n) < 1% (INT-PERIOD's n_min)."""
    budget = TARGET - c.eps_flat
    if budget <= 0:
        raise ValueError(f"ε_♭ = {c.eps_flat:.3%} leaves no logical budget under {TARGET:.0%}")
    return math.ceil(lam(c.draws, c.r, c.rounds) * c.delta_s_log2 * math.log(2) / (c.u * budget))


def credit(j: int, c: Config) -> float:
    """The share of an accepted prefix of j windows' live words credited, (1 − ε(j))₊ (RT4-INT-8)."""
    return max(0.0, 1.0 - eps(j, c))


def first_credit(c: Config) -> int:
    """The least prefix that earns any credit: ε(j) < 1."""
    return math.floor(lam(c.draws, c.r, c.rounds) * c.delta_s_log2 * math.log(2) / (c.u * (1 - c.eps_flat))) + 1


def all_skip_pays(c: Config) -> bool:
    """Whether a window that skips every drawn tile whole passes (p0^K_d) more often than its credit is wrong (1 − b), so
    that, credited without ε's discount, it out-earns an honest window (RT4-INT-8: at K_d = 8, 1.95×)."""
    return P0 ** c.draws > 1 - B


def statement_lanes(rt: Rates) -> int:
    """The most lanes a statement holds: `m ≤ 35`'s 256, and what one node's memory gives its prover."""
    if rt.prove_lane_gb <= 0:
        return STATEMENT_LANES
    fit = int((NODE_GB - rt.prove_gb) // rt.prove_lane_gb)
    if fit < 1:
        raise ValueError(f"{rt.name}: one lane needs {rt.prove_gb + rt.prove_lane_gb:.0f} GB, past a {NODE_GB} GB node")
    return min(STATEMENT_LANES, fit)


def window_cost(c: Config) -> dict:
    """One window's statements, prove and Lean seconds and node-seconds, with the per-process parts amortized as
    configured."""
    rt = RATES[c.rates]
    lanes = c.draws * lanes_per_unit(c.unit_outputs, c.k)
    statements = max(1, math.ceil(lanes / statement_lanes(rt)))
    per = lanes / statements
    session = rt.lean_session_s if c.lean_session_s is None else c.lean_session_s
    prove = statements * (rt.prove_load_s / c.prove_windows_per_process + rt.prove_session_s) + lanes * rt.prove_lane_s
    lean = statements * (rt.lean_process_s / c.lean_windows_per_process + session) + lanes * rt.lean_lane_s
    prove_slots = max(1, int(NODE_GB // (rt.prove_gb + per * rt.prove_lane_gb)))
    lean_slots = max(1, int(NODE_GB // (rt.lean_gb + per * rt.lean_lane_gb)))
    return dict(lanes=lanes, statements=statements, prove_s=prove, lean_s=lean, prove_slots=prove_slots,
                lean_slots=lean_slots, node_s=prove / prove_slots + lean / lean_slots)


def window_s(c: Config) -> float:
    """One window's serving wall-clock."""
    units = UNIT_ROWS * UNIT_COLS / c.unit_outputs if c.receipts_per_unit else 1.0
    return c.t_window_s * c.rounds / ROUNDS * units


def period(c: Config) -> dict:
    """What the first period at ε < 1% takes under `c`."""
    n = windows(c)
    w = window_cost(c)
    serving_days = n * window_s(c) / DAY
    node_days = n * w["node_s"] / DAY
    cert_days = node_days / c.nodes
    statement_s = (w["prove_s"] + w["lean_s"]) / w["statements"]
    last_s = max(statement_s, w["node_s"] / c.nodes)
    return dict(
        name=c.name, windows=n, draws=n * c.draws, lanes=n * w["lanes"], rounds=n * c.rounds,
        eps=eps(n, c), lam=lam(c.draws, c.r, c.rounds), credit=credit(n, c), first_credit=first_credit(c),
        all_skip_pays=all_skip_pays(c),
        statements_per_window=w["statements"], prove_s_per_window=w["prove_s"], lean_s_per_window=w["lean_s"],
        prove_slots=w["prove_slots"], lean_slots=w["lean_slots"],
        prove_process_days=n * w["prove_s"] / DAY, lean_process_days=n * w["lean_s"] / DAY,
        node_days=node_days, serving_days=serving_days, cert_days=cert_days,
        days_to_period=max(serving_days, (n - 1) * w["node_s"] / c.nodes / DAY) + last_s / DAY,
        nodes_to_keep_pace=node_days / serving_days,
        window_custody_gb=CUSTODY_GB * c.rounds / ROUNDS,
        config=dataclasses.asdict(c),
    )


@dataclass(frozen=True)
class Lever:
    """One change, its owner and work class. A field given as a tuple is tuned: it takes whichever of its values most
    shortens the period on the configuration it's applied to."""
    name: str
    owner: str
    work: int
    what: str
    change: dict


DRAWS = (4, 8, 16, 32, 64, 128, 256)
ROUNDS_SCAN = (9, 18, 36, 72)

LEVERS = [
    Lever("u: every row slot of a served step filled", "vllm", 2,
          "64 sequences in flight a step instead of 5: u from 119/1152 to 16/18 (the 2 empty rounds carry none)",
          dict(u=U_FULL)),
    Lever("fused receipt kernel", "kernels", 3,
          "a window at #2790's credited Dither TPOT, 18 × 66.65 ms = 1.2 s (a different receipt form, not yet accepted by "
          "a verifier on main), instead of 439.1 s of launch-bound torch receipts, at any unit size",
          dict(t_window_s=T_WINDOW_DITHER, receipts_per_unit=False)),
    Lever("draws per window", "compute-accounting", 1,
          "the service challenger's K_d (H1_DRAW), the best of 4 to 256: more draws mean fewer windows and so fewer "
          "statements' fixed costs, but more draws a period once x = K_d/(r·M) is not small",
          dict(draws=DRAWS)),
    Lever("Lean: one process checks many windows", "proofs", 2,
          "flock-verify takes many windows' public inputs and sessions in one process, so a circuit is parsed once in "
          "64 windows instead of once a statement",
          dict(lean_windows_per_process=64)),
    Lever("prover: one process proves many windows", "C-Flock", 2,
          "flock-circuit-prove keeps the unrolled circuit loaded across windows' statements",
          dict(prove_windows_per_process=64)),
    Lever("r → 1: units of equal work", "kernels + circuits", 3,
          "RT-INT-2's tile-width fix: every unit's words equal (down's units 3.5× narrower), r from 2.87 to 1 and every "
          "unit at the K = 4,096 unit's work",
          dict(r=1.0, k=1.0)),
    Lever("served unit 32×32 → 4×8", "circuits + kernels", 3,
          "the receipt's unit is the measured tile, so a draw is one lane, not 32·K/4,096: the period's draws are set "
          "by the lucks, not the unit's size; 32× the receipts and leaves a window, so 32× the torch receipts' time "
          "without the fused kernel (custody here stays the 32×32 window's)",
          dict(unit_outputs=LANE_OUTPUTS)),
    Lever("Lean session at O(description)", "proofs", 4,
          "the Lean session check at the Rust in-session verifier's cost (11.2 s in W2, not 1,175.7 s), the "
          "verifier-cost rule's O(circuit description)",
          dict(lean_session_s=11.2)),
    Lever("rounds per window", "vllm + compute-accounting", 1,
          "the window's served steps, the best of 9 to 72 rounds (a window's wall-clock scales with them): at small "
          "x = K_d/(r·M), Λ is already near its many-rounds limit b·r/((1 − p0)·K_d)",
          dict(rounds=ROUNDS_SCAN)),
    Lever(SPEND, "Daniel (spend)", 1, "eight CPU nodes certify windows in parallel instead of two", dict(nodes=8)),
]

TODAY = Config("today: SERVE-H1's window, served-unit draws, per-statement rates")


def lever(name: str) -> Lever:
    return next(lv for lv in LEVERS if lv.name == name)


def tuned(lv: Lever) -> bool:
    return any(isinstance(v, tuple) for v in lv.change.values())


def apply(lv: Lever, c: Config) -> Config:
    """`c` with `lv`'s change, its tuned fields at their best values, named for the lever and what it chose."""
    fixed = {k: v for k, v in lv.change.items() if not isinstance(v, tuple)}
    tune = {k: v for k, v in lv.change.items() if isinstance(v, tuple)}
    best = min((replace(c, **fixed, **dict(zip(tune, vs))) for vs in itertools.product(*tune.values())),
               key=lambda x: period(x)["days_to_period"])
    chose = ", ".join(f"{k} = {getattr(best, k)}" for k in tune)
    return replace(best, name=f"+ {lv.name}" + (f" ({chose})" if chose else ""))


def route(base: Config = TODAY) -> list[Config]:
    """The fastest route: from `base`, the lever that most shortens the time to the first period, then the next on
    that, while one shortens it by more than 1%. A tuned lever may be applied again later; the spend lever is left
    out (spend is a budget decision, not a tuning step)."""
    c, left, out = base, [lv for lv in LEVERS if lv.name != SPEND], []
    while left:
        steps = [(apply(lv, c), lv) for lv in left]
        nxt, best = min(steps, key=lambda s: period(s[0])["days_to_period"])
        if period(nxt)["days_to_period"] >= 0.99 * period(c)["days_to_period"]:
            break
        out.append(nxt)
        if not tuned(best):
            left.remove(best)
        c = nxt
    return out


NEAR_TERM = ("u: every row slot of a served step filled", "r → 1: units of equal work",
             "Lean: one process checks many windows", "prover: one process proves many windows", "draws per window")


def near_term(base: Config = TODAY) -> list[Config]:
    """`NEAR_TERM`'s levers, which need no new receipt kernel, served unit or proof form, in order, each on the last;
    then the last on eight nodes (the spend lever), its samples tuned again."""
    out, c = [], base
    for name in NEAR_TERM:
        c = apply(lever(name), c)
        out.append(c)
    spend = apply(lever("draws per window"), apply(lever(SPEND), c))
    return [*out, replace(spend, name=f"+ {SPEND} (draws = {spend.draws})")]


def configurations(base: Config = TODAY) -> list[Config]:
    """INT-PERIOD's line, `base`, and `route`'s steps, each on the last."""
    line = Config("INT-PERIOD's line: per-draw rates, W2's tile a draw", rates="int-period", unit_outputs=LANE_OUTPUTS,
                  k=1.0, receipts_per_unit=False)
    return [line, base, *route(base)]


def single_levers(base: Config = TODAY) -> list[dict]:
    """Each lever alone on `base`: its days and node-days to the first period, and the factor on each."""
    b = period(base)
    rows = []
    for lv in LEVERS:
        c = apply(lv, base)
        p = period(c)
        rows.append(dict(lever=c.name[2:], owner=lv.owner, work=lv.work, what=lv.what, windows=p["windows"],
                         days=p["days_to_period"], node_days=p["node_days"],
                         days_factor=b["days_to_period"] / p["days_to_period"],
                         node_days_factor=b["node_days"] / p["node_days"]))
    return sorted(rows, key=lambda r: (-max(r["days_factor"], r["node_days_factor"]), r["work"]))


LONG_DRAWS = (1.05, 1.1, 1.25, 1.5, 2, 3, 4, 8)


def least_one_window_draws(c: Config) -> int:
    """The least K_d for which some number of rounds brings one window under 1%."""
    need = lam_limit(1, c.r) * c.delta_s_log2 * math.log(2) / (c.u * (TARGET - c.eps_flat))
    return math.floor(need) + 1


def long_window(base: Config) -> Config:
    """One window long enough on its own: of K_d at `LONG_DRAWS` times `least_one_window_draws`, each at the least
    rounds that bring it under 1% (n falls with M), whichever ends soonest."""
    best, k0 = None, least_one_window_draws(base)
    for k in sorted({math.ceil(k0 * f) for f in LONG_DRAWS}):
        hi = 1
        while windows(replace(base, draws=k, rounds=hi)) > 1:
            hi *= 2
        lo = hi // 2 + 1
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if windows(replace(base, draws=k, rounds=mid)) == 1 else (mid + 1, hi)
        c = replace(base, draws=k, rounds=hi, name=f"one window: {base.name} (draws = {k:,}, rounds = {hi:,})")
        if best is None or period(c)["days_to_period"] < period(best)["days_to_period"]:
            best = c
    return best


def scan(c: Config, field: str, values) -> list[dict]:
    """`c` at each value of one field: windows, days and node-days to the first period."""
    out = []
    for v in values:
        p = period(replace(c, **{field: v}))
        out.append(dict(value=v, windows=p["windows"], days=p["days_to_period"], node_days=p["node_days"]))
    return out


def sensitivity() -> dict:
    """Windows to ε < 1% at today's served shape and u = 1, by δ_s (the claim's) and by ε_♭ (the machine model's)."""
    base = Config("sensitivity", u=1.0)
    return dict(
        delta_s_log2={str(d): windows(replace(base, delta_s_log2=d)) for d in (20, 30, 40, 64, 80)},
        eps_flat={f"{e:.3%}": windows(replace(base, eps_flat=e)) for e in (0.003, 0.00571, 0.007, 0.008, 0.009)},
    )


def table(rows: list[dict]) -> str:
    lines = ["| configuration | windows | first credit | all-skip pays | draws | lanes | statements a window "
             "| prove + Lean s a window | prove + Lean process-days | node-days | serving d | certifying d "
             "| days to period | nodes to keep pace |",
             "|---" * 14 + "|"]
    for p in rows:
        lines.append(
            f"| {p['name']} | {p['windows']:,} | {p['first_credit']:,} | {'yes' if p['all_skip_pays'] else 'no'} | "
            f"{p['draws']:,} | {p['lanes']:,.0f} | {p['statements_per_window']} | "
            f"{p['prove_s_per_window']:,.0f} + {p['lean_s_per_window']:,.0f} | "
            f"{p['prove_process_days']:,.1f} + {p['lean_process_days']:,.1f} | {p['node_days']:,.1f} | "
            f"{p['serving_days']:,.2f} | {p['cert_days']:,.2f} | {p['days_to_period']:,.2f} | "
            f"{p['nodes_to_keep_pace']:,.1f} |")
    return "\n".join(lines)


def lever_table(rows: list[dict]) -> str:
    lines = ["| lever | owner | work | windows | days to period | × days | node-days | × node-days |",
             "|---" * 8 + "|"]
    for r in rows:
        lines.append(f"| {r['lever']} | {r['owner']} | {r['work']} | {r['windows']:,} | {r['days']:,.2f} | "
                     f"{r['days_factor']:,.2f} | {r['node_days']:,.1f} | {r['node_days_factor']:,.2f} |")
    return "\n".join(lines)


def report() -> dict:
    configs = configurations()
    end = configs[-1]
    return dict(
        rates={k: dataclasses.asdict(v) for k, v in RATES.items()},
        lanes_per_unit=lanes_per_unit(), statement_lanes={k: statement_lanes(v) for k, v in RATES.items()},
        node_gb=NODE_GB, configurations=[period(c) for c in configs], near_term=[period(c) for c in near_term()],
        levers=single_levers(), route_end=end.name,
        draws_scan=scan(end, "draws", DRAWS), rounds_scan=scan(end, "rounds", ROUNDS_SCAN),
        long_windows=[period(long_window(c)) for c in (configs[1], configs[3], end)],
        sensitivity=sensitivity(),
    )


def markdown(rep: dict) -> str:
    out = [f"A drawn served unit is {rep['lanes_per_unit']:.1f} lanes of the measured tile; a statement holds at most "
           f"{rep['statement_lanes'][TODAY.rates]} lanes at `{TODAY.rates}`'s rates; a node has {NODE_GB} GB.", ""]
    out += [f"- rates `{k}`: {rt['source']}" for k, rt in rep["rates"].items()]
    out += ["", table(rep["configurations"]), "",
            "Near term (no new receipt kernel, served unit or proof form), each on the last, then on eight nodes:", "",
            table(rep["near_term"]), "",
            "Each lever alone on today's configuration (work: 1 a parameter, 2 one module, 3 a new path, 4 a new form):",
            "", lever_table(rep["levers"]), ""]
    for key, field in (("draws_scan", "draws"), ("rounds_scan", "rounds")):
        out.append(f"At the route's end, by {field} a window: " + "; ".join(
            f"{r['value']}: {r['windows']:,} windows, {r['days']:,.2f} d, {r['node_days']:,.1f} node-d" for r in rep[key]))
    out += ["", "One window long enough on its own (no period), with the custody it holds until its draw:", ""]
    out += [f"- {p['name']}: {p['days_to_period']:,.2f} days ({p['serving_days']:,.2f} serving), "
            f"{p['node_days']:,.1f} node-days, {p['statements_per_window']:,} statements, "
            f"{p['window_custody_gb']:,.0f} GB custody" for p in rep["long_windows"]]
    s = rep["sensitivity"]
    out += ["", "Windows to ε < 1% at u = 1, by δ_s: "
            + ", ".join(f"2^-{k}: {v:,}" for k, v in s["delta_s_log2"].items()),
            "... by ε_♭: " + ", ".join(f"{k}: {v:,}" for k, v in s["eps_flat"].items())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write the whole report here")
    ap.add_argument("--md", type=Path, help="write the tables here")
    a = ap.parse_args(argv)
    rep = report()
    md = markdown(rep)
    print(md, end="")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n")
    if a.md:
        a.md.write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
