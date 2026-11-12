"""`verity bench served_errors [--json]`: error terms of the quoted guarantees that had no value at the sizes we serve,
evaluated there. Standard library only; it reads the workload card and nothing else.

Each row is one term: the guarantee it sits in, the formula, the parameters, and its value as the log2 of a probability,
for one window, proof or block, against a prover of `2^80` SHA-512 calls unless the row says otherwise. Every value is
modeled: a formula from the Lean (or a reading of one, which the row names) evaluated at stated sizes. A row whose
formula rests on an open premise names it; evaluating the formula discharges nothing.

The served sizes it reads:

- PoUW: Pearl+ on Llama 3.2 1B on one H200, credited under the H100 record (`pearl-c-h100-v1-h3`, `devH100`) at the
  inert-work share 1/400: four passes of 64 calls (prefill at m = 128, three decode passes at m = 4) over the model's
  four linears in 16 layers, 29,440 credited 64 x 64 tiles, 459 work-weighted draws, sampling δ 1/100.
- Sampled proofs: the GLM-5.3 workload card (`benchmarks/spot_figures/cards/workload.glm-5.3.json`): its window of one
  request (the floor), and one request of 3,320.5 computed tokens (`benchmarks/spot_figures/breakeven/reprice.py`), at
  the card's cut.
- Monitoring periods: `verity/protocols/reporting.py`'s week, month and year, in hourly phases.
"""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

lg = math.log2
CARD = Path(__file__).resolve().parents[1] / "spot_figures" / "cards" / "workload.glm-5.3.json"
#: the prover's budget in SHA-512 calls, log2; each row also gives `2^64` where the budget moves it
T = 80
#: hourly phases in a week, a month and a year (`verity/protocols/reporting.py`'s PERIODS)
PHASES = {"week": 7 * 24, "month": 30 * 24, "year": 365 * 24}

# Flock, one `--zk` session (`security/Proofs/Flock/Claims/NumbersDeploy.lean`): κ ≤ n·2^-202.98 at n tables, at most
# 2 + 4400·n rounds and a weight of n·2^33; γ ≤ 2^51 at one table (`γ_le_51`), γ ≤ 16(√(2rC) + r)² in general.
KAPPA_TABLE = -202.98
GAMMA_ONE = 51.0
TABLE_CAP = 64

# The recursion's outer proof (`security/Proofs/Flock/Claims/Recursion/KappaEpsCap.lean`): Pr[acc ∧ ¬R] ≤ κ + t·ε.
REC_KAPPA, REC_EPS = -167.7, -174.95

# Pearl+ as served.
PASSES = (128, 4, 4, 4)
LAYERS = 16
LINEARS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}
TILES = 29440
DRAWS = 459
RHO = Fraction(1, 400)
# rev1's price of one unit row at the H100 record (`devH100`, `Prices.h100`; `security/Definitions/Pouw/PearlC/DeviceRev1.lean`)
H100 = {"r": 32, "G": 4, "add": 32, "bf16": 2, "fs": 64, "qa": 4, "c": 0}
G0 = Fraction(1, 400)

# The deployed PoUW window (`verity/protocols/spot/verifier/profile.py` WINDOW_DELTA) and the served run's sampling δ.
WINDOW_DELTA = Fraction(1, 2**40)
SERVED_DELTA = Fraction(1, 100)
# CircuitCredit's draw count (`security/Definitions/Accounting/Guarantees/Compose.lean`).
COMPOSE_KD = 96000
# The Sm120 headlines' window (`security/Definitions/Accounting/Guarantees/Headline.lean`): checks·(√δ_fork + n_π·δ_ZK) ≤ 2^-41.
SM120_CHECKS, SM120_BUDGET = 48000, -41.0

# Dither's design point (`verity/protocols/pouw/verifier/dither/PROTOCOL.md`; q at `Guarantees/Dither.lean`).
DITHER = {"delta_coin": 1e-6, "G": 1.15e7, "q": 13816}

# SPOT's link term as the sampled audit states it (`EndToEndHiddenDrawn`'s δ_link), at k = ⌈2^24/e⌉ and M, R_w at 2^192.
LINK_K = math.ceil(2**24 / math.e)
TOKENS_REQUEST = 3320.5


def l2sum(*xs: float) -> float:
    m = max(xs)
    return m + lg(sum(2.0 ** (x - m) for x in xs))


def session_ks(t: float = T, tables: int = 1) -> float:
    """One `--zk` session's knowledge error, κ + √(γ·Pr[B]), B two runs of `2^t` calls under `StrictBirthday`."""
    if tables == 1:
        gamma = GAMMA_ONE
    else:
        r, c = 2 + 4400 * tables, tables * 2.0**33
        gamma = lg(16 * (math.sqrt(2 * r * c) + r) ** 2)
    fork = 0.5 * (gamma + 2 * (t + 1) - 513)
    return l2sum(KAPPA_TABLE + lg(tables), fork)


def pearl_gap(n: int, k: int) -> Fraction:
    """rev1's gap at the H100 record on a linear `(n, k)`, any m: g = 1 − (1 − ρ)(1 − g₀)·credit/W_ref."""
    p = H100
    credit = (k * n * (1 + Fraction(p["add"], 32 * p["G"])) + k * p["r"] + 2 * p["bf16"] * p["r"] * n + p["fs"] * k
              - p["add"] * n)
    wref = credit + (p["qa"] + p["c"]) * k
    return 1 - (1 - RHO) * (1 - G0) * credit / wref


def delta_pearl(q: float, n: int = TILES) -> float:
    """`δPearlC(q, N) = (q + N)/2^128`, log2, at `q` oracle calls (the statement covers `q ≤ 2^64`)."""
    return lg(2.0**q + n) - 128


def sampling_eps(delta: Fraction, draws: int = DRAWS) -> float:
    return 1 - float(delta) ** (1 / draws)


def operand_positions() -> int:
    """Every operand element the served run's tiles read: each call's A (m x k) and B (n x k)."""
    return sum(LAYERS * (m * k + n * k) for m in PASSES for n, k in LINEARS.values())


def card_sizes() -> dict[str, dict[str, float]]:
    """The card's units and input-and-output bits (log2) at the floor and at one request of `TOKENS_REQUEST` tokens."""
    v = json.loads(CARD.read_text())["value"]
    units = sum(s["per_token"] for s in v["strata"])
    bits = sum(s["per_token"] * (s["in_bits"] + s["out_bits"]) for s in v["strata"])
    return {name: {"n": lg(units * tok), "W": lg(bits * tok)} for name, tok in (("floor", v["tokens"]),
                                                                                ("request", TOKENS_REQUEST))}


def link(w: float, t: float = T) -> float:
    """SPOT's link term at `2^w` input and output bits: W·(2t(1+k)/2^256 + 1/(eM) + k/(eR_w))."""
    return l2sum(w + 1 + t + lg(1 + LINK_K) - 256 + 2, w - lg(math.e) - 192 + 1, w + lg(LINK_K) - lg(math.e) - 192 + 1)


def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def row(term: str, guarantee: str, formula: str, at: str, value: float | str, rests_on: str = "") -> None:
        out.append({"term": term, "guarantee": guarantee, "formula": formula, "at": at,
                    "log2": round(value, 2) if isinstance(value, float) else value, "rests_on": rests_on})

    s1, s64 = session_ks(T, 1), session_ks(T, TABLE_CAP)
    row("flock session KS", "Flock.Guarantees.MaskedKS (one --zk session)", "κ + √γ·2T/2^256.5",
        f"T = 2^{T}, 1 table / {TABLE_CAP} tables", f"{s1:.2f} / {s64:.2f}", "StrictBirthday; ListWithin")

    ks_terms = "TileServiceSound / ServiceSound (open)"
    for kd, what in ((DRAWS, "Pearl+'s draw"), (COMPOSE_KD, "CircuitCredit's K_d")):
        row("pouw δ_ks", "Properties.Pouw.Period h1CheckedCreditPeriod, H1J10RCheckedCreditPeriod",
            "K_d · one session's KS (each session 2^80 calls)", f"K_d = {kd} ({what}), {TABLE_CAP} tables",
            lg(kd) + s64, ks_terms)
    row("pouw δ_ks", "same", "κ·K_d + √γ·2T/2^256.5 (the window's 2^80 calls shared)", f"K_d = {COMPOSE_KD}, {TABLE_CAP} tables",
        l2sum(KAPPA_TABLE + lg(TABLE_CAP) + lg(COMPOSE_KD), s64), ks_terms)
    row("pouw δ_ks", "Pearl+'s check as served (tiles opened on SHA-512 trees)", "q²/2^513 (a SHA-512 collision)",
        f"q = 2^{T}", 2 * T - 513, "StrictBirthday")

    pos = operand_positions()
    bind_el, bind_word = lg(pos) + (T - 256) / 2, lg(pos / 4) + (T - 256) / 2
    row("pouw δ_bind", "Pouw.Guarantees.PeriodBindingPerProver", "|L|·√(q/2^256 + κ), κ = 0",
        f"|L| = {pos:,} operand bytes (or /4 in 32-bit words), q = 2^{T}", f"{bind_el:.2f} / {bind_word:.2f}",
        "ExpectedCR; κ unbounded on main")

    ks_pouw = lg(DRAWS) + s64
    for d, what in ((WINDOW_DELTA, "WINDOW_DELTA"), (SERVED_DELTA, "the served run's δ")):
        p = l2sum(lg(d), bind_el, ks_pouw)
        row("pouw leg p", "Nci.Guarantees.RemotePeriod (p = δ_s + δ_bind + δ_ks)", "δ_s + δ_bind + δ_ks a window",
            f"δ_s = {what}, δ_bind at bytes, δ_ks at K_d = {DRAWS}", p, "PouwSound's bridge, not stated")

    nmax = SM120_BUDGET - lg(SM120_CHECKS) - s64
    row("sm120 n_π·δ_ZK", "Accounting.Guarantees.TS.Sm120SheetHeadline and siblings (ProofSoundAt)",
        "n_π · one session's KS, against 2^-41/checks", f"n_π = 1, {TABLE_CAP} tables; budget 2^{SM120_BUDGET - lg(SM120_CHECKS):.2f}",
        s64, f"holds for n_π up to 2^{nmax:.1f}; Flock's bound omits the rows link ProofSoundAt reads")

    gaps = {name: pearl_gap(n, k) for name, (n, k) in LINEARS.items()}
    g = max(gaps.values())
    eps_a = g / (1 - g)
    row("pearl+ ε_a", "Pouw.PearlC pearlCGapH100Rev1_llama32_1b (lemma, unlocked)", "g/(1 − g), the max over the four linears",
        "H100 record, ρ = 1/400", f"{float(eps_a):.5%} ({eps_a})", "TTOutTilePearlCH100Rev1 (open conjecture)")
    for q in (64, T):
        row("pearl+ δ_a", "same", "(q + N)/2^128", f"q = 2^{q}, N = {TILES:,} tiles" + (" (outside q ≤ 2^64)" if q > 64 else ""),
            delta_pearl(q), "TTOutTilePearlCH100Rev1")
    for d in (SERVED_DELTA, WINDOW_DELTA):
        e = (1 + float(eps_a)) / (1 - sampling_eps(d)) - 1
        row("pearl+ ε", "same, with the sample", "(1 + ε_a)/(1 − ε_s) − 1, ε_s = 1 − δ^(1/t)", f"t = {DRAWS}, δ_s = {d}",
            f"{e:.3%}", "")

    for t in (T, 64):
        one = l2sum(REC_KAPPA, t + REC_EPS)
        week = one + lg(DITHER["q"])
        budget = lg(DITHER["delta_coin"] / DITHER["G"] / DITHER["q"])
        row("dither κ_S", "Pouw.Guarantees.Dither WorkSound (ProofSystemSound, open)",
            "κ + t·ε of one outer proof (RecursiveKSDeployed), q proofs a replica-week",
            f"t = 2^{t}; budget 2^{budget:.2f} a proof", f"{one:.2f} a proof / {week:.2f} a replica-week",
            "ProofSystemSound has no lemma from RecursiveKSDeployed; ListWithin, Hm96Hiding, DenseAt")

    sizes = card_sizes()
    for name, sz in sizes.items():
        ks = l2sum(sz["n"] + KAPPA_TABLE, sz["n"] + s1)
        ks64 = l2sum(sz["n"] + KAPPA_TABLE + lg(TABLE_CAP), sz["n"] + s64)
        lk = link(sz["W"])
        kst = l2sum(ks, lk)
        row("integrity ks_t", "Nci.Guarantees.RemotePeriod (ks t = ε_ks + δ_link, EndToEndHiddenDrawn)",
            "n·(κ + fork) + W·(2T(1+k)/2^256 + …)",
            f"card {name}: n = 2^{sz['n']:.2f} units, W = 2^{sz['W']:.2f} bits, 1 / {TABLE_CAP} tables",
            f"{kst:.2f} / {l2sum(ks64, lk):.2f}", "StrictBirthday; ecr/sha-512 at the link finders")
        if name == "floor":
            for period, n in PHASES.items():
                row("integrity Σ ks_t", "Nci.Guarantees.RemotePeriod", "Σ_t ks t over hourly windows",
                    f"card floor, a {period} ({n} windows)", kst + lg(n), "")
            ks_floor = kst

    crypto = l2sum(delta_pearl(64), 2 * T - 513, ks_floor)
    row("joint, statistical", "Accounting.Guarantees.JointPeriodLive at the remote product (integrity + PoUW)",
        "Σ_a δ_a a block", "δ = 1/100 a leg", "0.02 a block", "by design")
    row("joint, cryptographic", "same", "PoUW's δ_a + opening binding + integrity's ks_t", "q = 2^64, card floor", crypto,
        "TTOutTilePearlCH100Rev1 at q ≤ 2^64")
    for period, n in PHASES.items():
        row("joint, cryptographic", "same", "over hourly blocks", f"a {period} ({n} blocks)", crypto + lg(n), "")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    a = ap.parse_args(argv)
    rs = rows()
    if a.json:
        print(json.dumps(rs, indent=1, ensure_ascii=False))
        return 0
    for r in rs:
        print(f"{r['term']:22s} {str(r['log2']):>34s}   at {r['at']}")
        print(f"{'':22s} {r['guarantee']}: {r['formula']}" + (f"  [{r['rests_on']}]" if r["rests_on"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
