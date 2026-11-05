"""The PoUS average-memory explorer's data file, `pous-explorer/v1`: the reference math of the certified average-ε bound,
the honest-latency presets as survival curves, the schemes' serving overhead, `ε_proof` by in-degree, the Poisson quantile
table the page looks `f₀` up in, and test vectors.

The model (the explorer note, §1): challenges arrive at Poisson rate λ over a game of T seconds; an answer is a miss if it
is wrong or later than τ = (d + 1) · call_ns(b) / m, just under the band's cliff; the memory challenger accepts iff the
misses F ≤ f₀, the (1 − α) quantile of the honest miss count, Poisson with mean λT · S_H(τ).  Then P[accept and ε̄ >
ε_proof(d) + g(f₀)] ≤ β with g(f) = min over θ > 0 of (θf + ln(1/β)) / ((1 − e^−θ) λT).  The honest model sets only the
false-alarm rate, never soundness.  ε_proof(d) is an ideal-permutation-model bound, and τ counts whole `Π₂` calls of
`call_ns`, the model's unit: at the reference Feistel-SHAKE256 `Π₂` a prover rebuilds a label from 1.53% of `C` in 481
Keccak-f, about a tenth of a call, so neither bounds an adversary there (the red team's state attack,
`verity.core.protocols.pous.primitives.pi2_keccak_f`).

The page evaluates the file with plain floats and closed forms only (§6 "The rules"), and `evaluate` is that evaluation
here, from the file alone:

1. D = min(d, the deepest D in `eps_proof`), since the memory challenger refuses a deadline past its certificate; then
   τ_us = (D + 1) · call_ns / m / 1000 − 0.001.
2. p = S(τ_us), ln S interpolated linearly in ln t between the curve's points; S = 1 below the grid, the last segment's
   slope above it; S floored at `S_FLOOR`.
3. μ = λT · p; f₀ is the smallest f with `f0_max_mean[α][f] ≥ μ`, past the table Bernstein's
   f₀ = ⌈μ + L/3 + √(L²/9 + 2μL)⌉ − 1 with L = ln(1/α).
4. With L = ln(1/β): sampling = L / (λT) when f₀ = 0, else θ = ln(1 + x + √(2x)) with x = L / f₀ and
   sampling = (θf₀ + L) / ((1 − e^−θ) λT).  Any θ bounds; `g_opt` is the optimum the closed form is checked against.
5. bound = ε_proof(D) + sampling.

    uv run python benchmarks/pous/explorer.py --out pous-explorer.json [--histograms verify-honest.json ...]
        [--soak pous-soak-v1.json=art:… [--soak-note TEXT] [--soak-raw pous-soak-full.json=art:…]]

`--histograms` pools the series' `verify-honest.json` histograms into the `measured` preset (two runs or more); without it
`measured` is run 1's quantiles and window counts (`RUN1`).  `--soak` builds the off-host presets from a `pous-soak/v1`
file's ToR-equivalent curves (`soak_presets`): `rack` from `tor_400_tuned`, `rack-100g` from `tor_100_default`, and
`datacenter` as `tor_100_default` plus its fabric round trip; each block from its own cells, not shifted, each audit rate
weighted equally.  `--soak-note` ends each of their descriptions with what the file's run was, and `--soak-raw` (the soak
file before the runs it leaves out) with the S at d = 0 those runs would give.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pous-explorer/v1"
BLOCK_BYTES = (65_536, 147_456, 393_216)
ALPHAS = (1e-2, 1e-3, 1e-6)
BETAS = ALPHAS
F0_TABLE_LEN = 1000
S_FLOOR = 1e-12
#: the curves' grid: 10 µs to 100 ms, 40 points per decade (the series histograms' edges 40 to 200)
POINTS_PER_DECADE = 40
T_US = tuple(10 ** (1 + k / POINTS_PER_DECADE) for k in range(4 * POINTS_PER_DECADE + 1))
#: a block longer than 64 KiB adds this much honest latency per KiB (device copy plus transfer)
US_PER_KIB = 0.15
#: z of the lognormal's 99th percentile
Z99 = 2.3263478740408408

#: ε_proof, the per-answer slack, by the adversary's effective call count D = d: estimates at 64 KiB (from
#: `Security/Proofs/Pous/Chain/Family.lean`'s certificate, not proved; D = 0 is the extracted single-segment
#: value ≈ 0.21%)
EPS_PROOF = (
    {"D": 0, "eps_proof_fraction": 0.0021, "status": "extracted", "source": "single-segment ε_proof(0) ≈ 0.21% from Chain/Family.lean's certificate (replaces the D = 1 placeholder 0.40%)"},
    {"D": 1, "eps_proof_fraction": 0.0040, "status": "estimate", "source": "estimated from Chain/Family.lean's certificate at k 93"},
    {"D": 2, "eps_proof_fraction": 0.0060, "status": "estimate", "source": "estimated from Chain/Family.lean's certificate at k 97"},
    {"D": 3, "eps_proof_fraction": 0.0081, "status": "estimate, upper", "source": "estimated from Chain/Family.lean's certificate at k 101"},
    {"D": 4, "eps_proof_fraction": 0.0101, "status": "estimate, upper", "source": "estimated from Chain/Family.lean's certificate at k 106"},
    {"D": 5, "eps_proof_fraction": 0.0120, "status": "estimate, upper", "source": "estimated from Chain/Family.lean's certificate at k 111"},
)

#: the weight sets the schemes were screened on (`weight_sets.json`); overheads compare only within one set, since each
#: set's plain arm streams at its own rate (761 GB/s on qwen2.5-0.5b, about 1,430 GB/s on the Llama sets)
WEIGHT_SETS = (
    {"id": "qwen2.5-0.5b/linears+head", "label": "Qwen2.5-0.5B, every linear + head", "bytes": 987_922_432},
    {"id": "llama3.1-8b/layer0+head", "label": "Llama-3.1-8B, layer 0 + head", "bytes": 1_486_880_768},
    {"id": "llama3.1-8b/layers0-8+head", "label": "Llama-3.1-8B, layers 0-8 + head", "bytes": 4_976_541_696},
    {"id": "llama3.1-8b/linears+head", "label": "Llama-3.1-8B, every linear + head", "bytes": 15_009_316_864},
)

#: serving overhead (POUS / plain − 1 at T tokens per weight pass): untimed screens on node 2, RTX PRO 6000, the A9
#: kernel, each art's `decode_overhead_T*` measurements to 2 decimals
_QWEN, _L0, _L9, _LFULL = (w["id"] for w in WEIGHT_SETS)
SCHEMES = (
    {"weight_set": _QWEN, "d": 0, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 28.17, "8": 32.92, "32": 31.97, "2048": 4.88},
     "art": "art:f26dcc43985e186972d1408f40d4337c8db2a76a220e082a06e61e20236a3948"},
    {"weight_set": _QWEN, "d": 1, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 71.47, "8": 59.82, "32": 58.23, "2048": 8.94},
     "art": "art:72529776ba02dc36228de198f6a58e5fb630712edd75350c5842cd64bd1254b9"},
    {"weight_set": _QWEN, "d": 2, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 69.14, "8": 80.24, "32": 77.84, "2048": 11.87},
     "art": "art:0daa432488c965e8ef090e73b2005be6b6e25f9bc6d89a63e49d34d706a84360"},
    {"weight_set": _QWEN, "d": 5, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 132.27, "8": 153.4, "32": 148.9, "2048": 22.8},
     "art": "art:88ea542b5f887b9ca25422bb58e05e2e9fbf3d3d149e1191d10d8ca559829c83", "note": "a timed run"},
    {"weight_set": _QWEN, "d": 12, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 376.39, "8": 314.44, "32": 305.56, "2048": 46.88},
     "art": "art:4354a29314cf3e25917e1ddbc30fda90807650b4d6c94caceffbf3ad8ab80924"},
    {"weight_set": _QWEN, "d": 0, "block_bytes": 147_456, "kind": "measured",
     "overhead_x_by_tokens": {"1": 40.88, "8": 47.74, "32": 46.42, "2048": 7.09},
     "art": "art:529b4a35c4a1fb9b3c63cc0ff51690897832a46b60c4dea9ad931d6c83d1362c",
     "note": "too few rows to fill the GPU"},
    {"weight_set": _QWEN, "d": 0, "block_bytes": 393_216, "kind": "measured",
     "overhead_x_by_tokens": {"1": 103.58, "8": 120.18, "32": 116.62, "2048": 17.84},
     "art": "art:06d9d7a96e3bc729ab3f5765bf320fcf52400963b581c181f6d4ba784e0ad281",
     "note": "too few rows to fill the GPU"},
    {"weight_set": _L0, "d": 12, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 329.96, "8": 357.74, "32": 514.27, "2048": 65.18},
     "art": "art:0c524d678607a65e0c06a9d01c400525ebc47c9748561f45f511e064c995b55e",
     "note": "T=1 and T=8 understate: their plain baselines ran below memory bandwidth"},
    {"weight_set": _L9, "d": 0, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 46.27, "8": 46.87, "32": 44.02, "2048": 5.58},
     "art": "art:12777159bb149d6b80b8646c4800844c2e1ea11ea5c24622ae54ad406fb6e1b1"},
    {"weight_set": _L9, "d": 1, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 85.18, "8": 86.27, "32": 81.01, "2048": 10.31},
     "art": "art:2bffd5f6c2bf776c5f83f5d72ed7bbc8d3ff29b82ba69e9b33aff94f9f8c1d5f"},
    {"weight_set": _L9, "d": 0, "block_bytes": 147_456, "kind": "measured",
     "overhead_x_by_tokens": {"1": 47.55, "8": 47.95, "32": 45.28, "2048": 5.74},
     "art": "art:ab8b372394bc9c2aaa62bd24e27a975c81a48dfa9dbb84e28fd04a29e60a9404"},
    {"weight_set": _L9, "d": 0, "block_bytes": 393_216, "kind": "measured",
     "overhead_x_by_tokens": {"1": 39.3, "8": 61.58, "32": 58.04, "2048": 7.36},
     "art": "art:232d274d2fd99ba1ac640838894df3f25a658f299cc14fee812e773538d82b70",
     "note": "12,800 rows take 3 task rounds instead of 2; T=1 sits on the plain arm's warm-up reps (about 61 steady)"},
    {"weight_set": _LFULL, "d": 0, "block_bytes": 65_536, "kind": "measured",
     "overhead_x_by_tokens": {"1": 45.46, "8": 45.92, "32": 42.91, "2048": 5.45},
     "art": "art:24a67c520d1c44c4490e91387fe7071a76cf736c906584365cc3a509f65969bd"},
    {"weight_set": _LFULL, "d": 0, "block_bytes": 393_216, "kind": "measured",
     "overhead_x_by_tokens": {"1": 47.74, "8": 47.77, "32": 45.01, "2048": 5.7},
     "art": "art:3ba0d32eb400fccc8603bcd9e1168abb975ee7ed94ce85948d2b2d372b7bb1e6"},
)

#: the `measured` preset before the series carries histograms: series 03946311d run 1, 99,900 honest answers on loopback
#: under vLLM generation (verify-honest.json): its min, p50, p99 and p99.9 latency, then the answers over each window
RUN1 = {"art": "art:343f7b6fc43dcad199672efe1cbac7d2a727ed3da57ba56dbc63371ed8f62c54", "answers": 99_900,
        "quantiles_us": ((40.103, 1.0), (57.254, 0.5), (84.467, 1e-2), (132.262, 1e-3)),
        "late_over": ((150, 62), (200, 23), (250, 19), (300, 14), (350, 11), (400, 8), (450, 8), (500, 7), (551, 7), (750, 1))}
#: past this latency the measured tail continues as a Pareto of exponent `MEASURED_ALPHA`
MEASURED_KNEE_US, MEASURED_ALPHA = 750.0, 1.8

#: the soak's loads pooled into the ToR presets: the responder serving its own POUS forwards (idle has no serving)
SOAK_LOADS = ("decode", "prefill", "mixed")
#: the responder configs pooled: on loopback MSG_ZEROCOPY's pages are copied anyway, so `zerocopy` there adds only its
#: completion wait and says nothing about a NIC path
SOAK_CONFIGS = ("baseline", "tuned")
#: past the soak's slowest pooled answer a curve continues as a Pareto of this exponent
SOAK_TAIL_ALPHA = 1.0
#: the datacenter fabric beyond the ToR: lognormal (median, p99) µs, and a (chance, xmin µs, exponent) Pareto incast delay
FABRIC_RTT = (120.0, 400.0)
FABRIC_INCAST = (1e-3, 500.0, 1.5)

# The presets' fine grid (µs): fine enough for the 57 µs body, long enough for the curves' last point plus the longest shift.
H_US = 0.5
FINE = np.arange(int(1.002e5 / H_US) + 1) * H_US


def call_ns(block_bytes: int) -> int:
    """The fastest whole `Π₂` call on `block_bytes` labels (`verity.core.protocols.pous.primitives.pi2_call_ns`, in bits): the
    ideal-permutation model's unit, not the adversary's floor at the reference Feistel-SHAKE256 `Π₂`."""
    from verity.core.protocols.pous.primitives import pi2_call_ns

    return pi2_call_ns(8 * block_bytes)


def rounds(doc: dict[str, Any], d: int) -> int:
    """Rule 1's D: the cliff d, capped at the deepest D with an ε_proof, since the memory challenger refuses a deadline
    past it."""
    return min(d, max(e["D"] for e in doc["eps_proof"]))


def tau_us(d: int, call: float, m: float) -> float:
    """Rule 1: the threshold just under D + 1 calls at margin m, with D from `rounds`."""
    return (d + 1) * call / m / 1000 - 0.001


def survival_at(t_us: list[float], survival: list[float], t: float) -> float:
    """Rule 2: S(t) from a curve, ln S linear in ln t, 1 below the grid, the last slope above, floored at `S_FLOOR`."""
    if t < t_us[0]:
        return 1.0
    j = min(bisect.bisect_right(t_us, t) - 1, len(t_us) - 2)
    x0, x1 = math.log(t_us[j]), math.log(t_us[j + 1])
    y0, y1 = math.log(max(survival[j], S_FLOOR)), math.log(max(survival[j + 1], S_FLOOR))
    return max(S_FLOOR, math.exp(y0 + (y1 - y0) * (math.log(t) - x0) / (x1 - x0)))


# Poisson tails and quantiles, in log space.

def poisson_tail(f: int, mu: float) -> float:
    """P[X > f] for X ~ Poisson(mu), summed from the largest term outwards (no underflow of e^−μ at large μ)."""
    if f < 0:
        return 1.0
    if mu <= 0:
        return 0.0
    if f + 1 > mu:                                   # the terms fall from k = f + 1 on
        s = r = 1.0
        k = f + 1
        while r > 1e-17 * s:
            k += 1
            r *= mu / k
            s += r
        return math.exp(-mu + (f + 1) * math.log(mu) - math.lgamma(f + 2)) * s
    s = r = 1.0                                      # P[X ≤ f], its terms falling from k = f down
    k = f
    while k > 0 and r > 1e-17 * s:
        r *= k / mu
        k -= 1
        s += r
    return max(0.0, 1.0 - math.exp(-mu + f * math.log(mu) - math.lgamma(f + 1)) * s)


def bernstein_f0(mu: float, alpha: float) -> int:
    """Rule 3 past the table: the smallest f with P[X ≥ f + 1] ≤ α by Bernstein's inequality, an upper bound on the quantile."""
    lg = math.log(1 / alpha)
    return math.ceil(mu + lg / 3 + math.sqrt(lg * lg / 9 + 2 * mu * lg)) - 1


def poisson_quantile(mu: float, alpha: float) -> int:
    """The exact (1 − α) quantile of Poisson(mu): the smallest f with P[X > f] ≤ α, by bisection under Bernstein's bound."""
    if mu <= 0:
        return 0
    hi = bernstein_f0(mu, alpha)
    while poisson_tail(hi, mu) > alpha:              # Bernstein's bound holds; this guards rounding only
        hi = 2 * hi + 1
    lo = -1                                          # P[X > −1] = 1 > α
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if poisson_tail(mid, mu) <= alpha:
            hi = mid
        else:
            lo = mid
    return hi


def f0_max_mean(alpha: float, n: int = F0_TABLE_LEN) -> list[float]:
    """For f = 0 … n − 1, the largest mean whose (1 − α) Poisson quantile is ≤ f, to a relative 1e-12 and under it:
    P[X > f; μ] = α solved for μ by bisection (P[X > f; μ] rises with μ), all f at once.  On (0, f + 1) the tail's terms
    fall from k = f + 1 on, so a fixed number of them sums it to double precision."""
    f = np.arange(n, dtype=float)[:, None]
    k = f + 1 + np.arange(600)[None, :]
    log_factorial = np.array([math.lgamma(x + 1) for x in range(int(k.max()) + 1)])[k.astype(int)]
    lo, hi = np.zeros(n), np.arange(n) + 1.0
    lg_alpha = math.log(alpha)
    for _ in range(80):
        mu = (lo + hi) / 2
        logterms = -mu[:, None] + k * np.log(mu)[:, None] - log_factorial
        top = logterms.max(axis=1)
        log_tail = top + np.log(np.exp(logterms - top[:, None]).sum(axis=1))
        ok = log_tail <= lg_alpha
        lo, hi = np.where(ok, mu, lo), np.where(ok, hi, mu)
    return [float(x) for x in lo * (1 - 1e-12)]      # under the boundary by more than the tails' rounding


def f0_from_table(mu: float, alpha: float, table: list[float]) -> int:
    """Rule 3: the smallest f with table[f] ≥ μ, else Bernstein's."""
    f = bisect.bisect_left(table, mu)
    return f if f < len(table) else bernstein_f0(mu, alpha)


# The sampling term g(f).

def g_theta(theta: float, f: int, n: float, beta: float) -> float:
    return (theta * f + math.log(1 / beta)) / ((1 - math.exp(-theta)) * n)


def sampling(f: int, n: float, beta: float) -> float:
    """Rule 4: g at the closed-form θ (L / n at f = 0, the θ → ∞ limit)."""
    lg = math.log(1 / beta)
    if f == 0:
        return lg / n
    x = lg / f
    return g_theta(math.log(1 + x + math.sqrt(2 * x)), f, n, beta)


def g_opt(f: int, n: float, beta: float) -> float:
    """g(f) = min over θ > 0 of `g_theta`: the best point of a fine log grid, then golden-section search around it.  At
    f = 0 the infimum is the θ → ∞ limit, L / n."""
    if f == 0:
        return math.log(1 / beta) / n
    grid = np.geomspace(1e-6, 60.0, 4001)
    vals = [g_theta(t, f, n, beta) for t in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = (math.sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    while b - a > 1e-13 * b:
        if g_theta(c, f, n, beta) < g_theta(d, f, n, beta):
            b, d = d, c
            c = b - r * (b - a)
        else:
            a, c = c, d
            d = a + r * (b - a)
    return min(vals[i], g_theta((a + b) / 2, f, n, beta))


# Honest-latency presets: survival functions of the latency at the memory challenger (µs), at 64 KiB.

def lognormal_s(t: np.ndarray, median: float, p99: float) -> np.ndarray:
    sigma = math.log(p99 / median) / Z99
    z = (np.log(np.maximum(t, 1e-12)) - math.log(median)) / (sigma * math.sqrt(2))
    return 0.5 * np.frompyfunc(math.erfc, 1, 1)(z).astype(float)


def pareto_s(t: np.ndarray, xmin: float, alpha: float) -> np.ndarray:
    return np.where(t < xmin, 1.0, (np.maximum(t, xmin) / xmin) ** -alpha)


def loglog(points: list[tuple[float, float]], t: np.ndarray) -> np.ndarray:
    """S through `points` (t, S), ln S linear in ln t; 1 before the first point, 0 after the last (`with_tail` continues
    it)."""
    xs = np.log([p[0] for p in points])
    ys = np.log([max(p[1], S_FLOOR) for p in points])
    s = np.exp(np.interp(np.log(np.maximum(t, 1e-12)), xs, ys))
    return np.where(t < points[0][0], 1.0, np.where(t > points[-1][0], 0.0, s))


def with_tail(body: np.ndarray, t: np.ndarray, anchor: float) -> np.ndarray:
    """The measured curve continued past `MEASURED_KNEE_US` as a Pareto of exponent `MEASURED_ALPHA` from `anchor`, and
    never under `anchor` before it: the larger of the two, so the result still falls."""
    tail = np.where(t <= MEASURED_KNEE_US, anchor, anchor * (np.maximum(t, MEASURED_KNEE_US) / MEASURED_KNEE_US) ** -MEASURED_ALPHA)
    return np.maximum(body, tail)


def measured_run1(t: np.ndarray) -> np.ndarray:
    """`RUN1`: its quantiles and window counts, ln S linear in ln t, past 750 µs a Pareto from S(750 µs) = 1 / 99,900."""
    n = RUN1["answers"]
    points = [*RUN1["quantiles_us"], *((w, c / n) for w, c in RUN1["late_over"])]
    return with_tail(loglog(points, t), t, dict(RUN1["late_over"])[750] / n)


def pool_histograms(docs: list[dict[str, Any]]) -> tuple[list[float], list[float], int]:
    """The series' `verify-honest.json` elapsed-time histograms pooled: (edges µs, the share of answers at or above each
    edge, answers)."""
    edges = docs[0]["histograms"]["edges_us"]
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    over = total = 0
    for doc in docs:
        h = doc["histograms"]
        if h["edges_us"] != edges:
            raise ValueError("the histograms' edges differ")
        e = h["elapsed"]
        counts += np.array(e["counts"], dtype=np.int64)
        over += e["over"]
        total += sum(e["counts"]) + e["under"] + e["over"]
    at_or_above = np.append(np.cumsum(counts[::-1])[::-1], 0) + over
    return edges, [float(x) for x in at_or_above / total], total


def measured_histograms(docs: list[dict[str, Any]]):
    """S from the pooled histograms (`pool_histograms`), ln S linear in ln t between edges, and past 750 µs a Pareto from
    the share above 750 µs, never under one answer in the pool."""
    edges, s_edges, total = pool_histograms(docs)
    knee = min(range(len(edges)), key=lambda i: abs(edges[i] - MEASURED_KNEE_US))
    anchor = max(s_edges[knee], 1 / total)

    def s(t: np.ndarray) -> np.ndarray:
        return with_tail(loglog([(e, x) for e, x in zip(edges, s_edges) if x > 0] or [(edges[0], 1.0)], t), t, anchor)

    return s, total


def _fftconv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    nfft = 1 << (2 * n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]


def sum_s(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """S of X + Y on `FINE` from S_X and S_Y there (X, Y ≥ 0, independent), with X rounded up to the grid:
    S(t) = S_X(t) + Σ_{x ≤ t} P[X' = x] S_Y(t − x).  Clipped to [0, 1] and made non-increasing (the FFT's rounding)."""
    p = np.empty_like(sx)
    p[0] = 1 - sx[0]
    p[1:] = sx[:-1] - sx[1:]
    s = np.clip(sx + _fftconv(p, sy), 0.0, 1.0)
    return np.minimum.accumulate(s)


def rtt_s(median: float, p99: float, tail_mass: float, xmin: float, alpha: float) -> np.ndarray:
    """A network round trip: lognormal, plus with probability `tail_mass` a Pareto(xmin, alpha) delay."""
    ln = lognormal_s(FINE, median, p99)
    return (1 - tail_mass) * ln + tail_mass * sum_s(ln, pareto_s(FINE, xmin, alpha))


LABELS = {
    "hardened": ("Same host, isolated cores and IRQs", "assumed",
                 "The measured body (lognormal, median 57 µs, p99 84 µs) plus 1% of the measured stall excess."),
    "measured": ("Same host, under vLLM load", "empirical",
                 ("Measured on loopback under vLLM generation (RTX PRO 6000, band d = 12 responder); past 750 µs continued "
                  "as a Pareto of exponent 1.8.")),
    "rack": ("Memory challenger on another host in the rack", "assumed",
             ("The measured server side (12 µs of loopback removed) plus a round trip lognormal (median 25 µs, p99 60 µs), "
              "with a 1e-4 chance of a Pareto(200 µs, 2) delay.")),
    "datacenter": ("Memory challenger a few hops away", "assumed",
                   ("The measured server side plus a round trip lognormal (median 120 µs, p99 400 µs), with a 1e-3 chance of "
                    "a Pareto(500 µs, 1.5) incast delay.")),
    "heavy-tail": ("Shared, noisy host", "assumed",
                   "The slower of measured and the body plus a 0.3% chance of an untruncated Pareto(100 µs, 1.2) stall."),
}


def presets(measured) -> dict[str, np.ndarray]:
    """Every preset's S on `FINE` at 64 KiB, from `measured` (a survival function of µs)."""
    sm = measured(FINE)
    body = lognormal_s(FINE, 57.0, 84.0)
    server = measured(FINE + 12.0)
    return {
        "hardened": body + 0.01 * np.clip(sm - body, 0.0, None),
        "measured": sm,
        "rack": sum_s(server, rtt_s(25.0, 60.0, 1e-4, 200.0, 2.0)),
        "datacenter": sum_s(server, rtt_s(*FABRIC_RTT, *FABRIC_INCAST)),
        "heavy-tail": np.maximum(sm, (1 - 3e-3) * body + 3e-3 * sum_s(body, pareto_s(FINE, 100.0, 1.2))),
    }


def curve(s_fine: np.ndarray, block_bytes: int) -> list[float]:
    """S at `T_US` for a block of `block_bytes`: the 64 KiB curve later by `US_PER_KIB` per KiB above 64, ln S linear in t
    on `FINE`, floored at `S_FLOOR` and rounded to 6 significant digits (rounding keeps it non-increasing)."""
    shift = US_PER_KIB * (block_bytes - 65_536) / 1024
    t = np.array(T_US) - shift
    s = np.where(t <= 0, 1.0, np.exp(np.interp(t, FINE, np.log(np.maximum(s_fine, S_FLOOR)))))
    return [float(f"{max(x, S_FLOOR):.6g}") for x in s]


# The ToR presets from a `pous-soak/v1` file.

def soak_pooled(soak: dict[str, Any], metric: str, block_bytes: int,
                by_answers: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """S of `metric` at `T_US` for one block: the soak's challenge answers with kernel stamps in the `SOAK_LOADS` and
    `SOAK_CONFIGS` cells of that block pooled, each audit rate weighted equally (the rate isn't fixed, and most answers are
    the fastest rate's) and the cells of one rate by their answers; `by_answers` weights every cell by its answers instead.
    Past the slowest pooled answer, a Pareto of exponent `SOAK_TAIL_ALPHA` from the last grid point with S > 0.  The soak's grid holds `T_US`, so S
    at a grid point is the cells' own value there (0 past a cell's `max_us`)."""
    cells = {c["cell"]: c for c in soak["cells"]}
    rows = [s for s in soak["survival"] if s["metric"] == metric and s["op"] == "challenge" and s["timestamping"]
            and cells[s["cell"]]["block_bytes"] == block_bytes and cells[s["cell"]]["load"] in SOAK_LOADS
            and cells[s["cell"]]["config"] in SOAK_CONFIGS]
    if not rows:
        raise ValueError(f"no {metric} challenge rows at {block_bytes} B in the soak file")
    grid = np.array(soak["t_us"])
    at = np.array([int(np.argmin(np.abs(grid - t))) for t in T_US])
    if not np.allclose(grid[at], T_US, rtol=1e-9):
        raise ValueError("the soak's grid does not hold the explorer's")
    n = np.array([r["n"] for r in rows], dtype=float)
    rate = np.array([cells[r["cell"]]["rate_hz"] for r in rows])
    rates = sorted(set(rate.tolist()))
    w = n if by_answers else n / np.array([n[rate == x].sum() for x in rate]) / len(rates)
    s = (w[:, None] * np.array([r["s"] for r in rows])[:, at]).sum(axis=0) / w.sum()
    last = int(np.flatnonzero(s > 0)[-1])
    t = np.array(T_US)
    s = np.where(np.arange(len(t)) > last, s[last] * (t / t[last]) ** -SOAK_TAIL_ALPHA, s)
    configs = sorted({cells[r["cell"]]["config"] for r in rows})
    return np.minimum.accumulate(s), {"cells": len(rows), "answers": int(n.sum()), "max_us": max(r["max_us"] for r in rows),
                                      "tail_from_us": float(t[last]), "rates_hz": rates, "configs": configs}


def soak_fine(s_grid: np.ndarray) -> np.ndarray:
    """A `T_US` curve on `FINE`: ln S linear in ln t between grid points, 1 below the grid."""
    s = np.exp(np.interp(np.log(np.maximum(FINE, 1e-12)), np.log(T_US), np.log(np.maximum(s_grid, S_FLOOR))))
    return np.where(FINE < T_US[0], 1.0, s)


def tor_terms(params: dict[str, Any]) -> str:
    """The modeled NIC and wire terms of a soak ToR metric, as the soak file states them."""
    return (f"NIC RX {params['t_nic_rx_us']:g} µs (base {params['nic_rx_base_us']:g} µs, rx-usecs {params['rx_usecs']:g}), "
            f"TX {params['tx_fixed_us']:g} µs fixed plus the larger of bytes / {params['b_pcie_bytes_per_s'] / 1e9:g} GB/s "
            f"PCIe and wire bytes · 8 / {params['link_bps'] / 1e9:g} Gb/s, wire bytes = header {params['header_bytes']} B + "
            f"block + {params['frame_overhead_bytes']} B per {params['mss_bytes']:,} B MSS frame (MTU {params['mtu']}), a "
            f"{params['request_frame_bytes']} B request frame serialized, and 2 × {params['cable_m']:g} m of cable at "
            f"{params['prop_ns_per_m']:g} ns/m ({params['nic']}, {params['setting']} NIC)")


SOAK_PRESETS = {
    "rack": ("tor_400_tuned", "Memory challenger on another host in the rack, 400 Gb/s"),
    "rack-100g": ("tor_100_default", "Memory challenger on another host in the rack, 100 Gb/s, default NIC"),
    "datacenter": ("tor_100_default", "Memory challenger a few hops away"),
}


def soak_presets(soak: dict[str, Any], source: str, note: str | None = None,
                 raw: tuple[dict[str, Any], str] | None = None) -> list[dict[str, Any]]:
    """The ToR presets (`SOAK_PRESETS`): per block, `soak_pooled` of the metric; `datacenter` adds the fabric round trip
    (`FABRIC_RTT`, `FABRIC_INCAST`) on `FINE`.  Each description states every modeled parameter, and S at d = 0, 64 KiB
    under both weightings, then `note`; with `raw` (the file before its exclusions, and its source), S at d = 0 per block
    from both files."""
    metrics = {m["id"]: m for m in soak["metrics"]}
    tau0 = tau_us(0, call_ns(65_536), 1)

    def pooled(mid: str, metric: str, b: int, by_answers: bool = False,
               doc: dict[str, Any] = soak) -> tuple[dict[str, Any], dict[str, Any]]:
        s, info = soak_pooled(doc, metric, b, by_answers)
        if mid == "datacenter":
            s = np.array(curve(sum_s(soak_fine(s), rtt_s(*FABRIC_RTT, *FABRIC_INCAST)), 65_536))
        return {"block_bytes": b, "t_us": [round(t, 9) for t in T_US],
                "survival_fraction": [float(f"{max(x, S_FLOOR):.6g}") for x in s]}, info

    def d0(cs: list[dict[str, Any]]) -> str:
        return ", ".join(f"{survival_at(c['t_us'], c['survival_fraction'], tau_us(0, call_ns(c['block_bytes']), 1)):.3g}"
                         for c in cs)

    out = []
    for mid, (metric, label) in SOAK_PRESETS.items():
        curves, infos = zip(*(pooled(mid, metric, b) for b in BLOCK_BYTES))
        ref = {w: survival_at(c["t_us"], c["survival_fraction"], tau0)
               for w, c in (("equal", curves[0]), ("answers", pooled(mid, metric, 65_536, by_answers=True)[0]))}
        i = infos[0]
        rates = "/".join(str(r) for r in i["rates_hz"])
        answers = ", ".join(f"{x['answers']:,}" for x in infos)
        tails = ", ".join(f"{x['tail_from_us'] / 1000:.1f}" for x in infos)
        desc = (f"The soak's {metric} ({metrics[metric]['label']}): R = K_tx_resp − K_rx_req measured on loopback on "
                f"{soak['run']['host']} ({soak['run']['gpu']}, band d = 12 responder) while vLLM ran POUS forwards "
                f"({', '.join(SOAK_LOADS)}), pooled over {i['cells']} cells per block with each rate ({rates} Hz) "
                f"weighted equally and a rate's cells by their answers (configs {' and '.join(i['configs'])}; {answers} "
                f"answers at 64/144/384 KiB), plus the modeled {tor_terms(metrics[metric]['params'])}.  Loopback's send "
                f"includes the receiver's processing, so R is pessimistic at long blocks.  Past the slowest pooled answer "
                f"({tails} ms) a Pareto of exponent {SOAK_TAIL_ALPHA:g}.  At d = 0, 64 KiB (τ = {tau0:.3f} µs) S is "
                f"{ref['equal']:.3g}; weighting every cell by its answers instead, it would be {ref['answers']:.3g}.")
        if mid == "datacenter":
            desc += (f"  Plus a fabric round trip beyond the ToR: lognormal (median {FABRIC_RTT[0]:g} µs, p99 "
                     f"{FABRIC_RTT[1]:g} µs), with a {FABRIC_INCAST[0]:g} chance of a Pareto({FABRIC_INCAST[1]:g} µs, "
                     f"{FABRIC_INCAST[2]:g}) incast delay.")
        if note:
            desc += f"  {note}"
        if raw:
            full = [pooled(mid, metric, b, doc=raw[0])[0] for b in BLOCK_BYTES]
            desc += (f"  S at d = 0 (64, 144, 384 KiB) is {d0(list(curves))} here; with the runs this file leaves out "
                     f"({raw[1]}) it would be {d0(full)}.")
        out.append({"id": mid, "label": label, "kind": "empirical" if mid != "datacenter" else "assumed",
                    "description": desc, "sources": [source], "curves": list(curves)})
    return out


# The file and its evaluation.

def evaluate(doc: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Rules 1–5 on the file `doc` alone, as the page evaluates them."""
    call = {b["block_bytes"]: b["call_ns"] for b in doc["block_sizes"]}[inputs["block_bytes"]]
    D = rounds(doc, inputs["d"])
    tau = tau_us(D, call, inputs["m"])
    model = next(m for m in doc["honest_models"] if m["id"] == inputs["model"])
    c = next(c for c in model["curves"] if c["block_bytes"] == inputs["block_bytes"])
    p = survival_at(c["t_us"], c["survival_fraction"], tau)
    n = inputs["lambda_per_s"] * inputs["T_s"]
    f0 = f0_from_table(n * p, inputs["alpha"], doc["f0_max_mean"][str(inputs["alpha"])])
    samp = sampling(f0, n, inputs["beta"])
    eps = {e["D"]: e["eps_proof_fraction"] for e in doc["eps_proof"]}[D]
    return {"rounds_D": D, "tau_us": tau, "p_fraction": p, "f0_count": f0, "sampling_fraction": samp,
            "bound_fraction": eps + samp}


def vector_inputs(models: list[str]) -> list[dict[str, Any]]:
    """About 50 settings: every model at every scheme with a bound, every α and β, m = 1.5, 2, 3, every D, f₀ = 0, table
    hits and Bernstein's branch."""
    base = {"m": 1, "lambda_per_s": 100, "T_s": 600, "alpha": 0.01, "beta": 0.01}
    schemes = [(0, 65_536), (1, 65_536), (2, 65_536), (0, 147_456), (0, 393_216)]
    out = [{"model": mo, "d": d, "block_bytes": b, **base} for mo in models for d, b in schemes]
    for mo, m in zip(models, (2, 2, 1.5, 3, 2, 1.5)):
        out.append({**out[0], "model": mo, "d": 1, "m": m})
    for a in ALPHAS[1:]:
        for mo in ("measured", "datacenter"):
            out.append({**out[0], "model": mo, "alpha": a})
    for bt in BETAS[1:]:
        out.append({**out[0], "model": "rack", "d": 1, "beta": bt})
        out.append({**out[0], "model": "heavy-tail", "d": 2, "beta": bt})
    for d in (3, 4, 5, 12):
        out.append({**out[0], "model": "measured", "d": d})
    out.append({**out[0], "model": "datacenter", "d": 12, "block_bytes": 393_216})
    out += [{**out[0], "model": "hardened", "T_s": 1},                                  # f₀ = 0
            {**out[0], "model": "measured", "lambda_per_s": 10, "T_s": 60},
            {**out[0], "model": "measured", "lambda_per_s": 10_000, "T_s": 86_400},       # Bernstein
            {**out[0], "model": "datacenter", "lambda_per_s": 1000, "T_s": 3600, "alpha": 1e-6, "beta": 1e-6},
            {**out[0], "model": "rack", "lambda_per_s": 10, "T_s": 60, "m": 3, "block_bytes": 393_216}]
    return out


def source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(histograms: list[dict[str, Any]] | None = None, generated_at: str | None = None,
          commit: str | None = None, soak: tuple[dict[str, Any], str] | None = None,
          soak_note: str | None = None, soak_raw: tuple[dict[str, Any], str] | None = None) -> dict[str, Any]:
    """The `pous-explorer/v1` document; `soak` (a `pous-soak/v1` document and its source) replaces the off-host presets
    with `soak_presets`, their descriptions ending with `soak_note` and, with `soak_raw`, the S at d = 0 the soak file's
    excluded runs would give."""
    if histograms and len(histograms) >= 2:
        measured, answers = measured_histograms(histograms)
        measured_sources = [d.get("source", "series verify-honest.json") for d in histograms]
        measured_note = f"pooled histograms of {len(histograms)} series runs, {answers:,} answers"
    else:
        measured, measured_sources, measured_note = measured_run1, [RUN1["art"]], "series 03946311d run 1, 99,900 answers"
    fine = presets(measured)
    models = []
    for mid, s in fine.items():
        label, kind, desc = LABELS[mid]
        sources = measured_sources if mid != "hardened" else [*measured_sources, "lognormal body"]
        if mid == "measured":
            desc = f"{desc} Data: {measured_note}."
        models.append({"id": mid, "label": label, "kind": kind, "description": desc, "sources": sources,
                       "curves": [{"block_bytes": b, "t_us": [round(t, 9) for t in T_US], "survival_fraction": curve(s, b)}
                                  for b in BLOCK_BYTES]})
    if soak:
        tor = {m["id"]: m for m in soak_presets(*soak, soak_note, soak_raw)}
        models = [x for m in models for x in ([tor["rack"], tor["rack-100g"]] if m["id"] == "rack" else [tor.get(m["id"], m)])]
    doc = {"schema": SCHEMA,
           "generated_at_utc": generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source_commit": commit or source_commit(),
           "block_sizes": [{"block_bytes": b, "call_ns": call_ns(b)} for b in BLOCK_BYTES],
           "us_per_kib_above_64": US_PER_KIB,
           "eps_proof": list(EPS_PROOF),
           "f0_max_mean": {str(a): f0_max_mean(a) for a in ALPHAS},
           "weight_sets": list(WEIGHT_SETS),
           "schemes": list(SCHEMES),
           "honest_models": models}
    doc["vectors"] = [{"inputs": i, "expected": evaluate(doc, i), "tolerance": {"relative": 1e-9, "f0_count": 0}}
                      for i in vector_inputs([m["id"] for m in models])]
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the pous-explorer/v1 JSON to write")
    ap.add_argument("--histograms", nargs="*", default=[], metavar="PATH=SOURCE",
                    help="series verify-honest.json files to pool into `measured`, each with its source (an art id)")
    ap.add_argument("--soak", metavar="PATH=SOURCE", help="a pous-soak/v1 file (and its art id) for the ToR presets")
    ap.add_argument("--soak-note", help="a sentence ending each ToR preset's description: what the soak file's run was")
    ap.add_argument("--soak-raw", metavar="PATH=SOURCE", help="the soak file before its exclusions (and its art id): each ToR "
                    "preset's description gives S at d = 0 from both")
    a = ap.parse_args(argv)
    docs = []
    for arg in a.histograms:
        path, _, source = arg.partition("=")
        d = json.loads(Path(path).read_text())
        d["source"] = source or path
        docs.append(d)
    soak = None
    if a.soak:
        path, _, source = a.soak.partition("=")
        soak = (json.loads(Path(path).read_text()), source or path)
    raw = None
    if a.soak_raw:
        path, _, source = a.soak_raw.partition("=")
        raw = (json.loads(Path(path).read_text()), source or path)
    doc = build(docs, soak=soak, soak_note=a.soak_note, soak_raw=raw)
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    Path(a.out).write_text(text + "\n")
    print(f"{a.out}: {len(text):,} bytes, {len(doc['vectors'])} vectors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
