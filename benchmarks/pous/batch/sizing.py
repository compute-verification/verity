"""How large a batched PoUS round must be off-region, and the most it can certify.

A batch reveals b indices together and reads all b blocks back under one deadline:

    T(b) = RTT_max + b·(s + i) / X + margin

with s = 2,056 bytes a block (`p2-16448/v4`), i = 8 bytes an index (`challenger_party`'s u64be frame; 0 if the challenger
sent a seed the indices derive from), and X the rate the timed bytes cross at: the link's goodput BW when the blocks
themselves are timed (`memory_challenger`'s batched sessions), or the prover's hash rate H when only a commitment to them
is (the keyed row: a SHA-512 of the b blocks under a fresh nonce from the challenger, the blocks sent after the receipt
time, the indices as a seed). The honest prover reads from HBM in microseconds, and the challenger decodes the answers
after the receipt time, so neither is in T.

A prover missing a fraction f of `C*` passes a batch only if it recomputes, before T, every distinct missing block the
batch asks for. The premise that replaces P2's SeqRoot (no root inside one round's deadline) is a throughput bound:
inside T, everything the prover can reach encodes at most R blocks a second, so it recomputes at most cap(b) = ⌊R·T(b)⌋
blocks, counting from the moment the batch is sent. Were the b draws distinct, the missing ones would be hypergeometric,
whose lower tail Hoeffding bounds by the binomial's Chernoff bound:

    P[≤ cap of b missing] ≤ exp(−b·KL(cap/b ‖ f))     (when cap < b·f; else 1)

The challenger draws with replacement (`CoinTree.draw`), so a batch with r repeats asks for b − r distinct blocks. Draw j
repeats with probability at most (j − 1)/N given the draws before it, so r is stochastically below a sum of independent
Bernoulli((j − 1)/N), mean μ = b(b − 1)/2N, and P[r ≥ m] ≤ exp(m − μ − m·ln(m/μ)). The bound used is the least over m ≥ 1
of P[r ≥ m] + exp(−d·KL(cap/d ‖ f)) at d = b − m + 1, the exponent rising with d. At the demo's N = 68,091,904 blocks, b =
524,288 has μ = 2,018. n batches pass with at most the n-th power. As b grows, cap/b falls to R·(s + i)/X, so no batch
catches a prover missing less than f_inf = R·(s + i)/X, and what the round certifies is at most (1 − f_inf)·|C*|.

    sizing.py --rate R [--rtt-ms 100.18] [--margin-ms 10] [--hash-gbps H] [--f 0.1,...] [--delta 0.01] [--out F]

prints, for each timed rate X and each f, f_inf, the smallest power-of-two batch one round of which reaches δ, and its
T and cap, and writes the same as JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BLOCK_BYTES = 2_056
"""A `p2-16448/v4` block: 16,448 bits (`verity/core/protocols/pous/schemes/p2.py`, `POINT`)."""
STORE_BYTES = 139_996_954_624
"""The demo's `C*`: 68,091,904 blocks of 2,056 bytes (note:20261025T2337Z-report-p2-encode)."""
STORE_BLOCKS = STORE_BYTES // BLOCK_BYTES
INDEX_BYTES = 8
"""An index on `challenger_party`'s wire: a u64be."""
U_GIB, OBLIGATION = 139.64, 0.93
"""ASK-132: at least 0.93·U of P2-encoded weights, U = 139.64 GiB."""
MIB = 1 << 20


def deadline_s(b: int, rtt_s: float, x: float, margin_s: float, s: int = BLOCK_BYTES, i: int = INDEX_BYTES) -> float:
    """T(b): the round trip, the b indices' and blocks' timed bytes at `x` bytes a second, and the margin."""
    return rtt_s + b * (s + i) / x + margin_s


def capacity(rate: float, t_s: float) -> int:
    """The blocks a prover recomputes inside `t_s` at `rate` blocks a second."""
    return math.floor(rate * t_s)


def kl(a: float, f: float) -> float:
    """KL(a ‖ f) between Bernoulli(a) and Bernoulli(f), in nats."""
    out = (1 - a) * math.log((1 - a) / (1 - f)) if a < 1 else 0.0
    return out + (a * math.log(a / f) if a > 0 else 0.0)


def log_pass(b: int, f: float, cap: int) -> float:
    """ln of the Chernoff bound on P[Bin(b, f) ≤ cap]: 0 when cap ≥ b·f."""
    return 0.0 if cap >= b * f else -b * kl(cap / b, f)


def log_repeats_at_least(b: int, n: int, m: int) -> float:
    """ln of a bound on P[b draws with replacement from n repeat at least m times]: 0 when m ≤ μ = b(b − 1)/2n."""
    mu = b * (b - 1) / (2 * n)
    if m <= mu:
        return 0.0
    return -math.inf if mu == 0 else m - mu - m * math.log(m / mu)


def log_pass_drawn(b: int, f: float, cap: int, n: int | None) -> float:
    """ln of a bound on P[b draws with replacement from n blocks ask for at most cap distinct missing ones]: the least
    over m ≥ 1 of P[r ≥ m] + the Chernoff bound at the b − m + 1 distinct draws fewer repeats leave; `log_pass` when n
    is None (draws distinct)."""
    if n is None:
        return log_pass(b, f, cap)
    mu = b * (b - 1) / (2 * n)
    ms = {1 << j for j in range(b.bit_length())} | {math.ceil(mu + k * math.sqrt(mu + 1) / 4) for k in range(200)}
    return min([0.0] + [_log_add(log_repeats_at_least(b, n, m), log_pass(b - m + 1, f, cap))
                        for m in ms if 1 <= m <= b])


def _log_add(a: float, c: float) -> float:
    hi, lo = max(a, c), min(a, c)
    return hi if lo == -math.inf else hi + math.log1p(math.exp(lo - hi))


def binomial_cdf(b: int, f: float, cap: int) -> float:
    """P[Bin(b, f) ≤ cap] exactly, for the tests' small b."""
    return sum(math.comb(b, i) * f ** i * (1 - f) ** (b - i) for i in range(min(cap, b) + 1))


def distinct_pmf(b: int, n: int) -> list[float]:
    """P[b draws with replacement from n hit exactly d distinct], d = 0..b, exactly, for the tests' small b."""
    p = [1.0] + [0.0] * b
    for _ in range(b):
        p = [p[d] * d / n + (p[d - 1] * (n - d + 1) / n if d else 0.0) for d in range(b + 1)]
    return p


def f_inf(rate: float, x: float, s: int = BLOCK_BYTES, i: int = INDEX_BYTES) -> float:
    """The missing fraction no batch catches below: R·(s + i)/X."""
    return rate * (s + i) / x


def f_obligation(store_bytes: int = STORE_BYTES, u_gib: float = U_GIB, obligation: float = OBLIGATION) -> float:
    """The missing fraction of `C*` at which the prover holds exactly obligation·U."""
    return 1 - obligation * u_gib * 2 ** 30 / store_bytes


def rounds(lp: float, delta: float) -> float:
    """Batches of one size until the bound reaches δ: inf when one passes for sure."""
    return math.inf if lp >= 0 else max(1, math.ceil(math.log(delta) / lp))


def plan(f: float, rate: float, rtt_s: float, x: float, margin_s: float, delta: float, *, s: int = BLOCK_BYTES,
         i: int = INDEX_BYTES, n: int | None = STORE_BLOCKS, b_max: int = 1 << 26) -> dict:
    """The smallest power-of-two batch one round of which, drawn with replacement from `n` blocks, takes a prover missing
    `f` below δ, or None past `b_max`."""
    b = 1
    while b <= b_max:
        t = deadline_s(b, rtt_s, x, margin_s, s, i)
        cap = capacity(rate, t)
        lp = log_pass_drawn(b, f, cap, n)
        if rounds(lp, delta) == 1:
            return {"b": b, "T_ms": round(t * 1e3, 3), "cap": cap, "missing_mean": round(b * f, 1),
                    "log10_pass": round(lp / math.log(10), 2), "timed_mb": round(b * (s + i) / 1e6, 2)}
        b *= 2
    return {"b": None}


def table(rate: float, rtt_s: float, margin_s: float, xs: dict[str, float], fs: dict[str, float], delta: float,
          store_bytes: int = STORE_BYTES, s: int = BLOCK_BYTES, seeded: frozenset[str] = frozenset()) -> dict:
    """Per timed rate, f_inf, what it certifies at most, and each f's plan; a rate in `seeded` sends no indices."""
    out = {}
    for name, x in xs.items():
        i = 0 if name in seeded else INDEX_BYTES
        fi = f_inf(rate, x, s, i)
        out[name] = {"x_bytes_per_s": x, "index_bytes": i, "f_inf": fi,
                     "certifiable_gib": max(0.0, (1 - fi) * store_bytes / 2 ** 30),
                     "plans": {k: plan(f, rate, rtt_s, x, margin_s, delta, s=s, i=i, n=store_bytes // s)
                               for k, f in fs.items()}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rate", type=float, required=True, help="R: blocks a second everything the prover reaches encodes")
    ap.add_argument("--rtt-ms", type=float, default=100.18, help="RTT_max (r20261025-233837-b76c: max of 20 connects)")
    ap.add_argument("--rtt-p50-ms", type=float, default=97.39, help="the round trip a window-limited link divides by")
    ap.add_argument("--margin-ms", type=float, default=10.0)
    ap.add_argument("--delta", type=float, default=0.01, help="the bound one round must reach (P2's δ = 1%%)")
    ap.add_argument("--f", default="", help="extra missing fractions, comma-separated (always: 0.10, 1/19, obligation)")
    ap.add_argument("--hash-gbps", type=float, default=0.0, help="H, GB/s: add the keyed commitment's row")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rtt, p50, margin = a.rtt_ms / 1e3, a.rtt_p50_ms / 1e3, a.margin_ms / 1e3
    xs = {"ssh channel (2 MiB window / RTT)": 2 * MIB / p50, "tcp stream (6 MiB rmem / RTT)": 6 * MIB / p50,
          "1 Gbit/s": 125e6, "10 Gbit/s": 1.25e9}
    keyed = f"keyed commitment (H = {a.hash_gbps:g} GB/s)"
    if a.hash_gbps:
        xs[keyed] = a.hash_gbps * 1e9
    fs = {"0.10": 0.10, "1/19": 1 / 19, "obligation": f_obligation()}
    fs |= {x: float(x) for x in a.f.split(",") if x}
    doc = {"inputs": {"rate_blocks_per_s": a.rate, "rtt_max_ms": a.rtt_ms, "rtt_p50_ms": a.rtt_p50_ms,
                      "margin_ms": a.margin_ms, "delta": a.delta, "block_bytes": BLOCK_BYTES, "index_bytes": INDEX_BYTES,
                      "store_bytes": STORE_BYTES, "u_gib": U_GIB, "obligation": OBLIGATION, "f": fs},
           "rows": table(a.rate, rtt, margin, xs, fs, a.delta, seeded=frozenset({keyed}))}
    for name, row in doc["rows"].items():
        print(f"{name}: X = {row['x_bytes_per_s'] / 1e6:.1f} MB/s, f_inf = {row['f_inf']:.4%}, "
              f"certifiable <= {row['certifiable_gib']:.2f} GiB")
        for k, p in row["plans"].items():
            print(f"  f = {k} ({fs[k]:.4%}): " + (f"b = {p['b']}, T = {p['T_ms']} ms, cap = {p['cap']}, "
                                                    f"E[missing] = {p['missing_mean']}, log10 P <= {p['log10_pass']}"
                                                    if p["b"] else "no batch reaches delta"))
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
