"""The host CPUs' modular-squaring rate at a Sloth prime: what a node's 2 × 64 cores add to the adversary's Φ.

Daniel's node-level ruling puts the host inside the device, so the host's squaring throughput adds to the
adversary's encode rate Φ, and γ's rebuild term d/(k·x₁,adv) grows with it. One Sloth root raises to the b-bit
exponent (p+1)/4, about b squarings mod p; the encode at k is k roots a word. This measures squarings/s per core and across a pinned set of
cores (aggregate, so cache and memory contention count), using `gmpy2.powmod(base, 1 << S, p)`, whose exponent is
S squarings and no multiplies, all inside GMP.

`uv run --no-project --with gmpy2 --with python-flint python host_sq.py [CORELIST [SQUARINGS]]` (CORELIST like `48-95` or `48,50,52`)
prints one JSON line per field: per-core squarings/s, the aggregate over the pinned cores, and the aggregate
scaled to 128 cores, with one root's time at k = 2 for the encode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import gmpy2
from gmpy2 import mpz

from partial_root import FIELDS


def parse_cores(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _worker(args: tuple[int, int, int, int]) -> float:
    core, n, cc, squarings = args
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        pass
    p = mpz((1 << n) - cc)
    e = mpz(1) << squarings  # a pure squaring chain: S squarings, no multiplies
    base = mpz(3)
    reps = max(1, 2_000_000 // squarings)
    t0 = time.perf_counter()
    for _ in range(reps):
        base = gmpy2.powmod(base, e, p)
    dt = time.perf_counter() - t0
    return reps * squarings / dt


def main(cores: list[int], squarings: int) -> None:
    for name, (n, cc) in FIELDS.items():
        with Pool(len(cores)) as pool:
            rates = pool.map(_worker, [(c, n, cc, squarings) for c in cores])
        per_core = sum(rates) / len(rates)
        agg = sum(rates)
        # (p+1)/4 is a b-bit exponent: a root is ≈ b squarings (a sliding window adds ≈ b/5 multiplies, left out so
        # the root count favors the adversary). The k = 2 encode is 2 roots a word.
        root_sq = n
        print(json.dumps({"field": name, "bits": n, "cores": len(cores), "squarings_per_call": squarings,
                          "per_core_sq_per_s": round(per_core, 1), "aggregate_sq_per_s": round(agg, 1),
                          "scaled_128_sq_per_s": round(agg / len(cores) * 128, 1),
                          "root_us_1core": round(root_sq / per_core * 1e6, 2),
                          "k2_word_us_128core": round(2 * root_sq / (agg / len(cores) * 128) * 1e6, 4)}), flush=True)


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    main(parse_cores(spec), int(sys.argv[2]) if len(sys.argv) > 2 else 4096)
