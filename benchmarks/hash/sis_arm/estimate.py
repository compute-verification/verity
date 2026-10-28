"""The lattice estimator on the Module-SIS commitment arm: binding of h = A x mod q and hiding of A x + B r.

Runs under Sage. passagemath's wheels are enough, so no Sage install is needed:

    uv venv --python 3.12 venv && . venv/bin/activate && uv pip install --only-binary :all: passagemath-standard
    git clone https://github.com/malb/lattice-estimator && cd lattice-estimator   # this sweep ran at ccb2c74
    python /path/to/estimate.py OUT.json [--jobs 4]

Binding: two openings x != x' of one digest give A (x - x') = 0 mod q with every |x_i - x'_i| <= beta, so SIS in the
l-infinity norm with length_bound beta: 255 for int8 words, 15 for nibbles, 1 for bits. `n` is the digest's rows (a
module of rank k over a degree-d ring has n = k d; the estimator, like every published module-lattice estimate, treats
the matrix as unstructured). `m` is the row's words; the attack picks its own sub-dimension, so m matters only up to it.

Hiding by LWE: with B = [B' | I_n] and r = (s, e), B r = B' s + e is n LWE samples in dimension k with secret and error
uniform int8, so A x + B r is pseudorandom when LWE(k, q, int8, n samples) is hard; r has k + n words. The leftover hash
lemma needs no estimate: it needs 8 (k + n) >= n log2 q + 2 lambda (the note derives it for q = 2^32).

Each row reports `SIS.estimate` (the estimator's default cost and shape models) and `SIS.estimate.rough` (ADPS16
Core-SVP, 0.292 beta), log2 of the attack's ring operations, the block size and the attack dimension.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from math import log2

Q = 2 ** 32
INT8_N = (96, 112, 128, 144, 160, 176, 192, 224, 256, 320)
INT8_M = (1024, 4096, 16384, 65536)
DIGITS = (("nibble", 15, 2, (48, 64, 80, 96, 112, 128)), ("bit", 1, 8, (16, 24, 32, 40, 48, 64)))
LWE_K = (32, 64, 96, 128, 192, 256, 384, 512)
LWE_SAMPLES = (64, 128, 256)


def _bits(cost) -> float:
    rop = float(cost["rop"])
    return round(log2(rop), 1) if rop not in (float("inf"),) and rop > 0 else float("inf")


def _row(cost) -> dict:
    return {"bits": _bits(cost), "beta": int(cost.get("beta", 0)), "d": int(cost.get("d", 0))}


def sis(job: tuple[str, int, int, int]) -> dict:
    from sage.all import oo
    from estimator import SIS
    enc, beta, n, m = job
    p = SIS.Parameters(n=n, q=Q, length_bound=beta, m=m, norm=oo, tag=f"{enc}-q32-n{n}-m{m}")
    full = SIS.estimate(p, jobs=1, quiet=True)
    rough = SIS.estimate.rough(p, quiet=True)
    return {"problem": "sis", "encoding": enc, "beta": beta, "q": Q, "n": n, "m": m,
            "default": {k: _row(v) for k, v in full.items()}, "rough": {k: _row(v) for k, v in rough.items()}}


def lwe(job: tuple[str, int, int]) -> dict:
    from estimator import LWE, ND
    dist, k, samples = job
    x = ND.Uniform(-128, 127) if dist == "int8" else ND.Uniform(-1, 1)
    p = LWE.Parameters(n=k, q=Q, Xs=x, Xe=x, m=samples, tag=f"hide-{dist}-k{k}-n{samples}")
    rough = LWE.estimate.rough(p, quiet=True)
    return {"problem": "lwe", "dist": dist, "k": k, "samples": samples, "q": Q,
            "rough": {a: {"bits": _bits(c), "beta": int(c.get("beta", 0))} for a, c in rough.items()}}


def main() -> None:
    out = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    work = [("int8", 255, n, m) for n in INT8_N for m in INT8_M]
    work += [(enc, beta, n, 16384 * per_byte) for enc, beta, per_byte, ns in DIGITS for n in ns]
    hide = [(d, k, s) for d in ("int8", "ternary") for k in LWE_K for s in LWE_SAMPLES]
    with ProcessPoolExecutor(jobs) as pool:
        rows = list(pool.map(sis, work)) + list(pool.map(lwe, hide))
    with open(out, "w") as f:
        json.dump({"estimator": "malb/lattice-estimator ccb2c74 under passagemath 10.8.13", "rows": rows}, f, indent=1)
    for r in rows:
        if r["problem"] == "sis":
            d, g = r["default"]["lattice"], r["rough"]["lattice"]
            print(f"sis {r['encoding']:6} n={r['n']:4} m={r['m']:6}  default {d['bits']:6} (beta {d['beta']}, d {d['d']})"
                  f"  rough {g['bits']:6} (beta {g['beta']})")
        else:
            best = min(v["bits"] for v in r["rough"].values())
            print(f"lwe {r['dist']:7} k={r['k']:4} samples={r['samples']:4}  rough min {best}")


if __name__ == "__main__":
    main()
