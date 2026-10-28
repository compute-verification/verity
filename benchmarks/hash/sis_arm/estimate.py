"""The lattice estimator on the Module-SIS commitment arm: binding of h = A x mod q and hiding of A x + B r.

Runs under Sage. passagemath's wheels are enough, so no Sage install is needed:

    uv venv --python 3.12 venv && . venv/bin/activate && uv pip install --only-binary :all: passagemath-standard
    git clone https://github.com/malb/lattice-estimator   # this sweep ran at ccb2c74
    PYTHONPATH=lattice-estimator python /path/to/estimate.py OUT.jsonl [--jobs 4]

Each result is one JSON line, written as it completes.

Binding: two openings x != x' of one digest give A (x - x') = 0 mod q with every |x_i - x'_i| <= beta, so SIS in the
l-infinity norm with length_bound beta: 255 for int8 words. `n` is the digest's rows (a module of rank k over a degree-d
ring has n = k d; the estimator, like every published module-lattice estimate, treats the matrix as unstructured). `m`
is the row's words; the attack picks its own sub-dimension.

Four models, log2 of the attack's ring operations:

- `default`: `SIS.estimate`'s lattice attack as shipped (MATZOV cost, GSA shape).
- `rough`: `SIS.estimate.rough` as shipped (ADPS16 Core-SVP 0.292 beta, lgsa shape).
- `slope`, `slope-matzov`: the same lattice attack (ADPS16, resp. MATZOV, with the lgsa shape) with `cost_infinity`'s
  Dilithium-style success probability for every m. As shipped, `cost_infinity` switches to its other branch when
  sqrt(m) * beta <= q, testing the row's m rather than the attack's dimension, and that branch spreads the vector's
  length over every coordinate of the attack lattice, the unit tail included, where a BKZ vector is zero. With
  beta = 255 and q = 2^32 the switch happens for every m <= 2^18, and the estimate then falls as m grows (the attack
  does not get easier with m: it can always zero the extra columns). The Dilithium-style branch spreads it over the
  slope region only and charges (2 beta + 1) / q per q-region coordinate; its result does not depend on m.

Hiding by LWE: with B = [B' | I_n] and r = (s, e), B r = B' s + e is n LWE samples in dimension k with secret and error
uniform int8, so A x + B r is pseudorandom when LWE(k, q, int8, n samples) is hard; r has k + n words. The leftover hash
lemma needs no estimate: it needs 8 (k + n) >= n log2 q + 2 lambda (the note derives it for q = 2^32).
"""
from __future__ import annotations

import json
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log2

Q = 2 ** 32
BETA = 255
SLOPE_N = (160, 176, 192, 208, 224, 240, 256, 288, 320)
SLOPE_M = (4096, 65536)
SHIPPED_N = (128, 160, 192, 224, 256, 320)
ROUGH_M = (1024, 4096, 16384, 65536)
DEFAULT_M = (1024, 4096)
LWE_K = (64, 128, 192, 256, 384, 512, 768, 1024)
LWE_SAMPLES = (192, 224, 256)


def _bits(cost) -> float:
    rop = float(cost["rop"])
    return round(log2(rop), 1) if 0 < rop < float("inf") else float("inf")


def _slope_lattice(p, red_cost_model):
    import estimator.sis_lattice as sl
    src = open(sl.__file__).read()
    body = src[src.index("    def cost_infinity("):src.index("    def cost_zeta(")]
    cond = "if RR(sqrt(d)) * params.length_bound <= params.q:"
    assert body.count(cond) == 1, "lattice-estimator's cost_infinity changed; re-read it before patching"
    ns = dict(vars(sl))
    exec(textwrap.dedent(body.replace(cond, "if False:")), ns)
    shipped = sl.SISLattice.__dict__["cost_infinity"]
    sl.SISLattice.cost_infinity = staticmethod(ns["cost_infinity"])
    try:
        return sl.SISLattice()(p, red_cost_model=red_cost_model, red_shape_model="lgsa")
    finally:
        sl.SISLattice.cost_infinity = shipped


def sis(job: tuple[str, int, int]) -> dict:
    from sage.all import oo
    from estimator import SIS
    from estimator.reduction import RC
    model, n, m = job
    p = SIS.Parameters(n=n, q=Q, length_bound=BETA, m=m, norm=oo, tag=f"int8-q32-n{n}-m{m}")
    if model == "default":
        c = SIS.estimate(p, jobs=1, quiet=True)["lattice"]
    elif model == "rough":
        c = SIS.estimate.rough(p, quiet=True)["lattice"]
    else:
        c = _slope_lattice(p, RC.MATZOV if model == "slope-matzov" else RC.ADPS16)
    return {"problem": "sis", "model": model, "beta_inf": BETA, "q": Q, "n": n, "m": m, "bits": _bits(c),
            "bkz_beta": int(c.get("beta", 0)), "d": int(c.get("d", 0)), "zeta": int(c.get("zeta", 0) or 0)}


def lwe(job: tuple[int, int]) -> dict:
    from estimator import LWE, ND
    k, samples = job
    x = ND.Uniform(-128, 127)
    p = LWE.Parameters(n=k, q=Q, Xs=x, Xe=x, m=samples, tag=f"hide-int8-k{k}-n{samples}")
    rough = LWE.estimate.rough(p, quiet=True)
    per = {a: _bits(c) for a, c in rough.items()}
    return {"problem": "lwe", "model": "rough", "dist": "int8", "k": k, "samples": samples, "q": Q,
            "bits": min(per.values()), "attacks": per}


def main() -> None:
    out = sys.argv[1]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    work = [(sis, (model, n, m)) for model in ("slope", "slope-matzov") for n in SLOPE_N for m in SLOPE_M[:1]]
    work += [(lwe, (k, s)) for s in LWE_SAMPLES for k in LWE_K]
    work += [(sis, ("rough", n, m)) for m in ROUGH_M for n in SHIPPED_N]
    work += [(sis, ("default", n, m)) for m in DEFAULT_M for n in SHIPPED_N]
    work += [(sis, (model, n, SLOPE_M[1])) for model in ("slope", "slope-matzov") for n in (192, 224, 256)]
    with ProcessPoolExecutor(jobs) as pool, open(out, "a") as f:
        futs = {pool.submit(fn, job): job for fn, job in work}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:  # one failed point is a row, not a lost sweep
                row = {"problem": "error", "job": list(futs[fut]), "error": repr(e)}
            row["estimator"] = "malb/lattice-estimator ccb2c74 under passagemath 10.8.13"
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
