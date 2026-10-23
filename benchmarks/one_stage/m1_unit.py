"""M1: one step family's unit through C-Flock, timed: the Definition a Program row names (its ``spec``) as M0's `UnitStager`
statement on the serving rows' ports (an input row per parameter and an output row per returned field, each a
``hm96-sha512/row/v2`` row of its u16 words), flat (`ir_lower`) or structural (``--structural``: `struct_lower`'s template,
the flat unit never built), and with ``--bin`` and ``--lean`` one round of ``--n`` units staged on finite bf16 values (every
parameter of a bf16 family; a stand-in for the served values, which only the window reads), proved by M0 under ``--zk``
(`a0.session`: serve and prove, the verifier's subset draw of all of them) and verified by the Lean verifier of record
(`a0.lean_verify`).

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/m1_unit.py --spec 'RMSNormFusedCudaVector8_v1{N=6144,EPS=...,BLOCK=256}' --out DIR \\
      [--structural iv/hybrid/4] [--bin flock-circuit --lean flock-verify] [--n 1]

Prints ``M1U {json}`` per phase: the build's seconds and peak RSS, the unit's ANDs and rows, k_log, vus_per_block and
chunk_cap (`a6.chunk_cap`'s rule), then the stage, the session and Lean's verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.setrecursionlimit(100000)

M_MAX = 35


def say(phase: str, **kw) -> None:
    print("M1U " + json.dumps({"phase": phase, "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2), **kw},
                              default=str), flush=True)


def bf16_rows(rng: np.random.Generator, n: int, words: int) -> np.ndarray:
    """``n`` rows of ``words`` finite bf16 values, N(0, 1) rounded to bf16 by truncation."""
    f = rng.standard_normal((n, words)).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--structural", default=None, metavar="LAYOUT", help="the structural stager at this layout mode (iv/hybrid/4)")
    ap.add_argument("--bin", default=None, help="M0's flock-circuit: prove one round")
    ap.add_argument("--lean", default=None, help="the Lean verifier of record (flock-verify)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--port", type=int, default=7611)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import a0
    import a11_kinds as K
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL
    from verity_vllm.pipeline import serving_rows as SRW

    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = SRW.template(a.spec).fn
    params, ret = fn.signature()
    rows = [(nm, V2, K.leaf_bits(t), [nm]) for nm, t in params]
    n_out = len(ret.fields) if hasattr(ret, "fields") else 1
    say("bound", spec=a.spec, word_gates=fn.gates, rows=[[r, b] for r, _s, b, _p in rows], outputs=n_out)

    d = out / "stage"
    t = time.time()
    with K.no_cyclic_gc():
        st = CS.UnitStager(fn, f"m1/{K.family(fn.id)}", rows, K.out_rows(n_out), circuit=d / "circuit.txt",
                           **({"layout": a.structural, "structural": True} if a.structural else {}))
    m, c = st.meta, st.low.layout.counts
    cap = int(m["vus_per_block"]) << (M_MAX - int(m["k_log"])) if M_MAX - int(m["k_log"]) >= 3 else 0
    say("built", seconds=round(time.time() - t, 1), structural=a.structural, ands=c["ands"], unit_rows=c.get("rows"),
        k_log=m["k_log"], unit_log=m["unit_log"], vus_per_block=m["vus_per_block"], chunk_cap=cap, pin=st.pin,
        circuit_bytes=(d / "circuit.txt").stat().st_size, domain=st.low.domain)
    if not a.bin:
        return 0

    rng = np.random.default_rng(a.seed)
    vals = {r: bf16_rows(rng, a.n, b // 16) for r, _s, b, _p in rows}
    t = time.time()
    with K.no_cyclic_gc():
        got = st.stage(vals, d, os.urandom(32))
    for f in ("inst", "pub"):
        (d / f"{f}.bin").rename(d / f"{f}-{a.n}.bin")
    IL.write_tables(d / "tables")
    say("staged", seconds=round(time.time() - t, 1), n=a.n, ands=got["ands"])

    t = time.time()
    record, live = a0.session(a.bin, {"dir": d, "pin": st.pin}, a.n, f"subset:{a.n}", a.port, out / "session", zk=True)
    say("proved", seconds=round(time.time() - t, 1), accepted=(live or {}).get("accepted"), why=(live or {}).get("why"),
        **{k: (live or {}).get(k) for k in ("k_log", "nbl", "dense_m", "prove_total_s", "witness_s", "proof_bytes", "units")},
        verdict=(record or {}).get("verdict"))
    v = a0.lean_verify(a.lean, {"dir": d}, a.n, out / "session", zk=True)
    say("verified", **v)
    return 0 if (live or {}).get("accepted") and v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
