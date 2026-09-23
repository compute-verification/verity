"""Store declaration for the registered-relation Ligero VU benchmarks (research.store README §6.1): ``--relation fp8-ada``
(the FP8 Ada E4M3 unit it was written for) and, since lane b-hopper, ``bf16-hopper`` / ``fp8-hopper`` on the same generic
runner (``relchain.py``) -- ``relation`` is a key param, so the three never share a derivation.

    research run --tool bench_vu_fp8 --project verity --scratch triton -- python -m backends.direct.ligero.run --relation fp8-ada bench-vu ...
    research run --tool bench_vu_fp8 --project verity --scratch triton -- python -m backends.direct.ligero.run --relation bf16-hopper bench-vu ...

The key params are ``tool.py``'s (the same argparse) plus ``relation``; the instances are synthetic (a seed in relations.py),
so ``--root`` is not a key param here.  Nothing here imports torch or the backend.
"""
from __future__ import annotations

import argparse

from research.store.tool import Tool

SUBCOMMANDS = ("gate", "bench", "gate-vu", "bench-vu")      # run.py's (the registry may load this file standalone)


def parse(argv: list[str]) -> dict:
    idx = next((i for i, t in enumerate(argv) if t in SUBCOMMANDS), None)
    if idx is None:
        return {"_unparsed": argv}
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--relation", default="bf16")
    ns_pre, _ = pre.parse_known_args(argv[:idx])
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target", type=float, default=-128.0)
    ap.add_argument("--rate-log2", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zk", action="store_true")
    ap.add_argument("--mode", default="interactive")
    ap.add_argument("--vus", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--total-vus", type=int, default=None)
    ap.add_argument("--row-negatives", type=int, default=60)
    ap.add_argument("--dump-dir", default=None)          # where the proof files go (bench-vu); not a key param
    ap.add_argument("--dump-reps", type=int, default=None)       # how many recorded reps are written out, not what is computed
    ap.add_argument("--instance-procs", type=int, default=None)  # processes recording the synthetic set: how, not what
    ap.add_argument("--instances-cache", default=None)           # where the synthetic set is kept between runs (deterministic in the seed)
    ns, unknown = ap.parse_known_args(argv[idx + 1:])
    out = vars(ns)
    out["cmd"] = argv[idx]
    out["relation"] = ns_pre.relation
    if unknown:
        out["_unknown"] = unknown
    return out


BENCH_VU_FP8 = Tool(
    name="bench_vu_fp8", version="1",
    command=["python", "-m", "backends.direct.ligero.run"],
    closure=["backends/direct/ligero/**", "backends/numerical/**", "packages/verity/src/verity/**"],
    key_params=["cmd", "relation", "mode", "zk", "batch", "total_vus", "vus", "reps", "rate_log2", "target", "row_negatives"],
    key_conditions=["hw.family"],
    caches=["triton"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)
