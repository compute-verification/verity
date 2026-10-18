"""Commit overhead as a share of serve time, projected from committed bytes per step and a hash's throughput.

    python commit_model.py                       # the seeded tables (markdown): SHA-512 today, and the throughput 5% needs
    python commit_model.py --gbps 900 --leaf 4096 [--hash-name X]   # one candidate hash at a measured GB/s
    python commit_model.py --rate 7.13e9 --per-leaf 44 --leaf 4096  # or at compressions/s and compressions per leaf
    python commit_model.py --target 0.05 --json  # every number as JSON
    python commit_model.py --check               # the seeds reproduce circuits' published cells, or exit 1

The model is throughput only: a phase's hashing time is the bytes it commits over the hash's GB/s, and its overhead is that over
the phase's stock serve time. It assumes hashing does not overlap serving, which circuits measured on H200 (hashing shares the
integer issue slots with the GEMMs; note:circuits-rows-sha512-floor, note:circuits-commit-defaults-overlap). It leaves out
gathers, staging copies, device-to-host traffic and host work; the baseline note breaks those down per run.

A coarser proof unit keeps a fraction of a step's bytes: the fraction applies to the bytes inside the decoder layers and the
embedding, and the logits (committed for decode tokens) are kept whole.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass

GB = 1e9

#: Compressions an hm96-sha512 row leaf in a frame-v3-sha512 tree costs besides its row: the salt y (3), c (2), the hm96 leaf (2),
#: the frame leaf (2) and its share of the tree's nodes (about 2). benchmarks/pouw/commit_gpu/README.md, "Leaf size".
SHA512_LEAF_FIXED = 11
#: Of those, the salt's (y and c), which the served kernel computes inside its leaf kernel.
SHA512_SALT = 5


def sha512_row_compressions(leaf_bytes: int) -> int:
    """`sha512/row/v2` over `leaf_bytes`: its constant 128-byte first block is precomputed, then the row plus 17 bytes of padding."""
    return math.ceil((leaf_bytes + 17) / 128)


def sha512_leaf_compressions(leaf_bytes: int) -> int:
    """Compressions per committed leaf of today's rows-sha512 scheme: 14 at 256 B, 44 at 4 KiB (note:circuits-commit-bytes-width-dma)."""
    return sha512_row_compressions(leaf_bytes) + SHA512_LEAF_FIXED


def gbps_from_rate(compressions_per_s: float, leaf_bytes: int, per_leaf: float) -> float:
    return compressions_per_s * leaf_bytes / per_leaf / GB


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    bytes_per_step: float          # committed bytes per step today (the Commit's per-Call boundaries)
    serve_ms_per_step: float       # stock serve time per step
    logits_bytes_per_step: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    gpu: str
    phases: tuple[Phase, ...]
    #: unit -> fraction of the non-logit bytes it keeps; "call" is today
    units: dict[str, float]
    source: str


@dataclass(frozen=True)
class Rate:
    name: str
    gbps: dict[int, float]         # leaf bytes -> GB/s
    source: str


def _frac(per_token_layer: float, today: float) -> float:
    return per_token_layer / today


# Per token per layer, today (compiled, layout_bytes.py) and what a unit keeps (granularity.py): note:circuits-commit-bytes-width-dma
# section 1, note:circuits-commit-defaults-overlap "Update (04:15Z)". block = layer + o_proj's output + the input norm's residual output;
# layer = K + V + the two hidden vectors vLLM's fused add-norm reads; layer-folded = K + V + one hidden vector (their sum).
LLAMA_TODAY, QWEN_TODAY = 84_736, 145_180
UNITS_LLAMA = {"call": 1.0, "block": _frac(18_432, LLAMA_TODAY), "layer": _frac(10_240, LLAMA_TODAY),
               "layer-folded": _frac(6_144, LLAMA_TODAY)}
UNITS_QWEN = {"call": 1.0, "block": _frac(36_864, QWEN_TODAY), "layer": _frac(20_480, QWEN_TODAY),
              "layer-folded": _frac(12_288, QWEN_TODAY)}

WORKLOADS = (
    Workload(
        "Llama-3.2-1B B32 compiled", "H200",
        (Phase("prefill", 4, 3.51328e9, 9.350, 5.130e6), Phase("decode", 63, 41.567e6, 1.300, 6.580e6)),
        UNITS_LLAMA,
        "bytes: layouts of r20261018-012341-e3b1 (b32-p512-mixed, 16.672 GB a run); serve: its stock-prof arm, nsys GPU ms per step "
        "(run 8, vy-glm53-s1)"),
    Workload(
        "Qwen3-8B B32 compiled", "H200",
        (Phase("prefill", 4, 13.4908e9, 57.423, 6.077e6), Phase("decode", 63, 142.279e6, 5.555, 7.795e6)),
        UNITS_QWEN,
        "bytes: layouts of r20261018-031955-d2b3 (b32-p512-mixed, 62.927 GB a run); serve: its stock-prof arm, nsys GPU ms per step "
        "(run 8, vy-glm53-s1)"),
    Workload(
        "GLM-5.3 TP8 eager B1, per rank", "H200",
        (Phase("run", 1, 7.184635164e9, 5713.7),),
        {"call": 1.0},
        "Commit of record r20261016-213336-6c35: bytes_bound per rank (28,064,992 leaves of 256 B, 33 steps), serve = the control "
        "run's wall (mean of 3; eager and host-bound, so wall, not GPU busy)"),
)

RATES = (
    Rate("SHA-512 H200 measured", {256: 66.12, 1024: 152.5, 4096: 230.6},
         "rows_floor r20261017-234343-61c6 (leaves + tree of a 256 MiB window; 4 KiB at 2 blocks/SM)"),
    Rate("SHA-512 H200 raw rate", {L: gbps_from_rate(4.79e9, L, sha512_leaf_compressions(L)) for L in (256, 1024, 4096)},
         "4.79 G compressions/s (r20261017-234343-61c6, COMMIT-GPU's sha mode on sm_90) / sha512_leaf_compressions"),
    Rate("SHA-512 sm_120 raw rate", {L: gbps_from_rate(7.13e9, L, sha512_leaf_compressions(L)) for L in (256, 1024, 4096)},
         "7.13 G compressions/s (r20261016-110413-c5d1, RTX PRO 6000) / sha512_leaf_compressions; against H200 serve times"),
)

#: Data-sheet HBM bandwidth, GB/s: a hash that reads committed bytes back from memory runs no faster than this.
HBM_GBPS = {"H200": 4800.0, "RTX PRO 6000": 1792.0}


def kept_bytes(p: Phase, frac: float) -> float:
    return p.logits_bytes_per_step + (p.bytes_per_step - p.logits_bytes_per_step) * frac


def overhead(w: Workload, gbps: float, unit: str = "call") -> dict[str, float]:
    """Hashing time over serve time, per phase and for the run (the steps summed)."""
    frac = w.units[unit]
    out, hash_ms, serve_ms = {}, 0.0, 0.0
    for p in w.phases:
        h = kept_bytes(p, frac) / (gbps * GB) * 1e3
        out[p.name] = h / p.serve_ms_per_step
        hash_ms += h * p.steps
        serve_ms += p.serve_ms_per_step * p.steps
    if len(w.phases) > 1:
        out["run"] = hash_ms / serve_ms
    return out


def required_gbps(w: Workload, target: float, unit: str = "call") -> dict[str, float]:
    """The GB/s at which hashing is `target` of serve time, per phase and for the run."""
    frac = w.units[unit]
    out = {p.name: kept_bytes(p, frac) / (target * p.serve_ms_per_step / 1e3) / GB for p in w.phases}
    if len(w.phases) > 1:
        out["run"] = (sum(kept_bytes(p, frac) * p.steps for p in w.phases)
                      / (target * sum(p.serve_ms_per_step * p.steps for p in w.phases) / 1e3) / GB)
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%" if x < 10 else f"{100 * x:,.0f}%"


def table_today(rates=RATES, leaves=(256, 4096)) -> str:
    cols = [(r, L) for r in rates for L in leaves if L in r.gbps]
    head = "| workload | phase | " + " | ".join(f"{r.name}, {L} B ({r.gbps[L]:.0f} GB/s)" for r, L in cols) + " |"
    rows = [head, "|" + "---|" * (2 + len(cols))]
    for w in WORKLOADS:
        per = {(r.name, L): overhead(w, r.gbps[L]) for r, L in cols}
        for ph in per[(cols[0][0].name, cols[0][1])]:
            rows.append(f"| {w.name} | {ph} | " + " | ".join(_pct(per[(r.name, L)][ph]) for r, L in cols) + " |")
    return "\n".join(rows)


def table_required(target: float, ref: Rate = RATES[0], ref_leaf: int = 4096) -> str:
    rows = [f"| workload | unit | phase | GB/s for {_pct(target)} | x {ref.name} at {ref_leaf} B | x HBM |", "|---|---|---|---|---|---|"]
    for w in WORKLOADS:
        for unit in w.units:
            for ph, g in required_gbps(w, target, unit).items():
                rows.append(f"| {w.name} | {unit} | {ph} | {g:,.0f} | {g / ref.gbps[ref_leaf]:.1f} | {g / HBM_GBPS[w.gpu]:.2f} |")
    return "\n".join(rows)


#: Cells of circuits' table (note:circuits-commit-bytes-width-dma section 1): GPU hashing at the H200 floor over stock serve, prefill.
CHECKS = (
    ("Llama-3.2-1B B32 compiled", "call", 256, "prefill", 5.68), ("Llama-3.2-1B B32 compiled", "call", 4096, "prefill", 1.63),
    ("Qwen3-8B B32 compiled", "call", 256, "prefill", 3.55), ("Qwen3-8B B32 compiled", "call", 4096, "prefill", 1.02),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 256, "prefill", 0.43),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 4096, "prefill", 0.125),
    ("Qwen3-8B B32 compiled", "layer-folded", 256, "prefill", 0.31), ("Qwen3-8B B32 compiled", "layer-folded", 4096, "prefill", 0.088),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 4096, "run", 0.059), ("Qwen3-8B B32 compiled", "layer-folded", 4096, "run", 0.044),
)


def check(tol: float = 0.05) -> list[str]:
    """Every CHECKS cell within `tol` (relative), every leaf count as circuits states it, and each workload's bytes summing to the run's."""
    bad = []
    by_name = {w.name: w for w in WORKLOADS}
    for name, unit, leaf, ph, want in CHECKS:
        got = overhead(by_name[name], RATES[0].gbps[leaf], unit)[ph]
        if abs(got / want - 1) > tol:
            bad.append(f"{name} {unit} {leaf} B {ph}: model {got:.3f}, circuits {want}; a seed moved, fix it or its source")
    if (sha512_leaf_compressions(256), sha512_leaf_compressions(4096)) != (14, 44):
        bad.append("rows-sha512 compressions per leaf are 14 at 256 B and 44 at 4 KiB (note:circuits-commit-bytes-width-dma)")
    for name, total in (("Llama-3.2-1B B32 compiled", 16.672e9), ("Qwen3-8B B32 compiled", 62.927e9)):
        got = sum(p.bytes_per_step * p.steps for p in by_name[name].phases)
        if abs(got / total - 1) > 1e-3:
            bad.append(f"{name}: phases sum to {got / GB:.3f} GB, the run commits {total / GB:.3f} GB")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=float, default=0.05, help="overhead budget as a fraction of serve time (default 0.05)")
    ap.add_argument("--gbps", type=float, help="a candidate hash's measured throughput, GB/s of committed bytes")
    ap.add_argument("--rate", type=float, help="or its compressions (or blocks) per second ...")
    ap.add_argument("--per-leaf", type=float, help="... and compressions per leaf, tree share included")
    ap.add_argument("--leaf", type=int, default=4096, help="leaf bytes the candidate's rate is for (default 4096)")
    ap.add_argument("--hash-name", default="candidate")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if a.check:
        bad = check()
        print("\n".join(bad) if bad else f"ok: {len(CHECKS)} cells of circuits' table, leaf counts and run bytes reproduce")
        return 1 if bad else 0
    rates = RATES
    if a.gbps or a.rate:
        if a.rate and not a.per_leaf:
            ap.error("--rate needs --per-leaf")
        g = a.gbps or gbps_from_rate(a.rate, a.leaf, a.per_leaf)
        rates = (Rate(a.hash_name, {a.leaf: g}, "command line"),)
    if a.json:
        print(json.dumps({
            "target": a.target,
            "workloads": [asdict(w) for w in WORKLOADS],
            "rates": [asdict(r) for r in rates],
            "overhead": {w.name: {r.name: {str(L): {u: overhead(w, g, u) for u in w.units} for L, g in r.gbps.items()} for r in rates}
                         for w in WORKLOADS},
            "required_gbps": {w.name: {u: required_gbps(w, a.target, u) for u in w.units} for w in WORKLOADS},
        }, indent=1))
        return 0
    leaves = sorted({L for r in rates for L in r.gbps}) if rates is not RATES else (256, 4096)
    print(f"Overhead at today's bytes (per-Call units), hashing over stock serve time:\n\n{table_today(rates, leaves)}\n")
    print(f"Throughput a hash needs for {_pct(a.target)} of serve time:\n\n{table_required(a.target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
