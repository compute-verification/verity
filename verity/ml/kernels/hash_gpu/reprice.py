"""Re-price ``explore.tensor`` with the measured 4090 Merkle rates (hash-gpu lane, deliverable 3).

    uv run --frozen --package verity-numerical python verity/ml/kernels/hash_gpu/reprice.py \
        --bench <fetched bench.json> --run-id r2026... --out backends/numerical/reports/hash_gpu.json

For every hash measured at the Ligero shapes of A-GKR and B-Ligero (BabyBear, 2^-128, B = 4096) this takes the
full-tree bytes/s on the RTX 4090 and evaluates ``Priced.times`` at TensorZKP's realistic utilisation (0.172) on:
the 4090 itself (measured rate, the 4090's tensor peak), and the A100 / H100 with the hash rate scaled by the
stated HBM/GDDR bandwidth ratio (2,039 / 1,008 and 3,350 / 1,008) -- a SCALING, labelled as such, not a measurement
(the SHA-256 and Blake3 kernels are SIMT-compute-bound, so bandwidth scaling is an upper bound for them).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as t

UTIL = 0.172
GPUS = {"RTX4090": t.GPU_RTX4090, "A100": t.GPUS["A100"], "H100": t.GPUS["H100"]}


def hash_key(r: dict) -> str:
    h = r["hash"]
    if h.startswith("poseidon2"):
        h += "/" + r.get("backend", r.get("mds", "torch"))
    return h


def measured_rates(bench: dict) -> dict[str, dict[str, dict]]:
    """{candidate: {hash: {bytes_per_s, tree_ms, ...}}} from the Ligero rows (full shapes only)."""
    out: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    for r in bench["ligero"]:
        if r.get("scaled_from_columns"):
            key = hash_key(r) + "-torch"
            rate = r["full_shape_bytes"] / (r["full_shape_tree_ms_scaled"] / 1e3)
            out[r["candidate"]][key] = {"bytes_per_s": rate, "tree_ms": r["full_shape_tree_ms_scaled"],
                                        "basis": f"torch elementwise on a {r['scaled_from_columns']}-column slice, scaled by columns"}
        else:
            out[r["candidate"]][hash_key(r)] = {"bytes_per_s": r["tree_bytes_per_s"], "tree_ms": r["tree_ms_median"],
                                                "leaf_ms": r["leaf_ms_median"], "bytes": r["bytes"],
                                                "device_memory_in_use_bytes": r["device_memory_in_use_bytes"],
                                                "basis": "measured, full Ligero shape, median of CUDA-event timings"}
    return out


def reprice(bench: dict, run_id: str | None) -> dict:
    rates = measured_rates(bench)
    fs = t.FIELDS["babybear"]
    rows = []
    for dname in ("A", "B"):
        design = t.DESIGNS[dname]
        p = t.price_design(design, fs, -128)
        base = {"design": dname, "field": "babybear", "target_log2": -128, "B": t.B_TARGET, "K": t.K_TARGET,
                "hash_bytes_per_batch": p.hash_bytes, "hash_bytes_per_vu": p.hash_bytes / t.B_TARGET,
                "committed_bytes_per_vu": p.committed_bytes / t.B_TARGET}
        # the model's stated rates for reference
        for hk, hv in t.HASH_RATES.items():
            if hv["bytes_per_s"]:
                rows.append(_row(base, p, fs, f"model:{hk}", hv["bytes_per_s"], "assumed (tensor.md)", "H100", 1.0))
        for hname, m in rates[dname].items():
            for gkey, gpu in GPUS.items():
                scale = 1.0 if gkey == "RTX4090" else gpu.hbm_bytes_per_s / t.GPU_RTX4090.hbm_bytes_per_s
                label = ("measured on RTX 4090" if gkey == "RTX4090"
                         else f"SCALING: 4090 rate x {scale:.2f} (bandwidth ratio {gpu.hbm_bytes_per_s / 1e9:.0f}/1008 GB/s)")
                rows.append(_row(base, p, fs, hname, m["bytes_per_s"] * scale, label, gkey, scale, gpu))
    ranking = {}
    for gkey in GPUS:
        sub = [r for r in rows if r["gpu"] == gkey and not r["hash"].startswith("model:")]
        ranking[gkey] = sorted(({"design": r["design"], "hash": r["hash"], "overhead": r["overhead.vs_native_peak"],
                                 "hash_share": r["hash_share_of_time"]} for r in sub), key=lambda x: x["overhead"])
    return {"meta": {"lane": "hash-gpu", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": run_id,
                     "device": bench["device"], "utilisation": UTIL, "native_seconds_per_vu": contract.native_seconds_per_vu(t.K_TARGET),
                     "scope": "component: proof-internal Merkle commitment hashing; every overhead is the tensor.py cost model with one input (the hash rate) replaced by a measurement"},
            "measured_rates": rates, "generic_trees": bench.get("generic", []), "permutation": bench.get("permutation", []),
            "rows": rows, "ranking": ranking}


def _row(base, p, fs, hname, rate, label, gkey, scale, gpu=None):
    gpu = gpu or t.GPUS["H100"]
    tm = p.times(gpu, gkey, UTIL, rate, "rs_matrix_ntt", fs)
    return {**base, "hash": hname, "gpu": gkey, "hash_bytes_per_s": rate, "rate_label": label, "rate_scale": scale,
            "t.hash_per_batch": tm["t.hash"], "t.hash_per_vu": tm["t.hash"] / t.B_TARGET, "t.total_per_batch": tm["t.total"],
            "hash_share_of_time": tm["t.hash"] / tm["t.total"], "matmul_share_of_time": tm["matmul_share_of_time"],
            "seconds_per_vu": tm["seconds_per_vu"], "overhead.vs_native_peak": tm["overhead.vs_native_peak"],
            "overhead.hash_bucket_only": contract.overhead(tm["t.hash"] / t.B_TARGET, t.K_TARGET)}


def table(rep: dict) -> str:
    lines = [f"{'design':6} {'hash':26} {'gpu':8} {'rate GB/s':>10} {'hash ms/batch':>14} {'hash us/VU':>11} {'hash%':>6} {'overhead':>9}  label"]
    for r in rep["rows"]:
        lines.append(f"{r['design']:6} {r['hash']:26} {r['gpu']:8} {r['hash_bytes_per_s'] / 1e9:10.1f} {r['t.hash_per_batch'] * 1e3:14.1f} "
                     f"{r['t.hash_per_vu'] * 1e6:11.2f} {r['hash_share_of_time']:6.1%} {r['overhead.vs_native_peak']:9.3g}  {r['rate_label']}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    bench = json.loads(Path(a.bench).read_text())
    rep = reprice(bench, a.run_id)
    print(table(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
