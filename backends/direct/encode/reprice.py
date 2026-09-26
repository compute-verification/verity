"""Re-price B-Ligero and A-GKR with the MEASURED encoder (this lane) and the MEASURED SHA-256 Merkle rate (hash-gpu lane).

    uv run --frozen --package verity-numerical python backends/direct/encode/reprice.py \
        --bench backends/direct/encode/results_<run>.json --run-id <run> --out backends/numerical/reports/encode_gpu.json

Every row is ``explore.tensor``'s cost model with two inputs replaced by measurements: the encoding bucket
(``Priced.times(encoder_seconds_per_row=...)``, the best SIMT NTT's seconds per Ligero row on the RTX 4090) and the
hash rate (``hash_gpu.json``: SHA-256 full tree, 446 GB/s at B's shape, 425 at A's).  The remaining buckets (Ligero
row combination and A's GKR matmuls at utilisation 0.172, SIMT field work at the model's rate, hint generation) stay
MODELLED and are listed as ``omitted_measurements``.  A100 / H100 rows are SCALINGS, labelled: the encoder is
memory-bound (0.78 of the HBM roofline), so it scales with the bandwidth ratio (2.02x / 3.32x); the SHA-256 rate is
scaled the same way (an upper bound: the kernel is SIMT-compute-bound, hash_gpu.md); the modelled matmul buckets scale
with the INT8 peak and the SIMT buckets with the FP32 peak, as ``Priced.times`` does for those GPUs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as t

UTIL = 0.172
GPUS = {"RTX4090": t.GPU_RTX4090, "A100": t.GPUS["A100"], "H100": t.GPUS["H100"]}
HASH_4090 = {"B": 446e9, "A": 425e9}       # hash_gpu.md, SHA-256 sequential-leaf full tree, run r20260922-054241-b966


def best_encoder(bench: dict, shape: str) -> dict:
    rows = [r for r in bench["encoders"] if r["shape"] == shape and r["kind"] == "simt"]
    return min(rows, key=lambda r: r["ms_median"])


def reprice(bench: dict, run_id: str | None) -> dict:
    fs = t.FIELDS["babybear"]
    out_rows = []
    for dname in ("B", "A"):
        p = t.price_design(t.DESIGNS[dname], fs, -128)
        enc = best_encoder(bench, dname)
        spr = enc["ms_median"] / 1e3 / enc["rows"]
        base = {"design": dname, "field": "babybear", "target_log2": -128, "B": t.B_TARGET, "K": t.K_TARGET, "ligero_rows": p.ligero_rows,
                "encoder": enc["encoder"], "encoder_us_per_row_4090": spr * 1e6, "encoder_ms_per_batch_4090": enc["ms_median"],
                "encoder_fraction_of_hbm_floor": enc["fraction_of_hbm_floor"]}
        # the model as it stood (hash-gpu re-pricing: measured hash, modelled rs_matrix_ntt encoder)
        tm = p.times(t.GPU_RTX4090, "RTX4090", UTIL, HASH_4090[dname], "rs_matrix_ntt", fs)
        out_rows.append(_row(base, tm, "RTX4090", "model encoder rs_matrix_ntt@0.172 + measured SHA-256 (hash_gpu.md)", 1.0, 1.0))
        for gkey, gpu in GPUS.items():
            bw = gpu.hbm_bytes_per_s / t.GPU_RTX4090.hbm_bytes_per_s
            tm = p.times(gpu, gkey, UTIL, HASH_4090[dname] * bw, "rs_matrix_ntt", fs, encoder_seconds_per_row=spr / bw)
            label = ("measured encoder + measured SHA-256 on RTX 4090" if gkey == "RTX4090"
                     else f"SCALING: encoder and SHA-256 x {bw:.2f} (bandwidth ratio), matmul/SIMT buckets at {gkey} peaks")
            out_rows.append(_row(base, tm, gkey, label, bw, bw))
    return {"meta": {"lane": "b-encode", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": run_id,
                     "hash_run_id": "r20260922-054241-b966", "device": bench["device"], "utilisation_modelled_matmul": UTIL,
                     "native_seconds_per_vu": contract.native_seconds_per_vu(t.K_TARGET),
                     "scope": "ARITHMETIC_DIAGNOSTIC-level re-pricing: tensor.py cost model with the encoding bucket and the hash rate replaced by measurements",
                     "omitted_measurements": ["witness/hint generation (modelled: 2e4 int ops per unit)", "Ligero row combination (modelled matmul @0.172)",
                                              "linear/quadratic tests, constraint combination, helper inversions (modelled SIMT)",
                                              "openings and transcript serialisation (not priced)", "A: GKR sumcheck and logUp GKR (modelled @0.172; tc_sumcheck_4090.md measured ~0.014)"]},
            "rows": out_rows}


def _row(base, tm, gkey, label, enc_scale, hash_scale):
    return {**base, "gpu": gkey, "label": label, "encoder_scale": enc_scale, "hash_scale": hash_scale,
            "t.encoding": tm["t.encoding"], "t.hash": tm["t.hash"], "t.matmul_other": tm["t.matmul"] - (0.0 if tm["encoding_measured"] else tm["t.encoding"]),
            "t.simt_field": tm["t.simt_field"], "t.hint_gen": tm["t.hint_gen"], "t.total": tm["t.total"],
            "encoding_share": tm["t.encoding"] / tm["t.total"], "hash_share": tm["t.hash"] / tm["t.total"],
            "seconds_per_vu": tm["seconds_per_vu"], "overhead.vs_native_peak": tm["overhead.vs_native_peak"]}


def table(rep: dict) -> str:
    lines = [f"{'design':6} {'gpu':8} {'enc ms':>7} {'hash ms':>8} {'mm ms':>6} {'simt ms':>8} {'total ms':>9} {'enc%':>5} {'hash%':>6} {'overhead':>9}  label"]
    for r in rep["rows"]:
        lines.append(f"{r['design']:6} {r['gpu']:8} {r['t.encoding'] * 1e3:7.1f} {r['t.hash'] * 1e3:8.1f} {r['t.matmul_other'] * 1e3:6.1f} "
                     f"{r['t.simt_field'] * 1e3:8.1f} {r['t.total'] * 1e3:9.1f} {r['encoding_share']:5.0%} {r['hash_share']:6.0%} "
                     f"{r['overhead.vs_native_peak']:9.3g}  {r['label']}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    bench = json.loads(Path(a.bench).read_text())
    bench = bench.get("bench", bench)
    rep = reprice(bench, a.run_id)
    print(table(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
