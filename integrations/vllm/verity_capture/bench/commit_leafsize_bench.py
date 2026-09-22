"""Root-ready delay vs leaf size (GPU Triton SHA-256 path) with the opening-size trade-off (EXPERIMENTAL, CMT-1).

    python -m verity_capture.bench.commit_leafsize_bench --out out/gen/r9/cmt-ref/leafsize --reps 5

Streams: the accepted workload's per-step commitment streams (naturally-materialized interface bytes, B0-cov_b0_b1_c256, prompt 155
tokens): prefill step 96,877,828 B, decode step 722,692 B (from bench/binding.json).  For each leaf size S in {64, 256, 4096, 65536}
the step stream is chunked into ceil(bytes/S) pos_leaf leaves (frame: SHA-256("verity/pos-leaf/v0" || u64be(S) || chunk)), the
step tree is folded with the production pairing/lift frames (one Triton launch per level), and the root is read back.
Measured [CUDA events + wall]: leaf hashing, tree fold, root-ready delay (wall from 'stream ready' to root bytes on the host).
Calculated: opening size for ONE value = leaf bytes + 32 B x path length (+ top path 32 B x 4 + ~100 B metadata), verifier hashing
= leaf blocks + path length.  Every timing is on the GPU named in the output; reps interleaved over leaf sizes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "vllm-poc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from verity_capture.experimental.commit import torch_sha256 as TS  # noqa: E402

STREAMS = {"prefill(155 tok)": 96_877_828, "decode(1 tok)": 722_692}
LEAF_SIZES = [64, 256, 4096, 65536]


def one(stream: torch.Tensor, S: int, compiled) -> dict:
    n = math.ceil(stream.numel() / S)
    padded = torch.zeros(n * S, dtype=torch.uint8, device=stream.device)
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e0.record()
    padded[: stream.numel()].copy_(stream)
    leaves = TS.pos_leaf_digests(padded.view(n, S), compiled=compiled)
    e1.record()
    levels = TS.fold_levels(leaves, compiled=compiled)
    e2.record()
    root = bytes(levels[-1][0].cpu().numpy().tobytes())
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    depth = len(levels) - 1
    return {"leaf_bytes": S, "leaves": n, "levels": depth, "leaf_hash_ms": e0.elapsed_time(e1), "tree_ms": e1.elapsed_time(e2),
            "root_ready_ms": (t1 - t0) * 1e3, "root": root.hex(),
            "opening_bytes_one_value": S + 32 * depth + 32 * 4 + 100,
            "verifier_sha_blocks_one_value": math.ceil((26 + S + 9) / 64) + 2 * (depth + 4),
            "sha_blocks_total": n * math.ceil((26 + S + 9) / 64) + 2 * (n - 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--backend", default="triton", choices=["triton", "torch", "compiled"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compiled = {"triton": "triton", "torch": False, "compiled": True}[args.backend]
    cap = torch.cuda.get_device_capability(0)
    gpu = {"name": torch.cuda.get_device_name(0), "sm": f"sm{cap[0]}{cap[1]}", "torch": torch.__version__}
    g = torch.Generator(device="cuda").manual_seed(7)
    streams = {k: torch.randint(0, 256, (v,), dtype=torch.uint8, device="cuda", generator=g) for k, v in STREAMS.items()}
    # warm-up / compile
    t0 = time.perf_counter()
    for S in LEAF_SIZES:
        one(streams["decode(1 tok)"], S, compiled)
    print(f"[leafsize] warm-up {time.perf_counter() - t0:.1f}s on {gpu['name']}", flush=True)
    # correctness spot check of the leaf frame vs hashlib (first leaf, each S)
    checks = []
    for S in LEAF_SIZES:
        s = streams["decode(1 tok)"]
        d = TS.pos_leaf_digests(s[:S].view(1, S), compiled=compiled)[0].cpu().numpy().tobytes()
        ref = hashlib.sha256(TS.POS_LEAF_TAG + S.to_bytes(8, "big") + bytes(s[:S].cpu().numpy().tobytes())).digest()
        checks.append({"leaf_bytes": S, "leaf_digest_equals_hashlib": bytes(d) == ref})
    samples = []
    for rep in range(args.reps):
        for name, s in streams.items():
            for S in LEAF_SIZES:
                r = one(s, S, compiled)
                r.update({"rep": rep, "stream": name, "stream_bytes": s.numel(), "gpu": gpu["name"], "backend": args.backend})
                samples.append(r)
                print(f"[leafsize] rep{rep} {name:16s} S={S:6d} leaves {r['leaves']:9,d} levels {r['levels']:2d} leaf {r['leaf_hash_ms']:8.2f} ms tree {r['tree_ms']:7.2f} ms root-ready {r['root_ready_ms']:8.2f} ms", flush=True)
    with open(out / "raw_samples.jsonl", "w") as f:
        for r in samples:
            f.write(json.dumps(r) + "\n")
    L = [f"## Root-ready delay vs leaf size — {gpu['name']} ({gpu['sm']}), backend {args.backend}, {args.reps} reps interleaved — [measured] GPU/wall; opening & verifier columns [calculated]", "",
         "| stream | leaf B | leaves | levels | leaf hash ms | tree ms | root-ready ms mean [min..max] | root stable | opening B (1 value) | verifier SHA blocks (1 value) | SHA blocks total |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    table = []
    for name in STREAMS:
        for S in LEAF_SIZES:
            rs = [r for r in samples if r["stream"] == name and r["leaf_bytes"] == S]
            rr = [r["root_ready_ms"] for r in rs]
            row = {"stream": name, "leaf_bytes": S, "leaves": rs[0]["leaves"], "levels": rs[0]["levels"],
                   "leaf_hash_ms_mean": st.mean(r["leaf_hash_ms"] for r in rs), "tree_ms_mean": st.mean(r["tree_ms"] for r in rs),
                   "root_ready_ms_mean": st.mean(rr), "root_ready_ms_min": min(rr), "root_ready_ms_max": max(rr),
                   "root_stable": len({r["root"] for r in rs}) == 1, "opening_bytes_one_value": rs[0]["opening_bytes_one_value"],
                   "verifier_sha_blocks_one_value": rs[0]["verifier_sha_blocks_one_value"], "sha_blocks_total": rs[0]["sha_blocks_total"]}
            table.append(row)
            L.append(f"| {name} | {S} | {row['leaves']:,} | {row['levels']} | {row['leaf_hash_ms_mean']:.2f} | {row['tree_ms_mean']:.2f} | "
                     f"{row['root_ready_ms_mean']:.2f} [{row['root_ready_ms_min']:.2f}..{row['root_ready_ms_max']:.2f}] | {row['root_stable']} | "
                     f"{row['opening_bytes_one_value']:,} | {row['verifier_sha_blocks_one_value']} | {row['sha_blocks_total']:,} |")
    L.append("")
    L.append("root-ready = wall from stream-ready to root bytes on the host (includes the D2H of 32 B and the sync).  Opening for one value = leaf bytes + 32 B per step-tree level + top path (4 x 32 B) + ~100 B metadata.  "
             "Leaf frame check vs hashlib: " + ", ".join(f"S={c['leaf_bytes']}: {c['leaf_digest_equals_hashlib']}" for c in checks) + ".")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    json.dump({"gpu": gpu, "backend": args.backend, "reps": args.reps, "streams": STREAMS, "table": table, "checks": checks,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(out / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
