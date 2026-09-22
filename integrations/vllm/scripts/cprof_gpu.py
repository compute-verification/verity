#!/usr/bin/env python3
"""CPROF: the GPU/PCIe side of the commitment path (§32.11).

The current commitment implementation hashes on the CPU, so the only device work in the
path is materialisation: every committed value is copied device -> host before it is
hashed (activations inside the capture's ``emit``, weights inside the replay's weights
stage via ``p.detach().contiguous().cpu()`` + ``.numpy().tobytes()``). This script measures:

1. the weights-stage decomposition on the real checkpoint: D2H copy vs ``tobytes`` host
   copy vs sha256+leaf hash vs disk write, per tensor and total (the split behind
   ``replay.json timing.weights.seconds``);
2. clean PCIe D2H / H2D bandwidth (pageable and pinned, 1 GiB buffer);
3. device HBM effective bandwidth (device->device copy of 1 GiB: read+write traffic) --
   the denominator of the in-kernel commitment opportunity bound;
4. GPU properties.

Usage:
    python scripts/cprof_gpu.py --checkpoint <safetensors dir> \
        --out docs/data/commitment-prof/gpu-side.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC),):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from verity_vllm import hashing  # noqa: E402


def sync() -> None:
    torch.cuda.synchronize()


def med(fn, reps: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dir holding model.safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    doc: dict = {
        "schema": "veritor/cprof/gpu-side/v1",
        "gpu": {"name": props.name, "sms": props.multi_processor_count,
                "memory_gb": props.total_memory / 2**30,
                "cc": f"{props.major}.{props.minor}"},
        "torch": torch.__version__,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 2/3: clean bandwidths, 1 GiB
    n = 1 << 30
    d_a = torch.empty(n, dtype=torch.uint8, device=dev)
    d_b = torch.empty(n, dtype=torch.uint8, device=dev)
    h_pageable = torch.empty(n, dtype=torch.uint8)
    h_pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    doc["bandwidth"] = {
        "d2d_copy_gb_per_s_traffic": 2 * n / med(lambda: d_a.copy_(d_b)) / 1e9,
        "d2h_pageable_gb_per_s": n / med(lambda: h_pageable.copy_(d_a)) / 1e9,
        "d2h_pinned_gb_per_s": n / med(lambda: h_pinned.copy_(d_a)) / 1e9,
        "h2d_pinned_gb_per_s": n / med(lambda: d_a.copy_(h_pinned)) / 1e9,
        "note": "d2d traffic counts read+write (2x buffer); HBM effective bandwidth",
    }
    del d_b, h_pageable

    # 1: weights-stage decomposition on the real checkpoint (replay.py lines ~480-501 path)
    from safetensors.torch import load_file  # noqa: E402
    st = sorted(Path(args.checkpoint).glob("*.safetensors"))
    assert st, f"no safetensors under {args.checkpoint}"
    tensors = {}
    for f in st:
        tensors.update(load_file(str(f), device="cuda:0"))
    per, tot = [], {"d2h_s": 0.0, "tobytes_s": 0.0, "sha256_s": 0.0, "leaf_hash_s": 0.0,
                    "write_s": 0.0, "bytes": 0}
    with tempfile.NamedTemporaryFile(dir="/tmp", delete=True) as tmp:
        for name, p in sorted(tensors.items()):
            sync()
            t0 = time.perf_counter()
            data = p.detach().contiguous().cpu().reshape(-1)  # the production D2H
            sync()
            t1 = time.perf_counter()
            raw = data.view(torch.uint8).numpy().tobytes()  # the production host copy
            t2 = time.perf_counter()
            hashlib.sha256(raw).hexdigest()  # WeightsWriter.add computes sha256 ...
            t3 = time.perf_counter()
            hashing.leaf_hash(f"w={name}", str(p.dtype).replace("torch.", ""), tuple(p.shape), raw)  # ... and the leaf
            t4 = time.perf_counter()
            tmp.write(raw)  # values.bin write
            t5 = time.perf_counter()
            per.append({"param": name, "bytes": len(raw), "d2h_s": t1 - t0, "tobytes_s": t2 - t1,
                        "sha256_s": t3 - t2, "leaf_hash_s": t4 - t3, "write_s": t5 - t4})
            tot["d2h_s"] += t1 - t0
            tot["tobytes_s"] += t2 - t1
            tot["sha256_s"] += t3 - t2
            tot["leaf_hash_s"] += t4 - t3
            tot["write_s"] += t5 - t4
            tot["bytes"] += len(raw)
        tmp.flush()
    tot["tensors"] = len(per)
    tot["total_s"] = sum(tot[k] for k in ("d2h_s", "tobytes_s", "sha256_s", "leaf_hash_s", "write_s"))
    doc["weights_stage"] = {"total": tot, "per_tensor_top": sorted(per, key=lambda r: -r["bytes"])[:8],
                            "note": "single pass in named order, per-phase perf_counter; "
                                    "matches replay.py weights loop (pageable D2H per tensor)"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"bandwidth": doc["bandwidth"], "weights_total": tot}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
