"""Where the in-place pass's time goes, on one build of session.cu over the A/B's store (Llama-3.1-8B bf16 as vLLM holds
it, `inplace_gate.llama8b`): the bare read over a contiguous copy (`rb_read`), the read through the map
(`rs_mapped_read`), the receipts alone over the copy (`rs_sketch16_receipts`) and through the map (`rs_mapped_receipts`),
the tree hashed after them by reader.cu's kernels (`InPlace.tree`) and the fused pass (`InPlace.fused`), each `reps`
times. The receipts alone against the read say how much of the sketch is the read; mapped against contiguous, how much
is the store's layout. Then one launch of each of --profiled (default every part) between cudaProfilerStart and Stop,
for `ncu --profile-from-start off`: ncu serializes the fused pass's two kernels, so it times the receipts' producer
alone and then the consumer alone, the leaves and tree over finished receipts. Writes OUT/profile_pass.json.

    python profile_pass.py --lib session.so --out OUT [--reps 10] [--no-timing] [--profiled read_mapped,tree]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ab_pass import samples  # noqa: E402
from device import BLOCK, Lib, rb  # noqa: E402
from inplace import InPlace, Region  # noqa: E402
from inplace_gate import llama8b, register  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--profiled", default="read_contiguous,read_mapped,receipts_contiguous,receipts_mapped,tree,kernels")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(1 << 62)))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    lib = Lib(a.lib, torch)
    weights = llama8b(torch, dev, gen)
    store, backs = register(torch, weights, [], 0, 0)
    regions = [Region.of_backing(r.first, b) for r, b in zip(store.layout().regions, backs)]
    N, key = store.layout().blocks, rng.bytes(32)
    ip = InPlace(torch, lib, regions, key)
    nbytes = N * BLOCK
    copy = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    for r, b in zip(regions, backs):
        copy[r.first * BLOCK:(r.first + r.blocks) * BLOCK].copy_(b.flat[:r.blocks * BLOCK])
    xr = torch.zeros(1, dtype=torch.int64, device=dev)
    seed, plan = rng.bytes(64), rb.Plan.whole(N)
    addrs, js = ip.map(plan)
    pk = ip.set_pass(seed, plan)
    ys = torch.empty((N, 8), dtype=torch.int64, device=dev)
    leaves = rb.leaves(N)
    width = 1 << (leaves - 1).bit_length()
    parts = {
        "read_contiguous": lambda: lib.call("rb_read", copy.data_ptr(), nbytes, xr.data_ptr(), 16 * sms),
        "read_mapped": lambda: lib.call("rs_mapped_read", addrs.data_ptr(), N, xr.data_ptr(), 16 * sms),
        "receipts_contiguous": lambda: lib.call("rs_sketch16_receipts", copy.data_ptr(), 0, N, pk.data_ptr(),
                                                ys.data_ptr(), 0),
        "receipts_mapped": lambda: lib.call("rs_mapped_receipts", addrs.data_ptr(), js.data_ptr(), N, pk.data_ptr(),
                                            ys.data_ptr(), 0),
        "tree": lambda: ip.tree(ys, 0, leaves, width),
        "kernels": lambda: ip.fused(addrs, js, pk, N),
    }
    out = {"device": torch.cuda.get_device_name(dev), "sms": sms, "blocks": N, "bytes": nbytes}
    if not a.no_timing:
        ms = {k: samples(torch, f, a.reps) for k, f in parts.items()}
        med = {k: statistics.median(v) for k, v in ms.items()}
        out["samples_ms"], out["median_ms"] = ms, med
        out["GBps"] = {k: nbytes / (med[k] / 1e3) / 1e9 for k in ("read_contiguous", "read_mapped",
                                                                    "receipts_contiguous", "receipts_mapped",
                                                                    "kernels")}
        out["over_read_contiguous"] = {k: med[k] / med["read_contiguous"] for k in med}
        print(json.dumps({k: out[k] for k in ("median_ms", "GBps", "over_read_contiguous")}), flush=True)
    profiled = [parts[k] for k in a.profiled.split(",")]
    for f in profiled:
        f()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for f in profiled:
        f()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    (a.out / "profile_pass.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
