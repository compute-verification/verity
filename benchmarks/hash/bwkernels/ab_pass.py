"""The in-place pass (`rs_mapped_pass`, `InPlace.run_pass`) of two builds of session.cu, interleaved on one GPU over one
store: Llama-3.1-8B's weights as vLLM holds them in bf16 (`inplace_gate.llama8b`, 195 tensors, 16.06 GB), registered as
the binder registers them. First the builds are checked against each other at that scale: every receipt, every kept
level and the root under one seed, and the root against the tree hashed after the receipts by reader.cu's own kernels
(`inplace_gate.unfused_pass`). Then `rounds` rounds, each build in turn (the order alternating a round), each timing
`reps` runs of the bare read over a contiguous copy of the store (`rb_read`, γ's denominator), the read through the map
(`rs_mapped_read`), the fused kernels alone (`InPlace.fused`) and the whole pass (map, constants, signs, kernels,
root). Writes OUT/ab_pass.json: every sample, and per build the medians, γ = pass / bare read, and the after/before
ratio of each round's medians.

    python ab_pass.py --before before.so --after after.so --out OUT [--rounds 12 --reps 5 --seed 0]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

RS = Path(__file__).resolve().parents[2] / "remote_session"
sys.path.insert(0, str(RS))
sys.path.insert(0, str(RS.parents[1] / "integrations" / "vllm"))

from device import BLOCK, Lib, digests, rb  # noqa: E402
from inplace import InPlace, Region  # noqa: E402
from inplace_gate import llama8b, register, unfused_pass  # noqa: E402

BUILDS = ("before", "after")
PARTS = ("read_contiguous", "read_mapped", "kernels", "pass")


def samples(torch, fn, reps: int) -> list[float]:
    """fn's GPU time in ms, `reps` runs after one untimed, each between two events on the current stream."""
    fn()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return out


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng.integers(1 << 62)))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    libs = {"before": Lib(a.before, torch), "after": Lib(a.after, torch)}
    weights = llama8b(torch, dev, gen)
    store, backs = register(torch, weights, [], 0, 0)
    regions = [Region.of_backing(r.first, b) for r, b in zip(store.layout().regions, backs)]
    N, key = store.layout().blocks, rng.bytes(32)
    ips = {k: InPlace(torch, lib, regions, key) for k, lib in libs.items()}
    nbytes = N * BLOCK
    copy = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    for r, b in zip(regions, backs):
        copy[r.first * BLOCK:(r.first + r.blocks) * BLOCK].copy_(b.flat[:r.blocks * BLOCK])
    xr = torch.zeros(1, dtype=torch.int64, device=dev)
    seed, plan = rng.bytes(64), rb.Plan.whole(N)
    out = {"device": torch.cuda.get_device_name(dev), "sms": sms, "blocks": N, "bytes": nbytes,
           "tensors": len(weights), "rounds": a.rounds, "reps": a.reps}

    ys = {k: ip.pass_receipts(seed, plan) for k, ip in ips.items()}
    passes = {k: ip.run_pass(seed, plan) for k, ip in ips.items()}
    unfused = unfused_pass(ips["before"], seed, plan)
    out["equal"] = {
        "receipts": bool(torch.equal(ys["before"], ys["after"])),
        "kept": bool(torch.equal(passes["before"][1].kept, passes["after"][1].kept)),
        "root": passes["before"][0] == passes["after"][0],
        "root_unfused": passes["after"][0] == unfused,
    }
    sample = sorted(map(int, rng.integers(0, N, 32)))
    r = rb.receipter(rb.SKETCH_SHA512, seed, rb.SKETCH_ROWS)
    host = ys["after"].cpu()
    out["equal"]["sampled_receipts_reference"] = all(
        digests(host[p:p + 1])[0] == r(p, copy[p * BLOCK:(p + 1) * BLOCK].cpu().numpy().tobytes()) for p in sample)
    out["root"] = passes["after"][0].hex()
    print(json.dumps({"equal": out["equal"]}), flush=True)
    del ys, passes, host
    if not all(out["equal"].values()):
        (a.out / "ab_pass.json").write_text(json.dumps(out, indent=1))
        return 1

    mapped = {k: ip.map(plan) for k, ip in ips.items()}
    pk = {k: ip.set_pass(seed, plan) for k, ip in ips.items()}

    def parts(k: str):
        lib, ip, (addrs, js) = libs[k], ips[k], mapped[k]
        return {"read_contiguous": lambda: lib.call("rb_read", copy.data_ptr(), nbytes, xr.data_ptr(), 16 * sms),
                "read_mapped": lambda: lib.call("rs_mapped_read", addrs.data_ptr(), N, xr.data_ptr(), 16 * sms),
                "kernels": lambda: ip.fused(addrs, js, pk[k], N),
                "pass": lambda: ip.run_pass(seed, plan)}

    fns = {k: parts(k) for k in BUILDS}
    rows = []
    for rd in range(a.rounds):
        for k in (BUILDS if rd % 2 == 0 else BUILDS[::-1]):
            row = {"round": rd, "build": k}
            for part in PARTS:
                row[part] = samples(torch, fns[k][part], a.reps)
            rows.append(row)
        print(json.dumps({"round": rd, **{f"{x['build']}_{p}": statistics.median(x[p]) for x in rows[-2:]
                                          for p in ("read_contiguous", "kernels", "pass")}}), flush=True)
    out["samples"] = rows

    def med(k: str, part: str) -> float:
        return statistics.median(v for x in rows if x["build"] == k for v in x[part])

    out["median_ms"] = {k: {p: med(k, p) for p in PARTS} for k in BUILDS}
    out["gamma"] = {k: {"pass": med(k, "pass") / med(k, "read_contiguous"),
                        "kernels": med(k, "kernels") / med(k, "read_contiguous")} for k in BUILDS}
    out["read_GBps"] = {k: nbytes / (med(k, "read_contiguous") / 1e3) / 1e9 for k in BUILDS}
    ratios = {}
    for p in PARTS:
        per = []
        for rd in range(a.rounds):
            m = {x["build"]: statistics.median(x[p]) for x in rows if x["round"] == rd}
            per.append(m["after"] / m["before"])
        per.sort()
        ratios[p] = {"median": statistics.median(per), "min": per[0], "max": per[-1]}
    out["after_over_before"] = ratios
    (a.out / "ab_pass.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("median_ms", "gamma", "read_GBps", "after_over_before")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
