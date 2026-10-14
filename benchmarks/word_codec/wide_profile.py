"""The fused GEMM under Nsight Compute: one library, one linear, one batch and k (`wide_run.sh` with PROFILE=1 runs it as
`ncu --profile-from-start off`). The store is built and encoded as `wide_bench.py` does, and the fused kernel's
partial sums are checked equal to the payload store's, with and without the tag, before anything is profiled; then
each arm (ww0, t0, d<k>, t<k>, t<k>_ahead with the L2 prefetch) is launched once between cudaProfilerStart and cudaProfilerStop. Nothing is
timed here: ncu's replays make its durations a profile's, not a measurement of record. `--l2 LAYERS` profiles
instead a pass that commits its sketch tags as it goes (`wide_run.sh` with L2PROBE=LAYERS, for DRAM bytes and L2 hits).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as B  # noqa: E402
import wide as W  # noqa: E402
import wide_bench as WB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True, help="NAME=PATH of a wide.cu build")
    ap.add_argument("--shape", default="qkv")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(WB.MODELS), default=WB.MODEL)
    ap.add_argument("--l2", type=int, metavar="LAYERS",
                    help="instead: a pass of LAYERS layers of the four linears that commits its sketch tags as it goes "
                    "(`pass_overlap`'s third arm, gated first), for the tags' DRAM bytes under ncu --cache-control none")
    a = ap.parse_args()

    import torch
    lib = WB.WLib(*a.lib.split("=", 1))
    WB.use_model(a.model)
    if a.l2:
        res = WB.pass_overlap(lib, torch, {name: {} for name in B.SHAPES}, a.k, [a.batch], 0,
                              lambda s: print(s, flush=True), sketch=True, layers=a.l2, profile=True)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "l2probe.json").write_text(json.dumps({"lib": lib.name, "model": a.model, **res}, indent=1))
        return 0 if res["root_equal"] and not res["profiled"]["rc"] else 5
    f, k, mb = lib.field, a.k, a.batch
    n, kd = B.SHAPES[a.shape]
    w = B.fp8_weights(torch, n, kd, 100 + n + kd)
    pay = WB.payload_store(torch, w.view(torch.uint8), f)
    nwords = pay.numel() // f.limbs
    blocks = nwords // lib.lane_words // lib.tb
    lib.set_key(W.key(f, WB.SEED, k))
    store = torch.empty_like(pay)
    lib.call("ww_encode", WB.ptr(pay), WB.ptr(store), nwords, 0, k, 188 * 4, WB.stream(torch))
    ws = torch.empty(lib.so.ww_workspace_bytes(blocks), dtype=torch.uint8, device="cuda")
    tags_p = lib.so.ww_workspace_tags(WB.ptr(ws), blocks)
    seed = ctypes.create_string_buffer(WB.TAG_SEED, 32)
    x = torch.randn(mb, kd, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)).to(
        torch.float8_e4m3fn)
    xp = WB.ptr(x.view(torch.uint8))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    nch = kd // W.TASK_COLS

    def gemm(kk, src, y, partials, tag, prefetch=False):
        ah = lib.so.ww_gemm_occupancy(kk, mb, int(tag), int(partials)) * sms if prefetch else 0
        lib.call("ww_gemm", kk, WB.ptr(src), xp, WB.ptr(y), mb, n, kd, 0, int(partials), seed if tag else None,
                 tags_p if tag else None, ah, WB.stream(torch))

    want = torch.full((nch * mb * n,), float("nan"), dtype=torch.float32, device="cuda")
    gemm(0, pay, want, True, False)
    gate = {}
    for tag in (False, True):
        got = torch.full_like(want, float("nan"))
        gemm(k, store, got, True, tag)
        gate[f"decode{'_tag' if tag else ''}_equal"] = bool(torch.equal(got, want))
    ok = all(gate.values())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile-gate.json").write_text(json.dumps({"lib": lib.name, "shape": a.shape, "batch": mb, "k": k,
                                                       "gate": gate, "ok": ok}, indent=1))
    print(f"profile gate {lib.name} {a.shape} M={mb} k={k}: {'ok' if ok else 'FAIL'} {gate}", flush=True)
    if not ok:
        return 5
    y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
    arms = {"ww0": (0, pay, False, False), "t0": (0, pay, True, False), f"d{k}": (k, store, False, False),
            f"t{k}": (k, store, True, False), f"t{k}_ahead": (k, store, True, True)}
    for kk, src, tag, pf in arms.values():
        gemm(kk, src, y, False, tag, pf)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for kk, src, tag, pf in arms.values():
        gemm(kk, src, y, False, tag, pf)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("profiled, in order: " + ", ".join(arms), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
