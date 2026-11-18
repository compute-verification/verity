"""`verity bench codec_tensor --lib PATH [--lib PATH ...] --out DIR [ARGS]`: the wide-word codec's benchmark on one
GPU, `wt.cu`'s libraries (`--lib PATH`, one per field) gated against `wide.py`, then timed. Writes `codec-tensor.json`
and `bench.log` under `--out`, and its numbers and its gates' validation as the run's `result.json` through the
harness's kit (`kit.writer`): under a research run `$RESEARCH_RUN_DIR/result.json`, outside one `DIR/result.json`. The
code it times is `bench.py`'s.

Nothing is timed before its gate passed, in this order:
- `gate`: per field and k, device encode equals `wide.encode_word` on random and edge payloads, and both decodes (CUDA
  cores, tensor cores) give the payloads back; the tensor-core squaring equals a^2 mod p on random and edge inputs; the
  plain fragment layout built on the device equals `codec.plain_fragments` (`benchmarks/word_codec`);
- `read`: the device read rate R over a store of `--store-gib`, the best of a few grids;
- `rates`: compute-only throughputs: the squaring on CUDA cores (ILP 1, 2) and on tensor cores (1 or 2 words a warp),
  both at once in one CTA (`wt_mix`), decode and encode chains on CUDA cores;
- `decode`: a store of `--store-gib` encoded on the device, its decode on CUDA and tensor cores equal to the payloads,
  then each decode timed (read, decode, XOR);
- `gemm`: at a Llama-3.1 model's linears (`--model`, 70B or 8B) and batch 1-32, the fused kernel's partial sums equal
  the plain kernel's bit for bit, the plain kernel's within FP32 rounding of torch's; then interleaved reps (each on cold
  weights, a ring of copies past the L2, queued behind a spin kernel) of cuBLASLt FP8 (`torch._scaled_mm`), the plain
  FP8 kernel, the plain kernel with `--mma-knobs` extra int8 MMAs a tile or `--imad-knobs` extra IMADs a tile a thread
  (the budget a decode on tensor or CUDA cores has before it slows the memory-bound step), and the fused decode at k = 1
  (and `--gemm-ks`); per batch, the layer's time over native with each linear at its best arm. With `--sketch` (a
  WT_RECEIPT build at full slot), the fused decode also runs with two sketch warps of its own (T = -2) that sketch
  each of the store's 4 KiB blocks at k_s = 16 or 64 and hash the sketch into its sketch-sha512 receipt
  (`sketch_receipt`), gated on every vector's sketch (a float32 GPU reference with TF32 off, exact), every receipt,
  and the commit's leaves and root (reader.cu against the driver's tree). Before its GEMMs, `gate_pass` checks the
  per-pass key (red team F29: `wt_pass` derives pass j's seed, signs and receipt midstate on the device from
  root_(j−1)), and once a linear one such launch runs under a pass's key and is gated at that pass's seed
  (`pass_keyed`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import bench as B
from verity_bench import kit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="a wt.cu build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="rates,decode,gemm")
    ap.add_argument("--store-gib", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gate-words", type=int, default=256)
    ap.add_argument("--gate-reps", type=int, default=1,
                    help="runs of each fused gate; above 1, every batch is gated and a failure logs where it differs")
    ap.add_argument("--model", default="70b", choices=sorted(B.MODELS))
    ap.add_argument("--shapes", default="", help="the model's linears to run (default: all)")
    ap.add_argument("--batches", default=",".join(map(str, B.BATCHES)))
    ap.add_argument("--gemm-libs", default="", help="field names the GEMM runs with (default: all)")
    ap.add_argument("--gemm-ks", default="1")
    ap.add_argument("--plain", default="4x4,8x8", help="plain kernel CONSUMERSxSTAGES")
    ap.add_argument("--fused", default="auto",
                    help="fused kernel DECODERSxSTAGES[x-2xK_S], or auto (fused_cfgs)")
    ap.add_argument("--dump", action="store_true",
                    help="after a failed fused gate, rerun it (until it fails, at most 4 x --gate-reps) dumping the MMA "
                         "warp's operands (WT_DEBUG_DUMP build)")
    ap.add_argument("--sketch", default="", help="k_s values (16, 64) for the sketch warps' sketch-sha512 receipts (a "
                                                 "WT_RECEIPT build at full slot)")
    ap.add_argument("--sketch-t", type=int, default=8, help="tasks a sketch tile (n = 32 ceil(SB / 512) T bytes)")
    ap.add_argument("--mma-knobs", default="8,16,32,48,64,96")
    ap.add_argument("--imad-knobs", default="32,64,128,192,256")
    ap.add_argument("--reader-lib", default="", help="reader.cu, the commit of a WT_RECEIPT=1 build (its gemm gate)")
    a = ap.parse_args()

    import torch
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    def save() -> None:
        (out_dir / "codec-tensor.json").write_text(json.dumps(res, indent=1))

    libs = [B.Lib(p) for p in a.lib]
    if a.reader_lib:
        import reader as RD
        B.READER = RD.Reader(a.reader_lib, torch)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    phases = set(a.phases.split(","))
    res = {"device": {"name": props.name, "sms": sms, "cc": f"{props.major}.{props.minor}"}, "seed": B.SEED.hex(),
           "libs": {}, "ok": True, "args": vars(a)}
    log(f"device {props.name}, {sms} SMs; fields {[lib.name for lib in libs]}")
    res["layout"] = B.gate_layout(torch, log)
    res["ok"] &= res["layout"]["ok"]
    for lib in libs:
        g = {k: B.gate_codec(lib, torch, k, a.gate_words, log) for k in range(1, lib.kmax + 1)}
        e = {"path": lib.path, "constants": {"n": lib.n, "NL": lib.nl, "S": lib.s, "PB": lib.pb, "TW": lib.tw,
                                             "U": lib.u, "c": lib.c, "tc_mmas": lib.tc_mmas},
             "sha256": hashlib.sha256(Path(lib.path).read_bytes()).hexdigest(), "gate": g}
        if lib.tc:
            e["gate_tc_square"] = B.gate_tc_square(lib, torch, 512, log)
        res["libs"][lib.name] = e
        res["ok"] &= all(v["ok"] for v in g.values()) and e.get("gate_tc_square", {"ok": True})["ok"]
    save()
    if not res["ok"]:
        log("a gate failed: nothing timed")
        write_result(res, libs, out_dir)
        return 5
    store = torch.randint(0, 255, (int(a.store_gib * (1 << 30)) // 16 * 16,), dtype=torch.uint8, device="cuda")
    res["read"] = B.read_rate(libs[0], torch, store, sms, a.reps)
    del store
    torch.cuda.empty_cache()
    log(f"read {res['read']['gbps']:.0f} GB/s")
    for lib in libs:
        e = res["libs"][lib.name]
        B.set_key(lib, torch, B.lib_key(lib, 2))
        if "rates" in phases:
            e["rates"] = B.rates(lib, torch, sms, a.reps, log)
            e["analysis"] = B.analysis(lib, res["read"]["gbps"], e["rates"])
            for k, row in e["analysis"]["k"].items():
                log(f"analysis {lib.name} k={k}: " + ", ".join(f"{kk} {v:.3g}" for kk, v in row.items()))
            save()
        if "decode" in phases:
            e["decode"] = B.standalone_decode(lib, torch, int(a.store_gib * (1 << 30)), sms, a.reps, log)
            res["ok"] &= all(v["ok"] for v in e["decode"].values())
            save()
    if "pipes" in phases:
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        res["pipes"] = {}
        for op, name in enumerate(B.PIPE_OPS):
            r = max((dict(B.calibrated(torch, lambda s, op=op, m=m: libs[0].call("wt_chain", 5, 1, op, s, B.ptr(xr), sms * m,
                                                                   B.stream(torch)), sms * m * 128 * 8, 0.08, a.reps),
                          grid_per_sm=m) for m in (4, 8, 16)), key=lambda v: v["per_s"])
            r["clock_hz"] = clock_hz = B.sm_clock(libs[0], torch)
            r["lanes_per_sm_clock"] = r["per_s"] / (sms * clock_hz)
            res["pipes"][name] = r
            log(f"pipe {name}: {r['per_s']:.3e}/s, {r['lanes_per_sm_clock']:.1f} lanes an SM a clock "
                f"(at {clock_hz / 1e6:.0f} MHz)")
        save()
    if "gemm" in phases:
        names = a.gemm_libs.split(",") if a.gemm_libs else [lib.name for lib in libs]
        ks = [int(k) for k in a.gemm_ks.split(",")]
        batches = [int(b) for b in a.batches.split(",")]
        knobs = ([int(v) for v in a.mma_knobs.split(",") if v], [int(v) for v in a.imad_knobs.split(",") if v])
        layers, shapes = B.MODELS[a.model]
        res["model"] = {"name": a.model, "layers": layers, "shapes": shapes}
        for lib in [x for x in libs if x.name in names]:
            e = res["libs"][lib.name]
            e["gemm"] = {}
            if any(B.sketch_warps(c) for c in B.fused_cfgs(lib, a.fused, a.sketch, a.sketch_t)):
                e["gate_pass"] = B.gate_pass(lib, torch, a.sketch_t, log)
                res["ok"] &= e["gate_pass"]["ok"]
                if not e["gate_pass"]["ok"]:
                    continue
            for name in a.shapes.split(",") if a.shapes else shapes:
                n, kd = shapes[name]
                g = B.gemm_case(lib, torch, name, n, kd, batches, ks, a.reps, log, B.parse(a.plain),
                              B.fused_cfgs(lib, a.fused, a.sketch, a.sketch_t), *knobs, gate_reps=a.gate_reps,
                              dump=a.dump, sketch_t=a.sketch_t)
                e["gemm"][name] = g
                res["ok"] &= g.get("ok", False)
                save()
            if not all(g.get("ok") for g in e["gemm"].values()):
                continue
            e["layer"] = B.layer_ratios(e["gemm"], batches, ks)
            for mb, row in e["layer"].items():
                log(f"layer {lib.name} M={mb} ({'+'.join(e['gemm'])}): "
                    + ", ".join(f"{kk} {v:.3f}" for kk, v in row.items()))
            knobs = ([], [])
    save()
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


def write_result(res: dict, libs: list[B.Lib], out_dir: Path) -> None:
    w = kit.writer(out_dir)
    if "read" in res:
        w.measurement("read", res["read"]["gbps"], "GB/s", mark="measured")
    for name, e in res["libs"].items():
        for unit, v in e.get("analysis", {}).get("squarings_s", {}).items():
            w.measurement(f"{name}_squarings_{unit}", v, "1/s", mark="measured")
        for k, row in e.get("analysis", {}).get("k", {}).items():
            for kk, v in row.items():
                w.measurement(f"{name}_k{k}_{kk}", v, "1", mark="measured")
        for k, d in e.get("decode", {}).items():
            for unit in ("cuda", "tc"):
                if d.get(unit, {}).get("ok"):
                    w.measurement(f"{name}_decode_k{k}_{unit}", d[unit]["gbps_stored"], "GB/s", mark="measured")
        for shape, g in e.get("gemm", {}).items():
            for mb, t in g.get("times", {}).items():
                for arm, ratio in t["ratio_vs_native"].items():
                    if arm != "empty":
                        w.measurement(f"{name}_{shape}_m{mb}_{arm}_vs_native", ratio, "1", mark="measured")
    rel = os.path.relpath(out_dir, w.out)
    files = [f"{rel}/codec-tensor.json", f"{rel}/bench.log"]
    w.fingerprint(libs={lib.name: hashlib.sha256(Path(lib.path).read_bytes()).hexdigest() for lib in libs},
                  seed=B.SEED.hex())
    for f in files:
        w.measurement_file(f)
    w.validation("passed" if res["ok"] else "failed", "codec-tensor gates",
                 "device codec = wide.py, TC square = a^2 mod p, fused partial sums = plain", files)


if __name__ == "__main__":
    sys.exit(main())
