"""`kernels.py`'s linears on Llama-3.1-8B's shapes without vLLM: the gates `serve.py` runs inside `load_model` (every GEMM
form's chunk sums equal plain-simple's bit for bit, encoded against plain; the plain sums against a float product), one
launch at a time with a line before each (``--check``), and each linear's time per call (``--time``): vLLM's CUTLASS FP8
GEMM with its per-token activation quantization (the stock arm's), and the fused GEMM over plain fragments and over codec
words with the same quantization and the in-order reduce, every form, over enough copies of the layer that the weights
stream from HBM.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels as K
import torch

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
SEED = bytes(range(32))
L2_BYTES = 128 << 20


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def rand_fp8(shape, gen) -> torch.Tensor:
    return (torch.randn(shape, generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def check(lib: K.Lib, ks: list[int], batches: list[int], shapes: dict) -> dict:
    gen = torch.Generator(device="cuda")
    out = {"configs": 0, "max_rel_err": 0.0}
    for name, (n, kd) in shapes.items():
        gen.manual_seed(n * 7 + kd)
        w = rand_fp8((n, kd), gen)
        ws = torch.ones(1, device="cuda")
        plain = K.PlainLinear(lib, w, ws)
        encs = {}
        for k in ks:
            if k:
                lib.set_key(SEED, k)
                encs[k] = K.EncodedLinear(lib, w, ws, k, 12345)
        torch.cuda.synchronize()
        log(name, "built", {k: e.nbytes for k, e in encs.items()})
        nch = kd // 512
        for m in batches:
            x8 = rand_fp8((m, kd), gen)
            log(name, m, "ref simple k=0")
            ref = plain.gemm(x8, K.Config(0))
            torch.cuda.synchronize()
            exact = x8.float() @ w.float().t()
            err = (ref.sum(0) - exact).abs().max().item() / exact.abs().max().item()
            out["max_rel_err"] = max(out["max_rel_err"], err)
            for k in ks:
                if k:
                    lib.set_key(SEED, k)
                op = encs.get(k, plain)
                for cfg in K.candidates(lib, k, m, nch):
                    log(name, m, f"k={k}", cfg)
                    got = op.gemm(x8, cfg)
                    torch.cuda.synchronize()
                    if not torch.equal(got.view(torch.int32), ref.view(torch.int32)):
                        raise SystemExit(f"{name} M={m} k={k} {cfg}: sums differ")
                    out["configs"] += 1
        del plain, encs
    return out


def time_linears(lib: K.Lib, ks: list[int], batches: list[int], shapes: dict, reps: int) -> list[dict]:
    from vllm import _custom_ops as ops
    gen = torch.Generator(device="cuda")
    rows = []
    for name, (n, kd) in shapes.items():
        copies = max(2, math.ceil(2 * L2_BYTES / (n * kd)))
        gen.manual_seed(n + kd)
        ws_ = [rand_fp8((n, kd), gen) for _ in range(copies)]
        scale = torch.full((1,), 0.01, device="cuda")
        ops_ = {0: [K.PlainLinear(lib, w, scale) for w in ws_]}
        for k in ks:
            if k:
                lib.set_key(SEED, k)
                ops_[k] = [K.EncodedLinear(lib, w, scale, k, i * (1 << 32)) for i, w in enumerate(ws_)]
                for e in ops_[k]:
                    e.drop_payloads()
        nch = kd // 512
        for m in batches:
            x = torch.randn((m, kd), generator=gen, device="cuda").to(torch.bfloat16)

            def stock():
                for w in ws_:
                    x8, xs = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
                    ops.cutlass_scaled_mm(x8, w.t(), xs, scale, torch.bfloat16)

            row = {"shape": name, "n": n, "k": kd, "m": m, "copies": copies, "stock_us": timed(lib, stock, reps) / copies}
            for k, lins in ops_.items():
                if k:
                    lib.set_key(SEED, k)
                best = None
                for cfg in K.candidates(lib, k, m, nch):
                    def fused(cfg=cfg, lins=lins):
                        for lin in lins:
                            x8, xs = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
                            lin.reduce(lin.gemm(x8, cfg), xs.reshape(-1))
                    t = timed(lib, fused, reps) / copies
                    row[f"k{k}_{cfg.name}_us"] = round(t, 2)
                    if best is None or t < best[1]:
                        best = (cfg.name, t)
                row[f"k{k}_best"], row[f"k{k}_us"] = best
            for k in ops_:
                row[f"k{k}_over_stock"] = row[f"k{k}_us"] / row["stock_us"]
                if k:
                    row[f"k{k}_over_plain"] = row[f"k{k}_us"] / row["k0_us"]
            log(json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()
                            if a.count("_") <= 1 or a.endswith(("_best", "_over_stock", "_over_plain"))}))
            rows.append(row)
        del ws_, ops_
        torch.cuda.empty_cache()
    return rows


def timed(lib: K.Lib, fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("wc_spin", 40_000_000, K.stream())
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ks", default="0,2,3")
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lib = K.Lib(args.lib)
    ks = [int(x) for x in args.ks.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    shapes = {s: SHAPES[s] for s in args.shapes.split(",")}
    res = {"lib": Path(args.lib).name, "bits": lib.bits, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name()}
    if args.check:
        res["check"] = check(lib, ks, batches, shapes)
        log("check", res["check"])
    if args.time:
        res["time"] = time_linears(lib, ks, batches, shapes, args.reps)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
