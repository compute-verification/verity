"""A model's linears at decode and prefill batch sizes in BF16, FP8 and int8 on one GPU: every layer's four linears with
their own weights (so L2 holds none of them between uses), captured once as a CUDA graph a dtype and M and replayed.
BF16 is cuBLAS (`torch.mm`, vLLM's unquantized path), FP8 is vLLM's CUTLASS kernel (`cutlass_scaled_mm`, per-tensor
scales, what `serve.py`'s engine runs), and int8 is cuBLASLt (`torch._int_mm`, s8 × s8 → s32, exact; M is padded to
32 below 17, which it needs) and vLLM's CUTLASS int8 where the build has it. Each time is the median replay of
`--reps`; a dtype's rate is its weight bytes over that time against `--R`, and its MAC rate. Writes `gemm_dtypes.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MODELS = {
    # (N, K) of qkv, o, gate_up, down per GPU, and layers
    "8b": ([(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)], 32),
    "70b-tp8": ([(1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)], 80),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=lambda s: s.split(","), default=list(MODELS))
    ap.add_argument("--ms", type=lambda s: [int(v) for v in s.split(",")], default=[1, 8, 32, 128, 512, 2048, 8192])
    ap.add_argument("--dtypes", type=lambda s: s.split(","), default=["bf16", "fp8", "int8", "int8-cutlass"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--R", type=float, default=1539.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    dev = torch.device("cuda")
    res = {"format": "verity-serve-bandwidth/gemm-dtypes/v0", "gpu": torch.cuda.get_device_name(), "R_GBps": a.R,
           "torch": torch.__version__, "rows": []}
    one = torch.ones(1, device=dev, dtype=torch.float32)

    def weights(dt: str, n: int, k: int):
        if dt == "bf16":
            return torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        if dt == "fp8":
            return (torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn).t()  # K × N column-major
        w = torch.randint(-127, 128, (n, k), device=dev, dtype=torch.int8)
        return w.t() if dt == "int8-cutlass" else w.t()

    def act(dt: str, m: int, k: int):
        if dt == "bf16":
            return torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        if dt == "fp8":
            return (torch.randn(m, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return torch.randint(-127, 128, (max(m, 17) if dt == "int8" else m, k), device=dev, dtype=torch.int8)

    def mm(dt: str, x, w):
        if dt == "bf16":
            return torch.mm(x, w.t())
        if dt == "fp8":
            return ops.cutlass_scaled_mm(x, w, one, one, torch.bfloat16)
        if dt == "int8-cutlass":
            return ops.cutlass_scaled_mm(x, w, one, one, torch.bfloat16)
        return torch._int_mm(x, w)

    for model in a.models:
        shapes, layers = MODELS[model]
        for dt in a.dtypes:
            try:
                ws = [[weights(dt, n, k) for n, k in shapes] for _ in range(layers)]
            except Exception as e:  # noqa: BLE001
                res["rows"].append({"model": model, "dtype": dt, "error": repr(e)[:300]})
                continue
            wbytes = sum(w.numel() * w.element_size() for lw in ws for w in lw)
            for m in a.ms:
                row = {"model": model, "dtype": dt, "M": m, "weight_GB": round(wbytes / 1e9, 3)}
                try:
                    xs = [act(dt, m, k) for _, k in shapes]
                    mm(dt, xs[0], ws[0][0])
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for lw in ws:
                            for x, w in zip(xs, lw):
                                mm(dt, x, w)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(g, stream=s):
                            for lw in ws:
                                for x, w in zip(xs, lw):
                                    mm(dt, x, w)
                    g.replay()
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(a.reps):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        g.replay()
                        e1.record()
                        e1.synchronize()
                        ts.append(e0.elapsed_time(e1))
                    ms_ = statistics.median(ts)
                    macs = layers * sum(m * n * k for n, k in shapes)
                    row.update({"ms": round(ms_, 4), "weight_of_R": round(wbytes / (ms_ * 1e-3) / (a.R * 1e9), 4),
                                "TMACs": round(macs / (ms_ * 1e-3) / 1e12, 2),
                                "M_run": m if dt != "int8" else max(m, 17)})
                    del g, xs
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)[:300]
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
            del ws
            torch.cuda.empty_cache()
    base = {(r["model"], r["M"]): r["ms"] for r in res["rows"] if r.get("dtype") == "fp8" and "ms" in r}
    for r in res["rows"]:
        if "ms" in r and (r["model"], r["M"]) in base:
            r["over_fp8"] = round(r["ms"] / base[r["model"], r["M"]], 4)
    (a.out / "gemm_dtypes.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
