#!/usr/bin/env python3
"""The vendor FP8 baseline beside run.py's own: cuBLASLt through torch._scaled_mm at each of SHAPES (a size s for
m = n = k = s, or MxNxK), E4M3 x E4M3 -> BF16, tensor-wise and row-wise scales, with promotion (use_fast_accum=False) and
without; the median of REPS launches after a warm-up, one JSON object per line.  A configuration torch refuses is written
with its error.

  python3 vendor.py      (env: SHAPES="8192 16384 2048x4608x3584", REPS=5)
"""

import json
import os
import statistics

import torch


def main() -> None:
    reps = int(os.environ.get("REPS", "5"))
    for spec in os.environ.get("SHAPES", "8192 16384").split():
        m, n, k = (int(x) for x in spec.split("x")) if "x" in spec else (int(spec),) * 3
        a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        for scales in ("tensor", "row"):
            if scales == "tensor":
                sa = sb = torch.tensor(1.0, device="cuda")
            else:
                sa = torch.ones(m, 1, device="cuda", dtype=torch.float32)
                sb = torch.ones(1, n, device="cuda", dtype=torch.float32)
            for fast in (False, True):
                row = {"kind": "vendor", "backend": "torch._scaled_mm (cuBLASLt)", "torch": torch.__version__,
                       "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "shape": [m, n, k],
                       "scales": scales, "fast_accum": fast}
                try:
                    def f():
                        return torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=fast)
                    f()
                    torch.cuda.synchronize()
                    runs = []
                    for _ in range(reps):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        f()
                        e1.record()
                        e1.synchronize()
                        runs.append(e0.elapsed_time(e1))
                    ms = statistics.median(runs)
                    row.update({"ms": round(ms, 4), "runs": [round(x, 4) for x in runs], "tflops": round(2 * m * n * k / ms / 1e9, 1)})
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)
                print(json.dumps(row), flush=True)
        del a, b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
