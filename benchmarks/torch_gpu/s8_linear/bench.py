"""Where the INT8 tensor cores, not DRAM, start to limit the large model's linears: each linear's time against M, for the exact
kernel and the fastest plain int8 GEMMs, interleaved rep by rep on one GPU.

Per linear (N, K) of `fixture.shapes(models)` and M, the arms (`kbench.timed`: one CUDA graph per arm over weight copies past 3x
the L2, so the weights stream from DRAM; items of about 10 ms alternating arm by arm, order rotated each rep; the SM clock read
around every item), each gated in this run before it is timed:
    exact         the exact kernel (run.py's `--kernel`: wgmma on sm_90, int8_fused elsewhere) at its picked config
                  (`configs.CONFIGS` or --configs, else its default; bf16 out, the Definition's epilogue): y into 0xA5 equal
                  to `run.reference` (run.py's gate holds that to the Definition's twin on every cell)
    exact_triton  on sm_90, int8_fused (Triton) the same way: the second exact kernel
    simple        int_mm.py's `linear` (run.py's `--kernel int_mm`): torch._int_mm's s32 sums,
                  then the fused epilogue; gated like exact: the path the exact kernel must beat
    lt            cuBLASLt int8, s32 out, autotuned: its top 32 heuristic algorithms (`lt.Plan`), each gated (z equal to
                  torch._int_mm's) and timed in a pre-pass of 3 interleaved items, the fastest kept
    cutlass       vLLM's CUTLASS int8 GEMM (`torch.ops._C.cutlass_scaled_mm`, bf16 out; its epilogue applies s_w first,
                  not the Definition's order): a plain-speed reference, its words counted against z's epilogue in that order
    cutlass_swap  the same kernel on the transposed product (y^T = W X^T, scale_b = s_a first: the Definition's order),
                  M a multiple of 16, its words counted against the Definition's
DRAM read bandwidth is measured in the run, before and after (a streaming read of 4 GiB, `read_kernel`, interleaved with a
copy), and the s8 tensor-core peak is the card's, `--peak-tops`, by default PEAKS' measured reading for the arch: sm_90's
1,942.7 TOPS (the H200's best second of dense s8 wgmma) beside its data sheet's 1,979 at 1,830 MHz,
sm_120's 807.5 (mma.sync at locked 2100 MHz). crossing.json records every reading (`peak_readings`); a
fraction of peak scales by their ratio. A model's layer set is its linears (`fixture.MODELS`) at the same M.

crossing.json, per arm and model (layer set) and per linear:
    tops, frac_peak       2 M N K / t, and over the peak
    gbs, frac_bw          (weights + X + Y + scales) / t, and over the measured bandwidth
    ridge_card            M* where 2 M W / P = bytes(M) / B: the card's roofline ridge for this shape
    ridge_arm             the same from the arm's own best achieved TOPS and bytes/s in the sweep: where its time stops being
                          flat in M and starts growing with it
    at_ridge              the arm's time, TOPS and fraction of peak at the card's ridge and at its own (log-log interpolation)

    python bench.py --out OUT [--models mqa16k52] [--ms 1,8,...] [--reps 9] [--arms exact,exact_triton,simple,lt,cutlass,cutlass_swap]
                    [--kernel wgmma|triton|int_mm] [--configs BEST_JSON] [--no-head]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import fixture as FX  # noqa: E402
import int_mm as IM  # noqa: E402
import kbench  # noqa: E402
import lt as LT  # noqa: E402
import roofline as RF  # noqa: E402
import run as R  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

SWEEP_MS = (1, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1023, 1024, 2048, 4096, 8192, 16384,
            20480)
ARMS = ("exact", "exact_triton", "simple", "lt", "cutlass", "cutlass_swap")
PEAKS = {"sm_90": {"measured": 1942.7, "datasheet_1830mhz": 1979.0}, "sm_120": {"measured": 807.5}}
OUT_BYTES = {"exact": 2, "exact_triton": 2, "simple": 2, "lt": 4, "cutlass": 2, "cutlass_swap": 2}
BW_BYTES = 4 << 30


@triton.jit
def read_kernel(src, out, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLOCK * ITERS
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc ^= tl.load(src + offs, mask=offs < n, other=0)
    tl.store(out + pid, tl.sum(acc, axis=0))


def bandwidth(clock, reps: int) -> dict:
    """DRAM bandwidth: a streaming int32 read of BW_BYTES (read_kernel) and a device copy (read + write), interleaved."""
    n = BW_BYTES // 4
    src = torch.randint(0, 1 << 30, (n,), dtype=torch.int32, device="cuda")
    dst = torch.empty_like(src)
    tile, iters = 4096, 16
    grid = triton.cdiv(n, tile * iters)
    out = torch.empty(grid, dtype=torch.int32, device="cuda")
    fns = {"read": lambda: read_kernel[(grid,)](src, out, n, BLOCK=tile, ITERS=iters, num_warps=8),
           "copy": lambda: dst.copy_(src)}
    med, us, meta = kbench.timed(fns, reps, clock, 1)
    doc = {"read_gbs": BW_BYTES / med["read"] / 1e3, "copy_gbs": 2 * BW_BYTES / med["copy"] / 1e3, "us": med,
           "sm_clock_mhz": meta["clocks"]}
    src = dst = out = None
    torch.cuda.empty_cache()
    return doc


class Linear:
    """One (N, K, M): weight copies, X, scales, the outputs, and the arms' gated callables."""

    def __init__(self, n: int, k: int, m: int, kerns: dict):
        self.n, self.k, self.m, self.kerns = n, k, m, kerns
        self.copies = kbench.copies_for(n, k)
        self.ws = [R.Weights(n, k, seed=2000 + i, special=False) for i in range(self.copies)]
        self.x = R.codes(m, k, 11 + m, FX.STRUCT_ROWS)
        self.sa = torch.from_numpy(FX.act_scales(m, 11 + m, special=False)).cuda()
        self.z_first = IM.int_mm(self.x, self.ws[0].w)
        self.ref_first = R.reference(self.z_first, self.sa, self.ws[0].sw)
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.yt = torch.empty((n, m), dtype=torch.bfloat16, device="cuda")
        self.z = torch.empty((m, n), dtype=torch.int32, device="cuda")
        self.gates: dict = {}

    def exact(self, arm: str):
        kern = self.kerns[arm]
        cfg = kern.pick(self.m, self.n, self.k)
        R.poison(self.y)
        h = kern.call(self.x, self.sa, self.ws[0].w, self.ws[0].sw, self.y, cfg)
        torch.cuda.synchronize()
        self.gates[arm] = {"kernel": kern.name, "cfg": str(cfg), "mismatches": R.diff(self.y, self.ref_first),
                           "checks_ok": kern.checks(h)["ok"]}
        if self.gates[arm]["mismatches"] or not self.gates[arm]["checks_ok"]:
            return None

        def f():
            for wt in self.ws:
                kern.call(self.x, self.sa, wt.w, wt.sw, self.y, cfg)
        return f

    def lt(self, clock) -> object:
        plan = LT.Plan(self.m, self.n, self.k)
        cands = plan.candidates()
        fns, names, gate = {}, {}, {}
        for rank, h in enumerate(cands):
            nm = LT.name(h, rank)
            try:
                self.z.view(torch.uint8).fill_(0x7F)
                plan.run(h, self.x, self.ws[0].w, self.z)
                torch.cuda.synchronize()
                d = int((self.z != self.z_first).sum())
            except Exception as ex:  # noqa: BLE001 - a candidate that fails to launch is a record
                torch.cuda.synchronize()
                gate[nm] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
                continue
            gate[nm] = {"mismatches": d, **LT.config(h)}
            if d == 0:
                fns[nm] = (lambda h=h: [plan.run(h, self.x, wt.w, self.z) for wt in self.ws])
                names[nm] = h
        self.gates["lt"] = {"candidates": len(cands), "passed": len(fns), "records": gate}
        if not fns:
            return None
        med, _, _ = kbench.timed(fns, 3, clock, self.copies)
        best = min(med, key=med.get)
        self.gates["lt"].update({"best": best, "prepass_us": med})
        self._plan = plan
        return fns[best]

    def cutlass(self, swap: bool):
        """vLLM's CUTLASS int8 GEMM (`torch.ops._C.cutlass_scaled_mm`, no fallback), timed as a speed reference whatever its
        words; the gate records them: the plain call against z's epilogue in its own order (s_w first), the swapped call
        against the Definition's words. A call that doesn't launch isn't timed."""
        key = "cutlass_swap" if swap else "cutlass"
        if swap and self.m % 16:
            self.gates[key] = {"skipped": "M is not a multiple of 16"}
            return None
        try:
            import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)

            def call(wt):
                if swap:
                    torch.ops._C.cutlass_scaled_mm(self.yt, wt.w, self.x.t(), wt.sw.reshape(-1, 1), self.sa.reshape(-1, 1), None)
                else:
                    torch.ops._C.cutlass_scaled_mm(self.y, self.x, wt.w.t(), self.sa.reshape(-1, 1), wt.sw.reshape(-1, 1), None)
            wt = self.ws[0]
            R.poison(self.yt if swap else self.y)
            call(wt)
            torch.cuda.synchronize()
            if swap:
                d = R.diff(self.yt.t().contiguous(), self.ref_first)
            else:
                rev = ((self.z_first.to(torch.float32) * wt.sw[None, :]) * self.sa[:, None]).to(torch.bfloat16)
                d = R.diff(self.y, R.bits16(rev))
        except Exception as ex:  # noqa: BLE001 - a finding
            torch.cuda.synchronize()
            self.gates[key] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
            return None
        self.gates[key] = {"mismatches": d, "against": "the Definition's words" if swap else "z's epilogue, s_w first"}
        return lambda: [call(wt_) for wt_ in self.ws]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(FX.DEFAULT_MODELS))
    ap.add_argument("--ms", default=",".join(map(str, SWEEP_MS)))
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--peak-tops", type=float, default=None, help="default: PEAKS' measured reading for the arch")
    ap.add_argument("--kernel", choices=sorted(R.KERNELS), default=None)
    ap.add_argument("--configs", default=None, help="a tune.py best.json for the exact kernel")
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(a.models.split(","))
    ms = tuple(int(v) for v in a.ms.split(","))
    kerns = {"exact": R.kernel(a.kernel)}
    if a.configs:
        R.load_configs(a.configs, kerns["exact"])
    arms = tuple(arm for arm in a.arms.split(",") if arm != "exact_triton" or kerns["exact"].name != "triton")
    if "exact_triton" in arms:
        kerns["exact_triton"] = R.kernel("triton")
    if "simple" in arms:
        kerns["simple"] = R.kernel("int_mm")
    readings = PEAKS.get(X.arch(), {})
    peak = a.peak_tops or readings["measured"]
    clock = kbench.nvml()
    meta = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "torch": torch.__version__, "triton": triton.__version__,
            "cublaslt": LT.version(), "exact": kerns["exact"].meta(),
            "configs": {key: {f"{n},{k}": v for (n, k), v in t.items()} for key, t in R.CONFIGS.items()},
            "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size, "peak_tops": peak, "peak_readings": readings}
    meta["bw_before"] = bandwidth(clock, a.reps)
    print(json.dumps({"bw_before": {k: meta["bw_before"][k] for k in ("read_gbs", "copy_gbs")}}), flush=True)
    rows = []
    for name, n, k in FX.shapes(models, with_head=not a.no_head):
        for m in ms:
            t0 = time.time()
            lin = Linear(n, k, m, kerns)
            fns = {}
            for arm in arms:
                f = (lin.exact(arm) if arm in kerns else lin.lt(clock) if arm == "lt"
                     else lin.cutlass(swap=arm == "cutlass_swap"))
                if f is not None:
                    fns[arm] = f
            med, us, tm = kbench.timed(fns, a.reps, clock, lin.copies) if fns else ({}, {}, {"clocks": {}})
            row = {"name": name, "n": n, "k": k, "m": m, "copies": lin.copies, "us": med,
                   "us_all": {arm: [round(v, 3) for v in us[arm]] for arm in us}, "sm_clock_mhz": tm["clocks"],
                   "gates": lin.gates, "s": round(time.time() - t0, 1)}
            rows.append(row)
            with open(out / "bench.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"name": name, "m": m, "us": {k_: round(v, 2) for k_, v in med.items()},
                              "frac_peak": {k_: round(2 * m * n * k / v / 1e6 / peak, 3) for k_, v in med.items()},
                              "lt": lin.gates.get("lt", {}).get("best"), "s": row["s"]}), flush=True)
            del lin, fns
            torch.cuda.empty_cache()
    meta["bw_after"] = bandwidth(clock, a.reps)
    bw = max(meta["bw_before"]["read_gbs"], meta["bw_after"]["read_gbs"])
    doc = {**meta, **RF.summarize(rows, models, peak, bw, OUT_BYTES)}
    (out / "crossing.json").write_text(json.dumps(doc, indent=1) + "\n")
    for name in models:
        for arm, c in doc["layers"].get(name, {}).items():
            print(json.dumps({"model": name, "arm": arm, "ridge_card": round(c["ridge_card"], 1), "ridge_arm": round(c["ridge_arm"], 1),
                              "at_ridge": {lab: {kk: round(vv, 3) for kk, vv in v.items()} for lab, v in c["at_ridge"].items()}}),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
