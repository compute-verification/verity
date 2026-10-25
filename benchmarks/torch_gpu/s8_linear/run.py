"""The exact W8A8 linear at the large model's linears on sm_120, and its gate against `ScaledMmS8Coordinate_v1{K,
BlackwellS8ImmaDot32_v1}` over W's rows.

The kernel is `verity_vllm.engine.int8_fused.scaled_mm`, as vLLM's int8 profile serves it: one Triton GEMM, s8 x s8 -> s32 on
the tensor cores (`tl.dot`, mma.sync m16n8k32 on sm_120), whose epilogue is the Definition's, in its order: cvt.rn.f32.s32,
mul.rn by s_a, mul.rn by s_w, cvt.rn.bf16.f32, no FTZ, no FMA. Its configs here are `configs.CONFIGS` (tune.py's gated
winners at these shapes), else `int8_fused.pick`.

    python run.py --out OUT [--ds 8192,12288] [--ms 1,7,16,...] [--extra-cfgs CFG,...] [--no-edge] [--no-swap]

Per linear (N, K) of `fixture.shapes(ds)` and M, on `fixture`'s inputs (all of s8, structured rows and columns, special
scales):
    sums       z = X W^T by torch._int_mm (cuBLASLt, s32), held to `fixture.sums` (float64, exact) on the structured and
               special rows and columns and sampled others
    twin       `fixture.epilogue(z, s_a, s_w)` on every cell: the Definition's words (tests/test_twin.py holds it to the
               Definition's evaluator)
    reference  torch's epilogue on the GPU with every NaN 0x7FFF (what tune.py compares against), counted against the twin
    kernel     the picked config and each --extra-cfgs: y written into 0xA5 equal to the twin on every cell; again on a second
               call (a split's workspace zeroes itself); again from a CUDA graph's replay into 0xA5; the split workspace zero
               after; the PTX (`int8_fused.ptx_check`) and the SASS (IMMA present; no FFMA; no .FTZ)
    control    the same 0xA5 y with no launch: must differ from the twin
    swap       vLLM's CUTLASS int8 GEMM on the transposed product, y^T = W X^T with scale_b = s_a (applied first, the
               Definition's order) and scale_a = s_w (`kgate.cutlass_swapped`), M a multiple of 16; a finding, not a gate
and the s32 edge: (N, K) = (64, 131040), K the largest the s8 Definitions take, whose all -128 row and column sum to
2,146,959,360. Writes gate.json; exits 0 only when every kernel case has no mismatch and every control differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as FX  # noqa: E402
from configs import CONFIGS  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

POISON = 0xA5
EXTRA_CFGS = ("16x32x256:w4:s4:k1", "32x64x128:w4:s4:k1", "64x64x128:w4:s3:k2", "128x64x128:w4:s4:k1",
              "128x128x64:w4:s3:k1:t1", "128x128x64:w4:s3:k1:t2")
EDGE_N, EDGE_MS = 64, (16, 64)
SAMPLE_ROWS, SAMPLE_COLS = 8, 64


def cfg_for(m: int, n: int, k: int) -> X.Cfg:
    rows = CONFIGS.get((n, k))
    if not rows:
        return X.pick(m, n, k)
    for m_max, text in rows:
        if m <= m_max:
            return X.Cfg.parse(text)
    return X.Cfg.parse(rows[-1][1])


def codes(rows: int, k: int, seed: int, structured: dict[int, str]) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.randint(-128, 128, (rows, k), generator=g, dtype=torch.int16, device="cuda").to(torch.int8)
    for i, kind in structured.items():
        if i < rows:
            t[i] = torch.from_numpy(FX.structured(kind, k)).cuda()
    return t


def bits16(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def poison(*ts: torch.Tensor) -> None:
    for t in ts:
        t.view(torch.uint8).fill_(POISON)


def diff(y: torch.Tensor, want16: torch.Tensor) -> int:
    return int((bits16(y) != want16).sum())


def reference(z: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """The epilogue in torch on the GPU, as int16 words, every NaN 0x7FFF (torch's CPU cast would give 0x7FC0)."""
    t = (z.to(torch.float32) * sa[:, None]) * sw[None, :]
    y = bits16(t.to(torch.bfloat16))
    return torch.where(torch.isnan(t), torch.full_like(y, 0x7FFF), y)


_SASS: dict[str, dict] = {}


def sass_counts(km) -> dict:
    """The compiled kernel's SASS: IMMA (the s8 tensor-core step), FFMA (a fused multiply-add would round once), .FTZ."""
    cubin = km.asm["cubin"]
    key = hashlib.sha256(cubin).hexdigest()
    if key not in _SASS:
        tool = shutil.which("cuobjdump")
        if tool is None:
            raise RuntimeError("no cuobjdump on the PATH")
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(cubin)
            f.flush()
            sass = subprocess.run([tool, "-sass", f.name], capture_output=True, text=True, check=True).stdout
        doc = {name: len(re.findall(pat, sass)) for name, pat in (
            ("imma", r"\bIMMA\.\S*"), ("ffma", r"\bFFMA\b"), ("ftz", r"\.FTZ\b"), ("fmul", r"\bFMUL\b"),
            ("i2f", r"\bI2F\S*"), ("f2fp_bf16", r"\bF2FP\.BF16\S*"), ("local", r"\b(?:LDL|STL)\b"))}
        doc["imma_forms"] = sorted(set(re.findall(r"\bIMMA\.\S+", sass)))
        doc["cubin_sha256"] = key
        doc["ok"] = doc["imma"] > 0 and doc["ffma"] == 0 and doc["ftz"] == 0 and doc["local"] == 0
        _SASS[key] = doc
    return _SASS[key]


class Weights:
    def __init__(self, n: int, k: int, seed: int, special: bool = True):
        self.n, self.k = n, k
        self.w = codes(n, k, seed, FX.STRUCT_COLS)
        self.sw_np = FX.weight_scales(n, seed, special)
        self.sw = torch.from_numpy(self.sw_np).cuda()


class Case:
    """One (N, K, M): the activations, the s32 sums, the twin's words and the reference's, on the GPU."""

    def __init__(self, wt: Weights, m: int, seed: int, special: bool = True, twin: bool = True):
        self.wt, self.m = wt, m
        n, k = wt.n, wt.k
        self.x = codes(m, k, seed, FX.STRUCT_ROWS)
        self.sa_np = FX.act_scales(m, seed, special)
        self.sa = torch.from_numpy(self.sa_np).cuda()
        self.z = E.int_mm(self.x, wt.w)
        self.ref16 = reference(self.z, self.sa, wt.sw)
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.twin16 = self._twin() if twin else None

    def _twin(self, chunk_cells: int = 1 << 24) -> torch.Tensor:
        out = torch.empty((self.m, self.wt.n), dtype=torch.int16, device="cuda")
        step = max(1, chunk_cells // self.wt.n)
        for i in range(0, self.m, step):
            zc = self.z[i:i + step].cpu().numpy()
            out[i:i + step] = torch.from_numpy(FX.epilogue(zc, self.sa_np[i:i + step], self.wt.sw_np).view(np.int16)).cuda()
        return out

    def sums_check(self, seed: int) -> dict:
        """z against float64's exact sums on the structured and special rows and columns and sampled others."""
        rng = np.random.default_rng(seed)
        m, n = self.m, self.wt.n
        edge = FX.FIRST_SPECIAL + len(FX.SPECIAL_SA)
        rows = sorted(set(range(min(m, edge))) | set(rng.integers(0, m, SAMPLE_ROWS).tolist()))
        cols = sorted(set(range(min(n, edge))) | set(rng.integers(0, n, SAMPLE_COLS).tolist()))
        want = FX.sums(self.x[rows].cpu().numpy(), self.wt.w[cols].cpu().numpy())
        got = self.z[rows][:, cols].cpu().numpy()
        return {"rows": len(rows), "cols": len(cols), "mismatches": int((got != want).sum()),
                "max_abs": int(np.abs(want.astype(np.int64)).max(initial=0))}


def run_cfg(case: Case, cfg: X.Cfg, work: X.Workspace) -> dict:
    wt, y = case.wt, case.y
    rec: dict = {"cfg": str(cfg)}
    if cfg[5] > 1 and case.m * wt.n > work.ws.numel():
        return {**rec, "skipped": "split-K workspace too small for M x N"}
    try:
        km = None
        for i in range(2):
            poison(y)
            km = X.scaled_mm(case.x, case.sa, wt.w, wt.sw, y, cfg, work)
            torch.cuda.synchronize()
            rec[f"mismatch_call{i}"] = diff(y, case.twin16)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            X.scaled_mm(case.x, case.sa, wt.w, wt.sw, y, cfg, work)
        poison(y)
        g.replay()
        torch.cuda.synchronize()
        rec["mismatch_graph"] = diff(y, case.twin16)
        del g
        rec["workspace_nonzero"] = int((work.ws != 0).sum()) + int((work.cnt != 0).sum())
        rec["ptx"] = X.ptx_check(km)
        rec["sass"] = sass_counts(km)
    except Exception as ex:  # noqa: BLE001 - a config that doesn't compile or launch is a failed case, recorded
        torch.cuda.synchronize()
        return {**rec, "ok": False, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
    rec["ok"] = (rec["mismatch_call0"] == 0 and rec["mismatch_call1"] == 0 and rec["mismatch_graph"] == 0
                 and rec["workspace_nonzero"] == 0 and rec["ptx"]["ok"] and rec["sass"]["ok"])
    return rec


def control(case: Case) -> dict:
    poison(case.y)
    torch.cuda.synchronize()
    d = diff(case.y, case.twin16)
    return {"mismatches": d, "cells": case.m * case.wt.n, "rejected": d > 0}


def swap(case: Case) -> dict:
    """vLLM's CUTLASS on the transposed product (`kgate.cutlass_swapped`), against the twin."""
    if case.m % 16:
        return {"skipped": "M is not a multiple of 16"}
    try:
        import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)
        yt = torch.empty((case.wt.n, case.m), dtype=torch.bfloat16, device="cuda")
        poison(yt)
        torch.ops._C.cutlass_scaled_mm(yt, case.wt.w, case.x.t(), case.wt.sw.reshape(-1, 1), case.sa.reshape(-1, 1), None)
        torch.cuda.synchronize()
        return {"mismatches": diff(yt.t().contiguous(), case.twin16)}
    except Exception as ex:  # noqa: BLE001 - a finding
        torch.cuda.synchronize()
        return {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}


def gate_shape(name: str, n: int, k: int, ms: tuple[int, ...], extra: tuple[X.Cfg, ...], work: X.Workspace, do_swap: bool,
               log) -> list[dict]:
    out = []
    wt = Weights(n, k, seed=(n * 131 + k) & 0xFFFFFF)
    for m in ms:
        t0 = time.time()
        case = Case(wt, m, seed=(n * 31 + k * 7 + m) & 0xFFFFFF)
        rec = {"name": name, "n": n, "k": k, "m": m, "sums": case.sums_check(m),
               "reference_vs_twin": diff(case.ref16.view(torch.bfloat16), case.twin16), "control": control(case)}
        picked = cfg_for(m, n, k)
        rec["picked"] = run_cfg(case, picked, work)
        rec["extra"] = [run_cfg(case, c, work) for c in extra if c != picked]
        if do_swap:
            rec["swap"] = swap(case)
        rec["ok"] = (rec["picked"]["ok"] and all(r.get("ok", True) for r in rec["extra"]) and rec["control"]["rejected"]
                     and rec["sums"]["mismatches"] == 0)
        rec["s"] = round(time.time() - t0, 2)
        log(rec)
        out.append(rec)
        del case
        torch.cuda.empty_cache()
    del wt
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", default="8192,12288")
    ap.add_argument("--ms", default=None, help="default: fixture.GATE_MS and every configs.CONFIGS m_max")
    ap.add_argument("--extra-cfgs", default=",".join(EXTRA_CFGS))
    ap.add_argument("--no-edge", action="store_true")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not FX.fp_env_ok():
        raise SystemExit("this process flushes float32 subnormals; the twin would be wrong")
    ds = tuple(int(v) for v in a.ds.split(","))
    tuned = {m for rows in CONFIGS.values() for m, _ in rows}
    ms = tuple(int(v) for v in a.ms.split(",")) if a.ms else tuple(sorted(set(FX.GATE_MS) | tuned))
    extra = tuple(X.Cfg.parse(c) for c in a.extra_cfgs.split(",") if c)
    work = X.Workspace("cuda")
    log_path = out / "gate.jsonl"

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({key: rec[key] for key in ("name", "m", "ok", "s")} | {
            "picked": rec["picked"].get("cfg"), "mismatch": rec["picked"].get("mismatch_call0"),
            "ref_vs_twin": rec["reference_vs_twin"]}), flush=True)

    recs = []
    for name, n, k in FX.shapes(ds, with_head=not a.no_head):
        recs += gate_shape(name, n, k, ms, extra, work, not a.no_swap, log)
    if not a.no_edge:
        recs += gate_shape("s32-edge", EDGE_N, FX.EDGE_K, EDGE_MS, extra, work, not a.no_swap, log)
    cases = [r["picked"] for r in recs] + [e for r in recs for e in r["extra"] if "skipped" not in e]
    doc = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "torch": torch.__version__,
           "triton": __import__("triton").__version__, "ds": ds, "ms": ms, "configs": {f"{n},{k}": v for (n, k), v in CONFIGS.items()},
           "cases": len(cases), "kernel_mismatches": sum(c.get("mismatch_call0", 0) + c.get("mismatch_call1", 0)
                                                         + c.get("mismatch_graph", 0) for c in cases),
           "failed": [(r["name"], r["m"]) for r in recs if not r["ok"]],
           "reference_vs_twin": sum(r["reference_vs_twin"] for r in recs),
           "swap_mismatches": sum(r.get("swap", {}).get("mismatches", 0) for r in recs),
           "cells_checked": sum(r["m"] * r["n"] for r in recs), "ok": all(r["ok"] for r in recs)}
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
