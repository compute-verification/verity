"""The exact W8A8 linear at the large model's linears, and its gate against `ScaledMmS8Coordinate_v1{K, DOT}` over W's rows.

Three exact backends (`--kernel`, default wgmma on sm_90, triton elsewhere), each with the Definition's epilogue in its order:
cvt.rn.f32.s32, mul.rn by s_a, mul.rn by s_w, cvt.rn.bf16.f32, no FTZ, no FMA:
    wgmma   s8_wgmma.cu through wgmma.py on sm_90a (the H100 and H200): warpgroup MMAs (wgmma m64nNk32 .s32.s8.s8) into one
            s32 accumulator; its SASS gate is build.sh's, held to the cubin by the MANIFEST.sha256 beside it
    triton  `verity_vllm.engine.int8_fused.scaled_mm`, as vLLM's int8 profile serves it (mma.sync m16n8k32 on sm_120), its PTX
            (`int8_fused.ptx_check`) and SASS (the s8 MMA present; no FFMA; no .FTZ) checked per compiled kernel
    int_mm  int_mm.py's `linear`: torch._int_mm (cuBLASLt's s32 GEMM) and a separate epilogue, its config: "fused" (one Triton
            kernel of inline PTX, held to `int_mm.ptx_audit` per compiled variant) or "eager" (four torch kernels)
Their configs are `configs.CONFIGS` (tune.py's gated winners, or --configs a tune.py best.json), else the kernel's default.

    python run.py --out OUT [--kernel wgmma|triton|int_mm] [--models mqa16k52] [--ms 1,7,16,...] [--extra-cfgs CFG,...]
                  [--configs BEST_JSON] [--no-edge] [--no-swap] [--no-head]

Per linear (N, K) of `fixture.shapes(models)` and M, on `fixture`'s inputs (all of s8, structured rows and columns, special
scales):
    sums       z = X W^T by `int_mm.int_mm` (torch._int_mm, cuBLASLt, s32), held to `fixture.sums` (float64, exact) on the structured and
               special rows and columns and sampled others
    twin       `fixture.epilogue(z, s_a, s_w)` on every cell: the Definition's words (tests/test_twin.py holds it to the
               Definition's evaluator)
    reference  torch's epilogue on the GPU with every NaN 0x7FFF (what tune.py compares against), counted against the twin
    kernel     the picked config and each --extra-cfgs: y written into 0xA5 equal to the twin on every cell; again on a second
               call; again from a CUDA graph's replay into 0xA5; triton's split workspace zero after; the kernel's checks
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
import int_mm as IM  # noqa: E402
import wgmma as W  # noqa: E402
from configs import CONFIGS  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

POISON = 0xA5
TRITON_EXTRA = ("16x32x256:w4:s4:k1", "32x64x128:w4:s4:k1", "64x64x128:w4:s3:k2", "128x64x128:w4:s4:k1",
                "128x128x64:w4:s3:k1:t1", "128x128x64:w4:s3:k1:t2")
EDGE_N, EDGE_MS = 64, (16, 64)
SAMPLE_ROWS, SAMPLE_COLS = 8, 64


def table_cfg(kern, m: int, n: int, k: int):
    rows = CONFIGS.get(f"{kern.name}:{X.arch()}", {}).get((n, k))
    if not rows:
        return None
    for m_max, text in rows:
        if m <= m_max:
            return kern.Cfg.parse(text)
    return kern.Cfg.parse(rows[-1][1])


def load_configs(path: str, kern) -> None:
    """A tune.py best.json's winners into CONFIGS for `kern` on this arch, each M its m_max."""
    table = json.loads(Path(path).read_text())["table"]
    CONFIGS[f"{kern.name}:{X.arch()}"] = {tuple(map(int, nk.split(","))): [(r[0], r[1]) for r in rows if r[1]]
                                          for nk, rows in table.items()}


class Wgmma:
    """s8_wgmma.cubin through wgmma.py (sm_90a)."""
    name, Cfg, extra = "wgmma", W.Cfg, W.EXTRA_CFGS

    def __init__(self):
        self.lib = W.lib()
        self.ident = self.lib.identity()

    def pick(self, m: int, n: int, k: int) -> W.Cfg:
        return table_cfg(self, m, n, k) or W.default_cfg(m, n, k, self.lib.sms)

    def baseline(self, m: int, n: int, k: int) -> W.Cfg:
        return W.default_cfg(m, n, k, self.lib.sms)

    def skip(self, cfg, m: int, n: int) -> str | None:
        return None

    def call(self, x, sa, w, sw, y, cfg):
        return W.scaled_mm(x, sa, w, sw, y, cfg)

    def checks(self, handle) -> dict:
        return {"launch": handle, "cubin_sha256": self.ident["sha256"][:16], "ok": self.ident["ok"]}

    def residue(self) -> int:
        return 0

    def meta(self) -> dict:
        return {"kernel": self.name, "identity": self.ident}


class Triton:
    """verity_vllm's int8_fused: sm_120's exact kernel, and on sm_90 a second one beside wgmma."""
    name, Cfg, extra = "triton", X.Cfg, TRITON_EXTRA

    def __init__(self):
        self.work = X.Workspace("cuda")

    def pick(self, m: int, n: int, k: int) -> X.Cfg:
        return table_cfg(self, m, n, k) or X.pick(m, n, k)

    def baseline(self, m: int, n: int, k: int) -> X.Cfg:
        return X.default_cfg(m, n, k)

    def skip(self, cfg, m: int, n: int) -> str | None:
        return "split-K workspace too small for M x N" if cfg[5] > 1 and m * n > self.work.ws.numel() else None

    def call(self, x, sa, w, sw, y, cfg):
        return X.scaled_mm(x, sa, w, sw, y, cfg, self.work)

    def checks(self, km) -> dict:
        ptx, sass = X.ptx_check(km), sass_counts(km)
        return {"ptx": ptx, "sass": sass, "ok": ptx["ok"] and sass["ok"]}

    def residue(self) -> int:
        return int((self.work.ws != 0).sum()) + int((self.work.cnt != 0).sum())

    def meta(self) -> dict:
        return {"kernel": self.name, "triton": __import__("triton").__version__}


class Epilogue(str):
    """IntMm's config: one of `int_mm.EPILOGUES`."""

    @classmethod
    def parse(cls, text: str) -> Epilogue:
        if text not in IM.EPILOGUES:
            raise ValueError(f"int_mm's config {text!r}: one of {IM.EPILOGUES}")
        return cls(text)


class IntMm:
    """int_mm.py's `linear`: torch._int_mm and the fused epilogue, the eager one beside it."""
    name, Cfg, extra = "int_mm", Epilogue, ("eager",)

    def pick(self, m: int, n: int, k: int) -> Epilogue:
        return table_cfg(self, m, n, k) or Epilogue("fused")

    def baseline(self, m: int, n: int, k: int) -> Epilogue:
        return Epilogue("fused")

    def skip(self, cfg, m: int, n: int) -> str | None:
        return None

    def call(self, x, sa, w, sw, y, cfg):
        IM.linear(x, sa, w, sw, epilogue=cfg, out=y)
        return cfg, x, sa, w, sw

    def checks(self, handle) -> dict:
        cfg, x, sa, w, sw = handle
        if cfg != "fused":
            return {"epilogue": cfg, "ok": True}
        audit = IM.ptx_audit(IM.fused_ptx(IM.int_mm(x, w), sa, sw))
        return {"epilogue": cfg, "ptx": audit, "ok": audit["ok"]}

    def residue(self) -> int:
        return 0

    def meta(self) -> dict:
        return {"kernel": self.name, "triton": IM.triton.__version__ if IM.triton else None}


KERNELS = {"wgmma": Wgmma, "triton": Triton, "int_mm": IntMm}


def kernel(name: str | None = None):
    return KERNELS[name or ("wgmma" if X.arch() == "sm_90" else "triton")]()


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
    """A Triton kernel's SASS: the s8 tensor-core step (IMMA, or IGMMA on sm_90), FFMA (a fused multiply-add would round
    once), .FTZ."""
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
            ("imma", r"\bI(?:G)?MMA\.\S*"), ("ffma", r"\bFFMA\b"), ("ftz", r"\.FTZ\b"), ("int_div", r"\bF2I\.FTZ\.U32\.TRUNC\.NTZ\b"),
            ("fmul", r"\bFMUL\b"), ("i2f", r"\bI2F\S*"), ("f2fp_bf16", r"\bF2FP\.BF16\S*"), ("local", r"\b(?:LDL|STL)\b"))}
        doc["imma_forms"] = sorted(set(re.findall(r"\bI(?:G)?MMA\.\S+", sass)))
        doc["cubin_sha256"] = key
        doc["ok"] = doc["imma"] > 0 and doc["ffma"] == 0 and doc["ftz"] == doc["int_div"] and doc["local"] == 0
        _SASS[key] = doc
    return _SASS[key]


class Weights:
    def __init__(self, n: int, k: int, seed: int, special: bool = True):
        self.n, self.k = n, k
        self.w = codes(n, k, seed, FX.STRUCT_COLS)
        self.sw_np = FX.weight_scales(n, seed, special)
        self.sw = torch.from_numpy(self.sw_np).cuda()

    @classmethod
    def given(cls, w: np.ndarray, sw: np.ndarray) -> Weights:
        """W's codes [N, K] int8 and s_w [N] float32 as they are."""
        self = cls.__new__(cls)
        self.n, self.k = w.shape
        self.w = torch.from_numpy(np.ascontiguousarray(w, dtype=np.int8)).cuda()
        self.sw_np = np.ascontiguousarray(sw, dtype=np.float32)
        self.sw = torch.from_numpy(self.sw_np).cuda()
        return self


class Case:
    """One (N, K, M): the activations, the s32 sums, the twin's words and the reference's, on the GPU."""

    def __init__(self, wt: Weights, m: int, seed: int, special: bool = True, twin: bool = True):
        self.wt, self.m = wt, m
        n, k = wt.n, wt.k
        self.x = codes(m, k, seed, FX.STRUCT_ROWS)
        self.sa_np = FX.act_scales(m, seed, special)
        self.sa = torch.from_numpy(self.sa_np).cuda()
        self.z = IM.int_mm(self.x, wt.w)
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


def run_cfg(case: Case, cfg, kern) -> dict:
    wt, y = case.wt, case.y
    rec: dict = {"cfg": str(cfg)}
    why = kern.skip(cfg, case.m, wt.n)
    if why:
        return {**rec, "skipped": why}
    try:
        h = None
        for i in range(2):
            poison(y)
            h = kern.call(case.x, case.sa, wt.w, wt.sw, y, cfg)
            torch.cuda.synchronize()
            rec[f"mismatch_call{i}"] = diff(y, case.twin16)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kern.call(case.x, case.sa, wt.w, wt.sw, y, cfg)
        poison(y)
        g.replay()
        torch.cuda.synchronize()
        rec["mismatch_graph"] = diff(y, case.twin16)
        del g
        rec["residue"] = kern.residue()
        rec["checks"] = kern.checks(h)
    except Exception as ex:  # noqa: BLE001 - a config that doesn't compile or launch is a failed case, recorded
        torch.cuda.synchronize()
        return {**rec, "ok": False, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
    rec["ok"] = (rec["mismatch_call0"] == 0 and rec["mismatch_call1"] == 0 and rec["mismatch_graph"] == 0
                 and rec["residue"] == 0 and rec["checks"]["ok"])
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


def gate_shape(name: str, wt: Weights, ms: tuple[int, ...], extra: tuple, kern, do_swap: bool, log) -> list[dict]:
    out, n, k = [], wt.n, wt.k
    for m in ms:
        t0 = time.time()
        case = Case(wt, m, seed=(n * 31 + k * 7 + m) & 0xFFFFFF)
        rec = {"name": name, "n": n, "k": k, "m": m, "sums": case.sums_check(m),
               "reference_vs_twin": diff(case.ref16.view(torch.bfloat16), case.twin16), "control": control(case)}
        picked = kern.pick(m, n, k)
        rec["picked"] = run_cfg(case, picked, kern)
        rec["extra"] = [run_cfg(case, c, kern) for c in extra if c != picked]
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


def logger(path: Path):
    def log(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({key: rec[key] for key in ("name", "m", "ok", "s")} | {
            "picked": rec["picked"].get("cfg"), "mismatch": rec["picked"].get("mismatch_call0"),
            "extra_mismatch": sum(e.get("mismatch_call0", 0) for e in rec["extra"]),
            "ref_vs_twin": rec["reference_vs_twin"], "error": rec["picked"].get("error")}), flush=True)
    return log


def summary(recs: list[dict], kern, ms: tuple[int, ...]) -> dict:
    cases = [r["picked"] for r in recs] + [e for r in recs for e in r["extra"] if "skipped" not in e]
    table = CONFIGS.get(f"{kern.name}:{X.arch()}", {})
    return {"device": torch.cuda.get_device_name(), "arch": X.arch(), "torch": torch.__version__, **kern.meta(), "ms": ms,
            "configs": {f"{n},{k}": v for (n, k), v in table.items()}, "cases": len(cases),
            "kernel_mismatches": sum(c.get("mismatch_call0", 0) + c.get("mismatch_call1", 0) + c.get("mismatch_graph", 0)
                                     for c in cases),
            "failed": [(r["name"], r["m"]) for r in recs if not r["ok"]],
            "reference_vs_twin": sum(r["reference_vs_twin"] for r in recs),
            "swap_mismatches": sum(r.get("swap", {}).get("mismatches", 0) for r in recs),
            "cells_checked": sum(r["m"] * r["n"] for r in recs), "ok": all(r["ok"] for r in recs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", choices=sorted(KERNELS), default=None)
    ap.add_argument("--models", default=",".join(FX.DEFAULT_MODELS))
    ap.add_argument("--ms", default=None, help="default: fixture.GATE_MS and every configs.CONFIGS m_max")
    ap.add_argument("--extra-cfgs", default=None, help="default: the kernel's (wgmma.EXTRA_CFGS, TRITON_EXTRA, int_mm's eager)")
    ap.add_argument("--configs", default=None, help="a tune.py best.json for this kernel")
    ap.add_argument("--no-edge", action="store_true")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not FX.fp_env_ok():
        raise SystemExit("this process flushes float32 subnormals; the twin would be wrong")
    kern = kernel(a.kernel)
    if a.configs:
        load_configs(a.configs, kern)
    table = CONFIGS.get(f"{kern.name}:{X.arch()}", {})
    models = tuple(a.models.split(","))
    tuned = {m for rows in table.values() for m, _ in rows}
    ms = tuple(int(v) for v in a.ms.split(",")) if a.ms else tuple(sorted(set(FX.GATE_MS) | tuned))
    extra = tuple(kern.Cfg.parse(c) for c in (a.extra_cfgs.split(",") if a.extra_cfgs is not None else kern.extra) if c)
    log = logger(out / "gate.jsonl")
    recs = []
    for name, n, k in FX.shapes(models, with_head=not a.no_head):
        recs += gate_shape(name, Weights(n, k, seed=(n * 131 + k) & 0xFFFFFF), ms, extra, kern, not a.no_swap, log)
    if not a.no_edge:
        recs += gate_shape("s32-edge", Weights(EDGE_N, FX.EDGE_K, seed=(EDGE_N * 131 + FX.EDGE_K) & 0xFFFFFF), EDGE_MS, extra,
                           kern, not a.no_swap, log)
    doc = {"models": models, **summary(recs, kern, ms)}
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
