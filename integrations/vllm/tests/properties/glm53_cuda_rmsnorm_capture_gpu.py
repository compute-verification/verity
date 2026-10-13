"""The served CUDA RMSNorm kernels on an H200, word-for-word against their registered models.

Runs vLLM d9105ea80's default (non-batch-invariant) BF16 ``rms_norm_kernel`` and width-8
``fused_add_rms_norm_kernel`` at M in {1,42,43,64,256,2048}, hidden widths 6144 and 7168.
Every normed and residual word is checked twice for determinism and against the numpy twins.
CUDA profiler traces provide each launch's grid/block dimensions and kernel symbol.

Exit 0 is GO, 1 is NO-GO, and 3 is a device-gate failure (cc 9.0, 132 SMs, driver >= 575).

    cd integrations/vllm
    python tests/properties/glm53_cuda_rmsnorm_capture_gpu.py --out DIR --scratch /tmp \\
      --want-cc 9.0 --want-sms 132 --min-driver 575
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from verity_vllm.program.kernels import cuda_rmsnorm_rows as R
from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.registry import cuda_rmsnorm as M
from verity_vllm.program.registry import prims as P
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-cuda-rmsnorm-capture/v1"
RECORD = "glm53_cuda_rmsnorm_capture.json"
MS = (1, 42, 43, 64, 256, 2048)
WIDTHS = (6144, 7168)
EPS_BY_WIDTH = {6144: 1e-5, 7168: 1e-6}
KINDS = ("rms_norm", "fused_add_rms_norm")
SASS_SOURCE = Path(__file__).with_name("cuda_rmsnorm_sass.cu")


def _u16(t) -> np.ndarray:
    return t.detach().view(np.int16).cpu().numpy().view(np.uint16).copy()


def _tensor(words: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(words, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).cuda()


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def special_rows(n: int, seed: int) -> np.ndarray:
    """Rows covering zero, subnormal, overflow, infinity, NaN, cancellation and reduction-sensitive magnitudes."""
    rng = np.random.default_rng(seed)
    rows = [
        np.zeros(n, np.uint16),
        np.full(n, 0x0001, np.uint16),
        np.full(n, 0x0080, np.uint16),
        np.full(n, 0x7F7F, np.uint16),
        np.tile(np.array([0x0000, 0x8000, 0x3F80, 0xBF80], np.uint16), n // 4),
    ]
    for word in (0x7F80, 0xFF80, 0x7FC1):
        row = _bf16(rng.standard_normal(n).astype(np.float32))
        row[17] = word
        rows.append(row)
    # Dynamic-range rows distinguish reduction trees; alternating signs do not change squares but vary the epilogue.
    for lo, hi in ((-20, 20), (-60, 30), (-12, 8), (-3, 3)):
        x = rng.standard_normal(n).astype(np.float32) * np.exp2(rng.integers(lo, hi + 1, n)).astype(np.float32)
        rows.append(_bf16(x))
    return np.stack(rows)


def operands(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((m, n)).astype(np.float32) *
              np.exp2(rng.integers(-8, 9, (m, 1))).astype(np.float32))
    res = _bf16(rng.standard_normal((m, n)).astype(np.float32) *
                np.exp2(rng.integers(-8, 9, (m, 1))).astype(np.float32))
    planted = special_rows(n, seed + 1)
    k = min(m, len(planted))
    x[:k], res[:k] = planted[:k], planted[::-1][:k]
    w = _bf16((1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32))
    w[:8] = np.array([0x0000, 0x8000, 0x0001, 0x0080, 0x3F80, 0xBF80, 0x7F80, 0x7FC1], np.uint16)
    return x, res, w


def _profiler_launch(trace: Path, needle: str) -> dict[str, Any]:
    events = json.loads(trace.read_text()).get("traceEvents", [])
    found = [e for e in events if needle in str(e.get("name", "")) and
             ("kernel" in str(e.get("cat", "")).lower() or "block" in (e.get("args") or {}))]
    if not found:
        return {"found": False, "kernel": None, "grid": None, "block": None}
    event = found[-1]
    args = event.get("args") or {}

    def dim(name: str):
        value = next((v for k, v in args.items() if k.lower().replace(" ", "") == name), None)
        if isinstance(value, str):
            value = [int(x) for x in value.strip("[]()").replace(",", " ").split() if x.isdigit()]
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
        return value

    return {"found": True, "kernel": event.get("name"), "grid": dim("grid"), "block": dim("block"),
            "args": {k: v for k, v in args.items() if any(q in k.lower() for q in ("grid", "block", "stream", "device"))}}


def sass(out: Path) -> dict[str, Any]:
    """Compile the pinned width-8 arithmetic for sm_90 and retain its SASS and operation census."""
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuobjdump = shutil.which("cuobjdump") or str(Path(nvcc).with_name("cuobjdump"))
    cubin, text = out / "cuda_rmsnorm_sm90.cubin", out / "cuda_rmsnorm_sm90.sass"
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-gencode", "arch=compute_90,code=sm_90"]
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    build = subprocess.run([nvcc, *flags, "-cubin", str(SASS_SOURCE), "-o", str(cubin)], capture_output=True, text=True)
    if build.returncode:
        return {"ok": False, "nvcc": version.stdout.strip().splitlines()[-1:], "flags": flags,
                "error": (build.stdout + build.stderr)[-2000:]}
    dump = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], capture_output=True, text=True)
    text.write_text(dump.stdout)
    ops = {op: len(re.findall(rf"\b{re.escape(op)}", dump.stdout)) for op in
           ("FFMA", "FADD", "SHFL", "MUFU.RSQ", "F2FP", "HADD2")}
    kernels = sorted(set(re.findall(r"Function\s*:\s*(\S+)", dump.stdout)))
    return {"ok": dump.returncode == 0 and len(kernels) == 2 and ops["SHFL"] > 0 and ops["MUFU.RSQ"] == 2 and
                  ops["FFMA"] > 0 and ops["FADD"] > 0 and ops["F2FP"] > 0,
            "nvcc": version.stdout.strip().splitlines()[-1:], "flags": flags, "kernels": kernels, "ops": ops,
            "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(), "sass": text.name,
            "error": dump.stderr[-1000:]}


def _profiled(kind: str, xw: np.ndarray, rw: np.ndarray, ww: np.ndarray, eps: float, trace: Path):
    import torch
    import vllm._custom_ops  # noqa: F401 -- registers torch.ops._C

    x, res, w = _tensor(xw), _tensor(rw), _tensor(ww)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        if kind == "rms_norm":
            out = torch.empty_like(x)
            torch.ops._C.rms_norm(out, x, w, eps)
        else:
            out = x
            torch.ops._C.fused_add_rms_norm(out, res, w, eps)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    return _u16(out), _u16(res)


def _plain(kind: str, xw: np.ndarray, rw: np.ndarray, ww: np.ndarray, eps: float):
    import torch

    x, res, w = _tensor(xw), _tensor(rw), _tensor(ww)
    if kind == "rms_norm":
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, w, eps)
    else:
        out = x
        torch.ops._C.fused_add_rms_norm(out, res, w, eps)
    torch.cuda.synchronize()
    return _u16(out), _u16(res)


def _alt_words(x: np.ndarray, w: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return D.f2fp_bf16(D.mul_rn(D.mul_rn(D.bf16_to_f32_words(x), scale[:, None]), D.bf16_to_f32_words(w)))


def alternatives(kind: str, x: np.ndarray, res: np.ndarray, w: np.ndarray, eps: float, block: int,
                 model: dict[str, np.ndarray]) -> dict[str, int]:
    """Word mismatches to the three nearby models the planted rows are required to refute."""
    n = x.shape[1]
    row = x if kind == "rms_norm" else model["residual"]
    # The batch-invariant block choice.  For fused this is still the width-8 tree, a conservative nearby alternative;
    # the actual BI generic kernel differs even earlier, at each thread's scalar ownership.
    bi_block = min(n // M.VECTOR if kind == "rms_norm" else n, M.LARGE_BLOCK)
    bi = (R.rms_norm_rows(x, w, eps, bi_block)["out"] if kind == "rms_norm" else
          R.fused_add_rms_norm_rows(x, res, w, eps, bi_block)["out"])

    xf = D.as_f32(D.bf16_to_f32_words(row)).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        linear_scale = D.as_u32((1.0 / np.sqrt((np.sum(xf * xf, axis=1) / n + eps).astype(np.float32))).astype(np.float32))
    linear = _alt_words(row, w, linear_scale)

    arg = D.add_rn(D.div_rn(R.cub_sum(
        R._pure_partials(D.bf16_to_f32_words(x), block) if kind == "rms_norm" else
        R._fused_partials(D.bf16_to_f32_words(model["residual"]), block)),
        np.full(len(x), P.bits_of(float(n)), np.uint32)), np.full(len(x), P.bits_of(eps), np.uint32))
    with np.errstate(invalid="ignore", divide="ignore"):
        sqrt_scale = D.as_u32((np.float32(1.0) / np.sqrt(D.as_f32(arg))).astype(np.float32))
    sqrt_div = _alt_words(row, w, sqrt_scale)
    return {"batch_invariant_block": int(np.count_nonzero(bi != model["out"])),
            "linear_f64_reduction": int(np.count_nonzero(linear != model["out"])),
            "sqrt_div_epilogue": int(np.count_nonzero(sqrt_div != model["out"]))}


def one_case(work: Path, kind: str, m: int, n: int) -> dict[str, Any]:
    eps = EPS_BY_WIDTH[n]
    x, res, w = operands(m, n, 100000 * n + 10 * m + (kind != "rms_norm"))
    block = M.launch_block(m, n, fused=kind != "rms_norm")
    trace = work / f"{kind}-m{m}-n{n}.json"
    got, got_res = _profiled(kind, x, res, w, eps, trace)
    again, again_res = _plain(kind, x, res, w, eps)
    model = (R.rms_norm_rows(x, w, eps, block) if kind == "rms_norm" else
             R.fused_add_rms_norm_rows(x, res, w, eps, block))
    launch = _profiler_launch(trace, "rms_norm_kernel" if kind == "rms_norm" else "fused_add_rms_norm_kernel")
    block_x = launch["block"][0] if isinstance(launch.get("block"), list) and launch["block"] else launch.get("block")
    grid_x = launch["grid"][0] if isinstance(launch.get("grid"), list) and launch["grid"] else launch.get("grid")
    alt = alternatives(kind, x, res, w, eps, block, model)
    residual_ok = True if kind == "rms_norm" else np.array_equal(got_res, model["residual"])
    return {"kind": kind, "M": m, "N": n, "eps": eps, "block": block, "vector": M.VECTOR, "launch": launch,
            "launch_ok": launch["found"] and block_x == block and grid_x == m,
            "words": int(got.size + (got_res.size if kind != "rms_norm" else 0)),
            "mismatch": int(np.count_nonzero(got != model["out"])),
            "residual_mismatch": 0 if residual_ok else int(np.count_nonzero(got_res != model["residual"])),
            "deterministic": np.array_equal(got, again) and
                             (kind == "rms_norm" or np.array_equal(got_res, again_res)),
            "alternatives": alt,
            "output_sha256": hashlib.sha256(got.tobytes()).hexdigest(),
            "residual_sha256": hashlib.sha256(got_res.tobytes()).hexdigest() if kind != "rms_norm" else None}


def _verdict(doc: dict[str, Any]) -> bool:
    cases = doc.get("cases") or []
    refuted = {name: sum(c["alternatives"][name] for c in cases) for name in (
        "batch_invariant_block", "linear_f64_reduction", "sqrt_div_epilogue")}
    return bool(doc.get("gate", {}).get("ok") and doc.get("sass", {}).get("ok") and
                len(cases) == len(KINDS) * len(MS) * len(WIDTHS) and
                all(c["launch_ok"] and c["mismatch"] == c["residual_mismatch"] == 0 and c["deterministic"] and
                    all(v >= 0 for v in c["alternatives"].values()) for c in cases) and
                all(v > 0 for v in refuted.values()) and not doc.get("errors"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--min-driver", type=int, default=575)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = C.gate(a.min_driver, a.want_cc, a.want_sms)
    if not gate["ok"]:
        print(f"GLM53-CUDA-RMSNORM GATE-FAIL {gate}")
        return 3
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    doc: dict[str, Any] = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "argv": sys.argv, "gate": gate, "environment": {"VLLM_BATCH_INVARIANT": "0"},
                           "sass": sass(out), "cases": [], "errors": {}}
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        work = Path(td)
        for n in WIDTHS:
            for m in MS:
                for kind in KINDS:
                    key = f"{kind}/M{m}/N{n}"
                    try:
                        case = one_case(work, kind, m, n)
                        doc["cases"].append(case)
                        print(f"[cuda-rmsnorm] {key} block={case['block']} mismatch={case['mismatch']} "
                              f"residual={case['residual_mismatch']} deterministic={case['deterministic']} launch={case['launch_ok']}")
                    except Exception as e:  # noqa: BLE001 -- preserve every case error in the record
                        doc["errors"][key] = f"{type(e).__name__}: {e}"
                        print(f"[cuda-rmsnorm] {key}: {doc['errors'][key]}", file=sys.stderr)
    sealed = seal_record(doc, _verdict(doc))
    (out / RECORD).write_text(json.dumps(sealed, indent=1) + "\n")
    verdict = "GO" if sealed["ok"] else "NO-GO"
    print(f"GLM53-CUDA-RMSNORM {verdict} cases={len(doc['cases'])} errors={len(doc['errors'])} digest={sealed['digest']}")
    return 0 if sealed["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
