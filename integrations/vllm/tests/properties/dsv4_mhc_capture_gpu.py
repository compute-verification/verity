"""DeepSeek-V4-Pro's hyper-connections (mHC) on Hopper against their Programs: the H200's go/no-go for binding `registry.mhc`
(`MhcFusedPostPre_v1` at <= 16 tokens, `MhcPostPreTf32_v1` above, `MhcPreBroadcast_v1` on layer 0 and `HcHead_v1`) at
DeepSeek-V4-Pro's statics on one TP8 rank (mHC is replicated: each rank runs it on the full hidden 7168), and the probe that pins
the TF32 wgmma step DeepGEMM's `tf32_hc_prenorm_gemm` issues (`HopperTf32WgmmaDot8_v1`, unprobed on sm_90: core's
`instructions.py`).

One GPU process writes `<out>/dsv4_mhc_capture.json`, the property record `dsv4_mhc_capture` (`properties.record`), and
TileLang's generated CUDA of the mHC kernels under `<out>/cuda/`; it prints one line per phase, a `DSV4-MHC TF32` line, then
`DSV4-MHC-CAPTURE GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else runs when it fails
(exit 3).  vLLM's mHC wrappers run as DeepSeek-V4-Pro's decoder layer calls them (deepseek_v4/nvidia/model.py:1186-1267, 1471),
DeepGEMM enabled (`VLLM_USE_DEEP_GEMM=1`, vLLM's default), with the Triton, DeepGEMM and TileLang caches in the scratch directory.
A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  stack      the versions (torch, Triton, TileLang and vLLM's TileLang pass configs, DeepGEMM); `is_deep_gemm_supported()`, so
             above 16 tokens the served GEMM is DeepGEMM's and not `_tilelang_hc_prenorm_gemm`; vLLM's `compute_num_split` on this
             device equals `mhc_twin.tf32_num_split` at every token count here.
  mufu       the special-function words the Definitions read: `rsqrt.approx.ftz.f32` on every word of [1, 4) against core's RSQ
             table (captured on sm_89) and `rsqrt.approx.f32` on a sweep and the specials against `RsqrtApprox_v1`;
             `ex2.approx.ftz.f32` on [0, 1) against the EX2 table; `torch.exp` (libdevice `expf`, no fast-math) on a sweep
             against `NvExpf_v1` (the comparison case mufu/expf).
  probe      `tf32_hc_prenorm_gemm` itself on crafted operands (`probe_launches`, twice each): which k8 step model of the grid
             (`step_models`), which reading of the f32 B operand (`mhc_twin.B_ROUNDINGS`) and which contraction of the squared
             sum (`mhc_twin.SQR_FORMS`) the words are (below).
  fused      `mhc_fused_post_pre_tilelang` at T in FUSED_T (the `use_small_fma` path: `mhc_fused_tilelang` over 4 splits of H,
             then `mhc_pre_big_fuse_with_norm_tilelang`), twice: residual', the split partials (read where the big fuse takes
             them), post', comb' and the layer input against `mhc_twin.fused_post_pre` (`MhcFusedPostPre_v1`).
  tf32       the same wrapper at T in TF32_T (`mhc_post_tilelang`, DeepGEMM over `tf32_num_split(T, 28672)` splits, the big
             fuse) against `mhc_twin.post_pre_tf32` on the candidate step (`MhcPostPreTf32_v1`).
  broadcast  `mhc_pre_broadcast_tilelang` (layer 0, DeepGEMM at every token count) at T in BROADCAST_T on the model's own
             `fn_broadcast` (`fn.view(24, 4, H).sum(1)`, model.py:1655): residual_out the four copies of x, and the partials,
             post, comb and layer input against `mhc_twin.pre_broadcast` (`MhcPreBroadcast_v1`).
  head       `hc_head_fused_kernel_tilelang` at T in HEAD_T against `mhc_twin.hc_head` (`HcHead_v1`).
  sources    TileLang's generated CUDA (`get_kernel_source`) of the five kernels, copied to `<out>/cuda/`: each compiled, and
             none with a fast-math intrinsic.
  bind       the Programs bind at the served statics.

Operands are random at the served statics (`weights`, `tokens`): bf16 streams and sublayer outputs over a spread of scales, a
zero token and a token scaled by 2^20, post mixes in (0, 2), comb rows normalised, `fn` f32 with every low bit live (so the B
operand's reading shows).  The reference: on the first and the last token of every case, the pieces through the evaluator
(`verity.ml.circuits.evaluation`) at the served statics, each on the kernel's own words: `MhcFusedSplit_v1` per split,
`MhcPost_v1`, `MhcPrenormTf32_v1`, `MhcPreBigFuse_v1` / `MhcPreBigFuseBroadcast_v1`, `HcHead_v1`.

The TF32 step.  The A operand is bf16 widened to f32 (already exact in TF32, so its conversion mode is not observable here);
B is `fn`'s f32 words read from shared memory, which the k8 step rounds RNE to TF32 (the probe also tests `trunc` and `rna`).
`HopperTf32WgmmaDot8_v1`
is a candidate: the sm_90 BF16 step's adder (one group of 8, 26 bits, exponent floor -133) on exact products of the TF32 codes.
The probe's families: `round` (one-hot A rows: each word the B operand as the step reads it, whatever the adder), `align`,
`cancel` and `tiny` (products over a spread of exponents, cancelling pairs within a step, products near f32's subnormals), `floor`
(products below 2^-139 into a zero accumulator, every split's first step: half the rows random, half two products that sum to
2^-149 exactly when the adder keeps bit 2^t and truncate to +0 when it doesn't, t in [-160, -155], so the width and floor show
as `floor - width + 1 <= t`), `sqr` (one `x[k]^2 + x[k+4]^2` per split, one square an inexact f32 subnormal and the other just
above 2^-126: the three contractions differ only in such a term's rounding) and `special` (inf, NaN, signed zeros, overflow);
all but `round` have B words exact in TF32, so they pin the adder alone.  `TF32` names the fits: GO when the step is the candidate
alone, B is `rne` and the squared sum `lo` (GPU FMUL/FFMA/FADD, including its NaN word); otherwise the line names what the
Definitions must become.

GO needs every case's words compared with 0 mismatches, the reference on every case with 0 mismatches, deterministic reruns,
the stack, mufu, probe, sources and bind checks, and no errors.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_mhc_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity.ml.circuits.silicon.models import GroupSum
from verity_vllm.program.kernels import mhc_twin as W
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-mhc-capture/v1"
RECORD = "dsv4_mhc_capture.json"
#: DeepSeek-V4-Pro @ b5968e91 (config.json: hidden_size 7168, hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6, rms_norm_eps 1e-6;
#: model.py:1130 hc_post_alpha 2.0) and the kernels' own constants (mhc_fused n_thr 256, hc_head n_thr 128, the norm tile 64)
HC, H, REPEAT, EPS, HC_EPS, POST_MULT = 4, 7168, 20, 1e-6, 1e-6, 2.0
OUT = HC * (2 + HC)
TH, TH_HEAD, TN, NS_SMALL, SMS = 256, 128, 64, 4, 132
STATICS = {"HC": HC, "H": H, "TH": TH, "TH_HEAD": TH_HEAD, "TN": TN, "REPEAT": REPEAT, "EPS": EPS, "HC_EPS": HC_EPS}
FUSED_T = (1, 2, 7, 8, 9, 15, 16)
TF32_T = (17, 64, 65, 128, 129)
BROADCAST_T = (1, 16, 64, 65)
HEAD_T = (1, 7, 16, 64, 129)
QUICK = {"fused": (1, 16), "tf32": (17, 65), "broadcast": (1,), "head": (1, 16)}
PHASES = ("fused", "tf32", "broadcast", "head")
SEED = 20261013
TOTALS = ("words", "mismatch", "reference", "ref_words", "ref_mismatch")
#: the step-model grid: groups of products (in k order; the accumulator joins the first), adder width, exponent floor (a floor
#: matters only where every term of a group lies below it, so the 8-group-of-1 models take one)
STEP_GROUPS = ((8,), (4, 4), (2, 2, 2, 2), (1,) * 8)
STEP_WIDTHS = (24, 25, 26, 27, 28)
STEP_FLOORS = (-133, -139, -132)
PROBE_FAMILIES = ("round", "align", "align", "cancel", "cancel", "tiny", "floor", "floor", "sqr", "special")
PROBE_M, PROBE_N = 64, OUT
EXPF_SWEEP, RSQ_SWEEP, JOB_EXPF = 1 << 20, 1 << 20, 1 << 16
FAST_MATH = re.compile(r"\b(__expf|__exp10f|__logf|__log2f|__powf|__fdividef|__sinf|__cosf|__tanf)\s*\(")
CUDA_COUNT = re.compile(r"\b(expf|rsqrtf|fmaxf|__shfl_xor_sync|__fdiv_rn|__frcp_rn|__float2bfloat16_rn)\s*\(")
KERNELS = ("mhc_fused_tilelang", "mhc_pre_big_fuse_with_norm_tilelang", "mhc_pre_big_fuse_broadcast_with_norm_tilelang",
           "mhc_post_tilelang", "hc_head_fuse_tilelang")


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(work: Path) -> dict[str, str]:
    """DeepSeek-V4-Pro's sm_90 engine env (DeepGEMM on, vLLM's default), the caches in scratch; set before vLLM is imported."""
    env = {"VLLM_USE_DEEP_GEMM": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache"),
           "DG_JIT_CACHE_DIR": str(work / "deep_gemm"), "TILELANG_CACHE_DIR": str(work / "tilelang")}
    os.environ.update(env)
    return env


def token_counts(quick: bool) -> dict[str, tuple[int, ...]]:
    return dict(QUICK) if quick else {"fused": FUSED_T, "tf32": TF32_T, "broadcast": BROADCAST_T, "head": HEAD_T}


def stack(plan: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """What the kernels of record came from, DeepGEMM's selection and vLLM's split counts on this device against the twin's."""
    import tilelang
    import torch
    import triton
    from vllm.model_executor.kernels.mhc import tilelang_kernels as TK
    from vllm.tilelang_utils import _get_pass_configs
    from vllm.utils import deep_gemm as VDG
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    rec: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                           "tilelang": getattr(tilelang, "__version__", None),
                           "pass_configs": {str(k): v for k, v in _get_pass_configs().items()},
                           "deep_gemm_supported": bool(is_deep_gemm_supported()),
                           "sms": torch.cuda.get_device_properties(0).multi_processor_count}
    try:
        dg = VDG._import_deep_gemm()
        rec["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None)}
    except Exception as e:  # noqa: BLE001
        rec["deep_gemm"] = {"error": _error(e)}
    splits = {}
    for T in sorted(set(plan["tf32"]) | set(plan["broadcast"]) | {1, 16, 17, 256, 257, 1024}):
        for K in (HC * H, H):
            splits[f"T{T}/K{K}"] = [int(TK.compute_num_split(64, K, -(-T // 64))), W.tf32_num_split(T, K, SMS)]
    rec["splits"] = splits
    rec["ok"] = bool(rec["deep_gemm_supported"] and rec["sms"] == SMS and all(a == b for a, b in splits.values()))
    return rec


# ---------------------------------------------------------------------------------------------------------------- word helpers
def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


def _bf16_dev(a: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).cuda()


def _f32_dev(a: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint32).view(np.int32)).view(torch.float32).cuda()


def _save(d: Path, **arrays: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))
        out[k] = str(d / f"{k}.npy")
    return out


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def _savez(outdir: Path, name: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A custody-preserved compressed replay file and its identity."""
    path = outdir / "replay" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return {"file": str(path.relative_to(outdir)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _chunks(n: int, step: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def bf16_words(v) -> np.ndarray:
    """RNE of f32 values to bf16 words."""
    u = np.asarray(v, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32_words(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).view(np.uint32).copy()


def splitmix(x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


# ------------------------------------------------------------------------------------------------------------------- operands
def weights(seed: int, hidden: int = H, hc: int = HC) -> dict[str, np.ndarray]:
    """One layer's mHC weights at random: fn [OUT, HC H] and the head's fnh [HC, HC H] f32 (every low bit live), the model's
    fn_broadcast is the GPU's sum (`broadcast_weight`), scale [3] and base [OUT], the head's scale_h [1] and base_h [HC], the
    norm weight w [H] bf16."""
    rng = np.random.default_rng(seed)
    out = hc * (2 + hc)
    return {"fn": f32_words(rng.standard_normal((out, hc * hidden)) * 2.0 ** -7),
            "fnh": f32_words(rng.standard_normal((hc, hc * hidden)) * 2.0 ** -7),
            "scale": f32_words(rng.uniform(0.5, 2.0, 3)), "base": f32_words(rng.standard_normal(out) * 0.5),
            "scale_h": f32_words(rng.uniform(0.5, 2.0, 1)), "base_h": f32_words(rng.standard_normal(hc) * 0.5),
            "w": bf16_words(1 + 0.2 * rng.standard_normal(hidden))}


def tokens(seed: int, T: int, hidden: int = H, hc: int = HC) -> dict[str, np.ndarray]:
    """T tokens' mHC inputs: x [T, H] and residual [T, HC, H] bf16 over scales 2^-2 .. 2^3 (token 1 zero, token 2 x 2^20), post
    [T, HC] in (0, 2) and comb [T, HC, HC] with rows summing to about 1, f32."""
    rng = np.random.default_rng(seed)
    s = np.exp2(rng.uniform(-2, 3, (T, 1)))
    x = rng.standard_normal((T, hidden)) * s
    r = rng.standard_normal((T, hc, hidden)) * s[:, :, None] * np.exp2(rng.uniform(-1, 1, (T, hc, 1)))
    if T >= 2:
        x[1], r[1] = 0, 0
    if T >= 3:
        x[2], r[2] = x[2] * 2.0 ** 20, r[2] * 2.0 ** 20
    comb = rng.random((T, hc, hc)) + 0.05
    return {"x": bf16_words(x), "residual": bf16_words(r), "post": f32_words(2 / (1 + np.exp(-rng.standard_normal((T, hc))))),
            "comb": f32_words(comb / comb.sum(-1, keepdims=True))}


# ------------------------------------------------------------------------------------------------------------- the TF32 probe
def model_name(m: GroupSum) -> str:
    return f"g{'+'.join(map(str, m.groups))}_w{m.width}_f{-m.zero_exponent}"


def step_models() -> list[GroupSum]:
    """The grid of k8 step models the probe tells apart; `mhc_twin.CANDIDATE` is one of them."""
    return [GroupSum(name=f"probe_{len(g)}x{g[0]}_{w}_{-f}", arch="sm_90", dtype="tf32", operand_bits=19, groups=g, width=w,
                     zero_exponent=f, validated="probe grid") for g in STEP_GROUPS for w in STEP_WIDTHS
            for f in (STEP_FLOORS if len(g) < 8 else STEP_FLOORS[:1])]


CANDIDATE_NAME = model_name(W.CANDIDATE)


def _pow2_bf16(sign, e) -> np.ndarray:
    """±2^e as bf16 words (e >= -133)."""
    e = np.asarray(e, dtype=np.int64)
    w = np.where(e >= -126, (e + 127).clip(1, 254) << 7, 1 << (e + 133).clip(0, 6))
    return (w | (np.asarray(sign, dtype=np.int64) << 15)).astype(np.uint16)


def _f32_parts(sign, e, mant) -> np.ndarray:
    """±2^e (1 + mant 2^-23) as f32 words (-126 <= e <= 127)."""
    w = ((np.asarray(e, dtype=np.int64) + 127).clip(1, 254) << 23) | np.asarray(mant, dtype=np.int64)
    return (w | (np.asarray(sign, dtype=np.int64) << 31)).astype(np.uint32)


def _b_tf32(rng, shape, lo: int, hi: int) -> np.ndarray:
    """Random f32 words exact in TF32 (low 13 bits zero), exponents in [lo, hi)."""
    return _f32_parts(rng.integers(0, 2, shape), rng.integers(lo, hi, shape), rng.integers(0, 1 << 10, shape) << 13)


def probe_launch(rng: np.random.Generator, family: str) -> dict[str, Any]:
    """One probe launch of `tf32_hc_prenorm_gemm`: a [PROBE_M, K] bf16, b [PROBE_N, K] f32, NS splits (module docstring)."""
    M, N, K, NS = PROBE_M, PROBE_N, 64, 1
    if family == "round":
        a = np.zeros((M, K), np.uint16)
        a[np.arange(M), np.arange(M) % K] = 0x3F80
        low = rng.choice(np.array([0x1000, 0x1FFF, 0x0FFF, 0x1001, 0x0001, 0x1800]), (N, K))
        low = np.where(rng.random((N, K)) < 0.25, rng.integers(0, 1 << 13, (N, K)), low)
        b = _f32_parts(rng.integers(0, 2, (N, K)), rng.integers(-4, 4, (N, K)), (rng.integers(0, 1 << 10, (N, K)) << 13) | low)
    elif family == "align":
        a = _pow2_bf16(rng.integers(0, 2, (M, K)), rng.integers(-24, 4, (M, K)))
        a[rng.random((M, K)) < 0.3] = 0
        b = _b_tf32(rng, (N, K), -24, 4)
    elif family == "cancel":
        a = np.zeros((M, K), np.uint16)
        b = _b_tf32(rng, (N, K), -30, -8)
        for n in range(N):
            for s in range(K // 8):
                i, j = rng.choice(8, 2, replace=False)
                b[n, 8 * s + i] = b[n, 8 * s + j] = _b_tf32(rng, (), 0, 4)
        for m in range(M):
            for s in range(K // 8):
                i, j = rng.choice(8, 2, replace=False)
                a[m, 8 * s + i], a[m, 8 * s + j] = 0x3F80, 0xBF80
                for k in range(8):
                    if k not in (i, j) and rng.random() < 0.6:
                        a[m, 8 * s + k] = _pow2_bf16(rng.integers(0, 2), rng.integers(-6, 3))
    elif family == "tiny":
        a = _pow2_bf16(rng.integers(0, 2, (M, K)), rng.integers(-75, -45, (M, K)))
        a[rng.random((M, K)) < 0.3] = 0
        b = _b_tf32(rng, (N, K), -110, -70)
    elif family == "floor":
        K, NS = 512, 8
        a = np.zeros((M, K), np.uint16)
        b = _b_tf32(rng, (N, K), -98, -78)
        half = M // 2
        for s in range(NS):
            k = 64 * s
            a[:half, k:k + 2] = _pow2_bf16(rng.integers(0, 2, (half, 1)), -60)
            a[half:, k:k + 8] = _pow2_bf16(rng.integers(0, 2, (M - half, 8)), rng.integers(-62, -59, (M - half, 1)))
            t = -160 + (np.arange(N) + s) % 6
            sign = rng.integers(0, 2, N)
            b[:, k] = _f32_parts(sign, -91, (1 << (t + 161)) << 13)
            b[:, k + 1] = _f32_parts(sign, -90, (512 - (1 << (t + 160))) << 13)
    elif family == "sqr":
        K, NS = 512, 8
        a = np.zeros((M, K), np.uint16)
        for s in range(NS):
            small = rng.integers(0, 2, M) << 15 | rng.integers(0, 128, M) | (rng.integers(-75, -66, M) + 127) << 7
            big = rng.integers(0, 2, M) << 15 | rng.integers(0, 128, M) | (rng.integers(-62, -60, M) + 127) << 7
            swap = rng.integers(0, 2, M).astype(bool)
            k = 64 * s + (np.arange(M) + s) % 4
            a[np.arange(M), k], a[np.arange(M), k + 4] = np.where(swap, small, big), np.where(swap, big, small)
        b = _b_tf32(rng, (N, K), -10, 10)
    elif family == "special":
        a = bf16_words(rng.standard_normal((M, K)))
        sp_a = np.array([0x7F80, 0xFF80, 0x7FC0, 0x8000, 0x0000, 0x0001, 0x7F7F, 0xFF7F], np.uint16)
        a[np.arange(M), rng.integers(0, K, M)] = sp_a[np.arange(M) % sp_a.size]
        b = _b_tf32(rng, (N, K), -4, 4)
        sp_b = np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0x80000000, 0x00000000, 0x00002000, 0x7F7FE000, 0xFF7FE000],
                        np.uint32)
        b[np.arange(N), rng.integers(0, K, N)] = sp_b[np.arange(N) % sp_b.size]
    else:
        raise ValueError(f"no probe family {family!r}")
    return {"family": family, "a": a, "b": b, "NS": NS}


def probe_launches(seed: int = SEED) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [probe_launch(rng, f) for f in PROBE_FAMILIES]


def probe_tables(launches: list[dict[str, Any]], models: list[GroupSum] | None = None) -> dict[str, dict[str, list]]:
    """The probe's words under every hypothesis: "step" {model name: yp [M, NS, N] per launch (None for `round`)}, "rounding"
    {B reading: yp of each launch on the candidate step (the `round` family's are the step's B words)}, "sqrsum" {form: rp
    [M, NS] per launch}."""
    models = step_models() if models is None else models
    step = {model_name(m): [None if L["family"] == "round" else W.prenorm_tf32(L["a"], L["b"], L["NS"], m)[0] for L in launches]
            for m in models}
    rounding = {r: [W.prenorm_tf32(L["a"], L["b"], L["NS"], W.CANDIDATE, r)[0] if L["family"] == "round" else None
                    for L in launches] for r in W.B_ROUNDINGS}
    sqrsum = {s: [W.prenorm_sqrsum(L["a"], L["NS"], s) for L in launches] for s in W.SQR_FORMS}
    return {"step": step, "rounding": rounding, "sqrsum": sqrsum}


def _misfit(preds: list, got: list, key: str) -> int:
    return int(sum(int(np.count_nonzero(np.asarray(p) != np.asarray(g[key]))) for p, g in zip(preds, got) if p is not None))


def probe_fit(launches: list[dict[str, Any]], tables: dict[str, dict[str, list]], got: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Which hypotheses the GPU's words ``got`` (per launch: yp [M, NS, N], rp [M, NS]) are: each axis's names with no mismatch,
    the nearest three steps by mismatches, and the candidate's mismatches by family."""
    step = {k: _misfit(v, got, "yp") for k, v in tables["step"].items()}
    rounding = {k: _misfit(v, got, "yp") for k, v in tables["rounding"].items()}
    sqrsum = {k: _misfit(v, got, "rp") for k, v in tables["sqrsum"].items()}
    fams: dict[str, int] = Counter()
    for L, p, g in zip(launches, tables["step"].get(CANDIDATE_NAME, []), got):
        if p is not None:
            fams[L["family"]] += int(np.count_nonzero(np.asarray(p) != g["yp"]))
    rec = {"step": [k for k, v in step.items() if v == 0], "rounding": [k for k, v in rounding.items() if v == 0],
           "sqrsum": [k for k, v in sqrsum.items() if v == 0], "nearest": sorted(step.items(), key=lambda kv: kv[1])[:3],
           "rounding_mismatch": rounding, "sqrsum_mismatch": sqrsum, "candidate_by_family": dict(fams),
           "words": int(sum(np.asarray(g["yp"]).size + np.asarray(g["rp"]).size for g in got))}
    rec["ok"] = rec["step"] == [CANDIDATE_NAME] and rec["rounding"] == ["rne"] and rec["sqrsum"] == ["lo"]
    return rec


def gemm(a: np.ndarray, b: np.ndarray, NS: int) -> dict[str, np.ndarray]:
    """`tf32_hc_prenorm_gemm` on a [M, K] bf16 and b [N, K] f32 words over NS splits: yp [M, NS, N], rp [M, NS]."""
    import torch
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    M, N = a.shape[0], b.shape[0]
    d = torch.empty(NS, M, N, dtype=torch.float32, device="cuda")
    s = torch.empty(NS, M, dtype=torch.float32, device="cuda")
    tf32_hc_prenorm_gemm(_bf16_dev(a), _f32_dev(b), d, s, NS)
    torch.cuda.synchronize()
    return {"yp": _u32(d).transpose(1, 0, 2).copy(), "rp": _u32(s).T.copy()}


def phase_probe(outdir: Path) -> dict[str, Any]:
    launches = probe_launches()
    got, again = [gemm(L["a"], L["b"], L["NS"]) for L in launches], [gemm(L["a"], L["b"], L["NS"]) for L in launches]
    det = all(np.array_equal(x[k], y[k]) for x, y in zip(got, again) for k in ("yp", "rp"))
    rec = probe_fit(launches, probe_tables(launches), got)
    rec["deterministic"] = det
    rec["ok"] = bool(rec["ok"] and det)
    arrays = {}
    for i, (launch, words) in enumerate(zip(launches, got)):
        arrays.update({f"p{i}_a": launch["a"], f"p{i}_b": launch["b"], f"p{i}_yp": words["yp"], f"p{i}_rp": words["rp"]})
    rec["replay"] = {**_savez(outdir, "probe.npz", arrays),
                     "launches": [{"index": i, "family": launch["family"], "NS": launch["NS"]}
                                  for i, launch in enumerate(launches)]}
    return rec


# ------------------------------------------------------------------------------------------------------------------------ mufu
def _asm(src: str, x):
    """One PTX instruction on every word of a CUDA f32 tensor (Triton inline asm, as `registry.mla_capture` probes MUFU)."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def k(X, Y, n, BLOCK: tl.constexpr, ASM: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(X + i, mask=i < n)
        y = tl.inline_asm_elementwise(ASM, "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + i, y, mask=i < n)

    y = torch.empty_like(x)
    k[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024, ASM=src)
    return y


def mufu_inputs() -> dict[str, np.ndarray]:
    j = np.arange(1 << 23, dtype=np.uint32)
    specials = np.array([0, 0x80000000, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xBF800000,
                         0x3F800000, 0x00400000, 0x0000F000], dtype=np.uint32)
    sweep = splitmix(np.arange(RSQ_SWEEP, dtype=np.uint64) + (np.uint64(0xD54) << np.uint64(40))).astype(np.uint32)
    rng = np.random.default_rng(SEED)
    expf = np.concatenate([f32_words(rng.uniform(-110, 92, EXPF_SWEEP // 2)), sweep[:EXPF_SWEEP // 2], specials])
    return {"rsq_table": np.concatenate([np.uint32(0x3F800000) | j, np.uint32(0x40000000) | j]),
            "rsq_sweep": np.concatenate([sweep, specials, np.arange(1 << 16, dtype=np.uint32) * 128]),
            "ex2_table": f32_words(j.astype(np.float64) / (1 << 23)), "expf": expf}


def phase_mufu(work: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The tables and sweeps on this GPU (module docstring); the expf sweep's words go to comparison jobs."""
    import torch

    from verity.ml.circuits.ops import mufu as MU
    from verity_vllm.program.kernels import derived_rows as DR

    x = mufu_inputs()

    def run(src: str, words: np.ndarray) -> np.ndarray:
        return _u32(_asm(src, _f32_dev(words)))

    rec: dict[str, Any] = {}
    for name, src, words, want in (("rsq_ftz_table", "rsqrt.approx.ftz.f32 $0, $1;", x["rsq_table"], MU.table("rsq")),
                                   ("rsqrt_approx", "rsqrt.approx.f32 $0, $1;", x["rsq_sweep"], DR.rsqrt_approx(x["rsq_sweep"])),
                                   ("ex2_ftz_table", "ex2.approx.ftz.f32 $0, $1;", x["ex2_table"], MU.table("ex2"))):
        y = run(src, words)
        bad = np.flatnonzero(y != np.asarray(want, dtype=np.uint32))
        rec[name] = {"words": int(y.size), "mismatches": int(bad.size),
                     "first": [[int(words[i]), int(y[i]), int(want[i])] for i in bad[:8]]}
    y = _u32(torch.exp(_f32_dev(x["expf"])))
    paths = _save(work / "mufu", x=x["expf"], got=y)
    rec["ok"] = all(v["words"] > 0 and v["mismatches"] == 0 for v in rec.values())
    jobs = [{"phase": "mufu", "case": "expf", "kind": "expf", "paths": paths, "range": list(r)} for r in _chunks(y.size, JOB_EXPF)]
    return rec, jobs


# ------------------------------------------------------------------------------------------------------------------ the layers
class Spy:
    """Stands in for a TileLang kernel of `tilelang_kernels` while the wrapper runs: keeps copies of the split partials it is
    given (its first two arguments, [NS, T, OUT] and [NS, T]) and calls the kernel."""

    def __init__(self, module, name: str):
        self.module, self.name, self.kernel, self.calls = module, name, getattr(module, name), []

    def __enter__(self) -> "Spy":
        setattr(self.module, self.name, self)
        return self

    def __exit__(self, *exc) -> None:
        setattr(self.module, self.name, self.kernel)

    def __call__(self, *args, **kwargs):
        self.calls.append((args[0].clone(), args[1].clone()))
        return self.kernel(*args, **kwargs)


def _partials(spy: Spy) -> dict[str, np.ndarray]:
    yp, rp = spy.calls[-1]
    return {"yp": _u32(yp).transpose(1, 0, 2).copy(), "rp": _u32(rp).T.copy()}


def run_layer(phase: str, wd: dict, td: dict) -> dict[str, np.ndarray]:
    """One served launch of ``phase`` on device operands (weights ``wd``, tokens ``td``): its output words, and the split
    partials where the big fuse reads them (fused, tf32, broadcast)."""
    import torch
    from vllm.model_executor.kernels.mhc import tilelang as TL
    from vllm.model_executor.kernels.mhc import tilelang_kernels as TK

    if phase == "head":
        out = TL.hc_head_fused_kernel_tilelang(td["residual"], wd["fnh"], wd["scale_h"], wd["base_h"], EPS, HC_EPS)
        torch.cuda.synchronize()
        return {"out": _u16(out)}
    if phase == "broadcast":
        with Spy(TK, "mhc_pre_big_fuse_broadcast_with_norm_tilelang") as spy:
            res, post, comb, out = TL.mhc_pre_broadcast_tilelang(
                td["x"], wd["fn"], wd["scale"], wd["base"], EPS, HC_EPS, HC_EPS, POST_MULT, REPEAT, norm_weight=wd["w"],
                norm_eps=EPS, fn_broadcast=wd["fnb"])
    else:
        with Spy(TK, "mhc_pre_big_fuse_with_norm_tilelang") as spy:
            res, post, comb, out = TL.mhc_fused_post_pre_tilelang(
                td["x"], td["residual"], td["post"].unsqueeze(-1), td["comb"], wd["fn"], wd["scale"], wd["base"], EPS, HC_EPS,
                HC_EPS, POST_MULT, REPEAT, n_splits=1, tile_n=1, norm_weight=wd["w"], norm_eps=EPS)
    torch.cuda.synchronize()
    T = out.shape[0]
    return {"res": _u16(res), "out": _u16(out), "post2": _u32(post).reshape(T, HC), "comb2": _u32(comb).reshape(T, HC, HC),
            **_partials(spy)}


def broadcast_weight(fn_dev) -> Any:
    """`fn_broadcast` as the model builds it (model.py:1655): fn [OUT, HC H] summed over the streams on the GPU."""
    return fn_dev.view(OUT, HC, H).sum(dim=1).contiguous()


def _splits_for(phase: str, T: int) -> int | None:
    if phase == "fused":
        return NS_SMALL
    if phase == "tf32":
        return W.tf32_num_split(T, HC * H, SMS)
    if phase == "broadcast":
        return W.tf32_num_split(T, H, SMS)
    return None


def reference_pieces(phase: str, NS: int | None) -> list[dict[str, Any]]:
    if phase == "fused":
        return [{"piece": "split", "s": s} for s in range(NS)] + [{"piece": "bigfuse"}]
    if phase == "tf32":
        return [{"piece": "post"}, {"piece": "prenorm"}, {"piece": "bigfuse"}]
    if phase == "broadcast":
        return [{"piece": "prenorm"}, {"piece": "bigfuse_b"}]
    return [{"piece": "head"}]


def layer_jobs(phase: str, T: int, NS: int | None, paths: dict[str, str], statics: dict[str, Any] = STATICS) -> list[dict[str, Any]]:
    """One case's comparison jobs: every output word against the twin, and the reference pieces on its first and last token."""
    job = {"phase": phase, "case": f"T{T}", "paths": paths, "T": T, "NS": NS, "statics": statics}
    return [{**job, "kind": "layer"}] + [{**job, "kind": "ref", "token": t, **p} for t in sorted({0, T - 1})
                                         for p in reference_pieces(phase, NS)]


def phase_layers(work: Path, outdir: Path,
                 plan: dict[str, tuple[int, ...]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Every served case of every phase, twice; returns (launch records, comparison jobs)."""
    import torch

    wts = weights(SEED)
    wd = {k: (_bf16_dev(v) if v.dtype == np.uint16 else _f32_dev(v)) for k, v in wts.items()}
    wd["fnb"] = broadcast_weight(wd["fn"])
    wts["fnb"] = _u32(wd["fnb"])
    wpaths = _save(work / "weights", **wts)
    launches, jobs, replay = [], [], {}
    for phase in PHASES:
        for T in plan[phase]:
            case = f"T{T}"
            tk = tokens(SEED + 1000 * PHASES.index(phase) + T, T)
            if phase == "broadcast":
                tk = {"x": tk["x"]}
            elif phase == "head":
                tk = {"residual": tk["residual"]}
            td = {k: (_bf16_dev(v) if v.dtype == np.uint16 else _f32_dev(v)) for k, v in tk.items()}
            got, again = run_layer(phase, wd, td), run_layer(phase, wd, td)
            NS = _splits_for(phase, T)
            checks = {"deterministic": all(np.array_equal(got[k], again[k]) for k in got)}
            if "yp" in got:
                checks["splits"] = got["yp"].shape[1] == NS and got["rp"].shape[1] == NS
            if phase == "broadcast":
                checks["copies"] = bool(np.array_equal(got.pop("res"), np.broadcast_to(tk["x"][:, None, :], (T, HC, H))))
            paths = {**wpaths, **_save(work / phase / case, **tk, **got)}
            launches.append({"phase": phase, "case": case, "checks": checks, "info": {"T": T, "NS": NS}})
            jobs += layer_jobs(phase, T, NS, paths)
            if phase in ("tf32", "broadcast"):
                aa = got["res"].reshape(T, -1) if phase == "tf32" else tk["x"]
                bb = wts["fn"] if phase == "tf32" else wts["fnb"]
                off, n = W.M.tf32_ranges(aa.shape[1], NS)[0]
                prefix = f"{phase}_T{T}"
                replay.update({f"{prefix}_a": aa[0, off:off + n], f"{prefix}_b": bb[0, off:off + n],
                               f"{prefix}_yp": np.asarray([got["yp"][0, 0, 0]], np.uint32),
                               f"{prefix}_rp": np.asarray([got["rp"][0, 0]], np.uint32),
                               f"{prefix}_meta": np.asarray([T, NS, off, n], np.int64)})
            del td, got, again
    del wd
    torch.cuda.empty_cache()
    return launches, jobs, _savez(outdir, "layer_samples.npz", replay)


# ---------------------------------------------------------------------------------------------------------------- comparisons
def _tally(rec: dict[str, Any], name: str, want, got) -> None:
    want, got = np.asarray(want).astype(np.int64), np.asarray(got).astype(np.int64)
    if want.shape != got.shape:
        raise ValueError(f"{name}: the twin's shape {want.shape}, the kernel's {got.shape}")
    bad = np.flatnonzero(want.reshape(-1) != got.reshape(-1))
    rec["words"] += int(want.size)
    rec["mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        i = int(bad[0])
        rec["first_mismatch"] = {"output": name, "index": [int(v) for v in np.unravel_index(i, want.shape)],
                                 "want": int(want.reshape(-1)[i]), "kernel": int(got.reshape(-1)[i])}


def twin_outputs(phase: str, a: dict[str, np.ndarray], NS: int | None, S: dict[str, Any]) -> dict[str, np.ndarray]:
    """The twin's words of every output `run_layer` records for ``phase`` (broadcast: all but the residual copies), on the
    case's operands ``a`` at statics ``S``."""
    norm = (S["TN"], S["REPEAT"], S["EPS"], S["HC_EPS"])
    want: dict[str, Any] = {}
    if phase == "fused":
        want["res"], want["yp"], want["rp"] = W.fused_partials(a["x"], a["residual"], a["post"], a["comb"], a["fn"], NS, S["TH"])
        res = want["res"]
    elif phase == "tf32":
        res = want["res"] = W.post(a["x"], a["residual"], a["post"], a["comb"])
        want["yp"], want["rp"] = W.prenorm_tf32(res.reshape(res.shape[0], -1), a["fn"], NS)
    elif phase == "broadcast":
        want["yp"], want["rp"] = W.prenorm_tf32(a["x"], a["fnb"], NS)
        res = a["x"]
    if phase == "head":
        want["out"] = W.hc_head(a["residual"], a["fnh"], a["scale_h"], a["base_h"], S["TH_HEAD"], S["EPS"], S["HC_EPS"])
    else:
        want["out"], want["post2"], want["comb2"] = W.big_fuse(want["yp"], want["rp"], a["scale"], a["base"], res, a["w"], *norm,
                                                               broadcast=phase == "broadcast")
    return want


def compare_layer(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Every output word of one case against the twin at the job's statics."""
    a = _load(job["paths"])
    for name, v in twin_outputs(job["phase"], a, job["NS"], job["statics"]).items():
        _tally(rec, name, v, a[name])


def reference_call(job: dict[str, Any], a: dict[str, np.ndarray]):
    """The piece's Definition at the job's statics and its arguments on token ``job["token"]``, and the kernel's words it
    states."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import mhc as R

    S, NS, t, piece = job["statics"], job["NS"], job["token"], job["piece"]
    hc, h = S["HC"], S["H"]
    out = hc * (2 + hc)
    norm = {"TN": S["TN"], "REPEAT": S["REPEAT"], "EPS": S["EPS"], "HC_EPS": S["HC_EPS"]}
    if piece == "split":
        N, s = h // NS, job["s"]
        sl = slice(N * s, N * s + N)
        d = bind(R.MhcFusedSplit, HC=hc, N=N, OUT=out, TH=S["TH"])
        args = (a["x"][t, sl], a["residual"][t, :, sl], a["post"][t], a["comb"][t], np.asarray(a["fn"]).reshape(out, hc, h)[:, :, sl])
        return d, args, (a["res"][t, :, sl], a["yp"][t, s], a["rp"][t, s])
    if piece == "post":
        return bind(R.MhcPost, HC=hc, N=h), (a["x"][t], a["residual"][t], a["post"][t], a["comb"][t]), (a["res"][t],)
    if piece == "prenorm":
        x, b = (a["res"][t].reshape(-1), a["fn"]) if job["phase"] == "tf32" else (a["x"][t], a["fnb"])
        return bind(R.MhcPrenormTf32, K=x.size, NS=NS, OUT=out), (x, b), (a["yp"][t], a["rp"][t])
    if piece in ("bigfuse", "bigfuse_b"):
        broadcast = piece == "bigfuse_b"
        d = bind(R.MhcPreBigFuseBroadcast if broadcast else R.MhcPreBigFuse, HC=hc, H=h, NS=NS, **norm)
        rows = a["x"][t] if broadcast else a["res"][t]
        return d, (a["yp"][t], a["rp"][t], a["scale"], a["base"], rows, a["w"]), (a["out"][t], a["post2"][t], a["comb2"][t])
    if piece == "head":
        d = bind(R.HcHead, HC=hc, H=h, TH=S["TH_HEAD"], EPS=S["EPS"], HC_EPS=S["HC_EPS"])
        return d, (a["residual"][t], a["fnh"], a["scale_h"], a["base_h"]), (a["out"][t],)
    raise ValueError(f"no reference piece {piece!r}")


def compare_ref(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One piece through the evaluator on one token's kernel words."""
    from verity.ml.circuits.evaluation import evaluate

    d, args, want = reference_call(job, _load(job["paths"]))
    y = np.array(evaluate(d, *[np.asarray(v).reshape(-1).astype(np.int64).tolist() for v in args]), dtype=np.int64)
    w = np.concatenate([np.asarray(v).reshape(-1).astype(np.int64) for v in want])
    rec["reference"] += 1
    rec["ref_words"] += int(w.size)
    if y.shape != w.shape:
        rec["ref_mismatch"] += int(w.size)
        return
    bad = np.flatnonzero(y != w)
    rec["ref_mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {"piece": job["piece"], "token": job["token"], "s": job.get("s"), "word": int(bad[0]),
                                 "want": int(y[bad[0]]), "kernel": int(w[bad[0]])}


def compare_expf(job: dict[str, Any], rec: dict[str, Any]) -> None:
    a = _load(job["paths"])
    lo, hi = job["range"]
    _tally(rec, "expf", W.expf(a["x"][lo:hi]), a["got"][lo:hi])


COMPARE = {"layer": compare_layer, "ref": compare_ref, "expf": compare_expf}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# ------------------------------------------------------------------------------------------------------------ sources and bind
def sources(outdir: Path) -> dict[str, Any]:
    """TileLang's generated CUDA of each compiled specialization of the five kernels (`JITImpl._kernel_cache`), copied to
    `<out>/cuda/`, with the counts of the operations the Definitions state; ok when each kernel compiled and none uses a
    fast-math intrinsic."""
    from vllm.model_executor.kernels.mhc import tilelang_kernels as TK

    dest = outdir / "cuda"
    dest.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"kernels": {}}
    for name in KERNELS:
        entries = []
        for kernel in list(getattr(getattr(TK, name), "_kernel_cache", {}).values()):
            text = kernel.get_kernel_source()
            sha = hashlib.sha256(text.encode()).hexdigest()
            (dest / f"{name}.{sha[:12]}.cu").write_text(text)
            entries.append({"sha256": sha, "counts": dict(Counter(CUDA_COUNT.findall(text))),
                            "fast_math": sorted(set(FAST_MATH.findall(text)))})
        rec["kernels"][name] = entries
    rec["ok"] = all(rec["kernels"][k] and not any(e["fast_math"] for e in rec["kernels"][k]) for k in KERNELS)
    return rec


def served_bindings() -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import mhc as R

    norm = {"REPEAT": REPEAT, "EPS": EPS, "HC_EPS": HC_EPS}
    out = {"HopperTf32WgmmaDot8_v1": lambda: R.HopperTf32WgmmaDot8, "MhcSigmoid_v1": lambda: bind(R.MhcSigmoid),
           "MhcSinkhorn_v1{4,20}": lambda: bind(R.MhcSinkhorn, HC=HC, REPEAT=REPEAT, EPS=HC_EPS),
           "MhcPreNorm_v1{4,7168,64}": lambda: bind(R.MhcPreNorm, HC=HC, H=H, TN=TN, EPS=EPS),
           "MhcPreNormBroadcast_v1{4,7168,64}": lambda: bind(R.MhcPreNormBroadcast, HC=HC, H=H, TN=TN, EPS=EPS),
           "MhcPost_v1{4,7168}": lambda: bind(R.MhcPost, HC=HC, N=H),
           "MhcFusedSplit_v1{4,1792,24,256}": lambda: bind(R.MhcFusedSplit, HC=HC, N=H // NS_SMALL, OUT=OUT, TH=TH),
           "MhcFusedPostPre_v1{4,7168,4,256,64}": lambda: bind(R.MhcFusedPostPre, HC=HC, H=H, NS=NS_SMALL, TH=TH, TN=TN, **norm),
           "MhcPrenormTf32Split_v1{256,24}": lambda: bind(R.MhcPrenormTf32Split, K=256, OUT=OUT),
           "HcHead_v1{4,7168,128}": lambda: bind(R.HcHead, HC=HC, H=H, TH=TH_HEAD, EPS=EPS, HC_EPS=HC_EPS)}
    for K, NS in sorted({(HC * H, W.tf32_num_split(T, HC * H, SMS)) for T in TF32_T} | {(H, W.tf32_num_split(T, H, SMS)) for T in BROADCAST_T}):
        out[f"MhcPrenormTf32_v1{{{K},{NS},24}}"] = lambda K=K, NS=NS: bind(R.MhcPrenormTf32, K=K, NS=NS, OUT=OUT)
        out[f"MhcMixes_v1{{4,{NS},{K}}}"] = lambda K=K, NS=NS: bind(R.MhcMixes, HC=HC, NS=NS, K=K, **norm)
        if K == H:
            out[f"MhcPreBigFuseBroadcast_v1{{4,7168,{NS},64}}"] = lambda NS=NS: bind(R.MhcPreBigFuseBroadcast, HC=HC, H=H, NS=NS,
                                                                                      TN=TN, **norm)
            out[f"MhcPreBroadcast_v1{{4,7168,{NS},64}}"] = lambda NS=NS: bind(R.MhcPreBroadcast, HC=HC, H=H, NS=NS, TN=TN, **norm)
        else:
            out[f"MhcPreBigFuse_v1{{4,7168,{NS},64}}"] = lambda NS=NS: bind(R.MhcPreBigFuse, HC=HC, H=H, NS=NS, TN=TN, **norm)
            out[f"MhcPostPreTf32_v1{{4,7168,{NS},64}}"] = lambda NS=NS: bind(R.MhcPostPreTf32, HC=HC, H=H, NS=NS, TN=TN, **norm)
    out["MhcMixes_v1{4,4,28672}"] = lambda: bind(R.MhcMixes, HC=HC, NS=NS_SMALL, K=HC * H, **norm)
    out["MhcPreBigFuse_v1{4,7168,4,64}"] = lambda: bind(R.MhcPreBigFuse, HC=HC, H=H, NS=NS_SMALL, TN=TN, **norm)
    return out


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics, at every split count served here."""
    out = {}
    for name, make in served_bindings().items():
        try:
            out[name] = bool(make().signature())
        except Exception as e:  # noqa: BLE001
            out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def gpu(work: Path, outdir: Path, plan: dict[str, tuple[int, ...]], doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every GPU phase in order; each phase's error goes to doc["errors"].  Fills doc["stack"], doc["mufu"], doc["probe"],
    doc["launches"]; returns the comparison jobs."""
    jobs: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-mhc] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["stack"] = step("stack", lambda: stack(plan)) or {"ok": False}
    print(f"[dsv4-mhc] stack {json.dumps(doc['stack'])[:1500]}", flush=True)
    r = step("mufu", lambda: phase_mufu(work))
    if r is not None:
        doc["mufu"], j = r
        jobs += j
    print(f"[dsv4-mhc] mufu {json.dumps(doc.get('mufu', {}))[:1500]}", flush=True)
    doc["probe"] = step("probe", lambda: phase_probe(outdir)) or {"ok": False}
    print(f"[dsv4-mhc] probe {json.dumps(doc['probe'])[:2000]}", flush=True)
    r = step("layers", lambda: phase_layers(work, outdir, plan))
    if r is not None:
        doc["launches"] += r[0]
        jobs += r[1]
        doc["replay"]["layers"] = r[2]
        for x in r[0]:
            print(f"[dsv4-mhc] {x['phase']:9s} {x['case']:5s} {json.dumps(x['checks'])} {json.dumps(x['info'])}", flush=True)
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any], reference: bool = True) -> bool:
    return (t["words"] > 0 and t["mismatch"] == 0 and t["ref_mismatch"] == 0
            and (not reference or (t["reference"] > 0 and t["ref_words"] > 0)))


def expected_cases(plan: dict[str, tuple[int, ...]]) -> list[tuple[str, str]]:
    return [("mufu", "expf")] + [(p, f"T{T}") for p in PHASES for T in plan[p]]


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each planned case's verdict (mufu/expf, fused/T1, ...): its launch checks, its words and reference, its errors."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in expected_cases(doc["plan"]):
        key = f"{phase}/{case}"
        launches = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e.startswith(f"compare/{key}")]
        launches_ok = phase == "mufu" or (bool(launches) and all(_launch_ok(x) for x in launches))
        out[key] = {"launches_ok": launches_ok, "totals": t, "errors": errors,
                    "ok": launches_ok and _words_ok(t, reference=phase != "mufu") and not errors}
    return out


def tf32_go(doc: dict[str, Any]) -> bool:
    cs = cases(doc)
    return bool(doc["probe"].get("ok") is True and all(c["ok"] for k, c in cs.items() if k.split("/")[0] in ("tf32", "broadcast")))


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and all(doc[k].get("ok") is True for k in ("stack", "mufu", "probe",
                                                                                                  "sources", "bind"))
                and all(c["ok"] for c in cases(doc).values()) and _words_ok(doc["totals"]))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment(work)
    plan = token_counts(quick)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "plan": plan, "statics": STATICS,
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {},
                           "launches": [], "replay": {}}
    print(f"[dsv4-mhc] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-MHC GATE-FAIL", flush=True)
        raise SystemExit(3)
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, plan, doc)
    for k in ("stack", "mufu", "probe"):
        doc.setdefault(k, {"ok": False})
    for name, fn in (("sources", lambda: sources(outdir)), ("bind", bind_check)):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            doc[name] = {"ok": False}
        print(f"[dsv4-mhc] {name} {json.dumps(doc[name])[:2000]}", flush=True)
    print(f"[dsv4-mhc] comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"].setdefault(f"compare/{res['phase']}/{res['case']}", res["error"])
                continue
            agg = results.setdefault((res["phase"], res["case"]), {"phase": res["phase"], "case": res["case"],
                                                                  **{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        print(f"[dsv4-mhc] words {r['phase']:9s} {r['case']:5s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"ref_words={r['ref_words']} ref!={r['ref_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-mhc] case {key:14s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} ref!={t['ref_mismatch']} errors={c['errors']}", flush=True)
    doc["tf32_go"] = tf32_go(doc)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    p = doc["probe"]
    print(f"DSV4-MHC TF32 {'GO' if doc['tf32_go'] else 'NO-GO'} step={p.get('step')} rounding={p.get('rounding')} "
          f"sqrsum={p.get('sqrsum')} nearest={p.get('nearest')} candidate={CANDIDATE_NAME}", flush=True)
    print(f"DSV4-MHC-CAPTURE {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's mHC kernels and the TF32 wgmma step on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and cuda/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton, DeepGEMM and TileLang caches)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=SMS)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"the token counts {QUICK}: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
