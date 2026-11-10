"""GLM-5.3-FP8's routed experts on Hopper as vLLM d9105ea80 serves them, against their Programs: the H200's go/no-go for every link
of the served fused-MoE chain, `Fp8GroupQuantUe8m0_v1` (the input's quant), `MoeExpertFp8Gemm_v1` (the w13 launch), `SiluMul_v2`
(`silu_and_mul`), `Fp8GroupQuantUe8m0_v1` (the activation's quant), `MoeExpertFp8GemmW_v1` (the w2 launch) and `MoeSum_v1`
(`moe_sum`), at GLM-5.3's shapes and at each launch's served tile and DOT (`fp8_moe.sm90_dot`).

One GPU process writes `<out>/glm53_block_fp8_capture.json`, the property record `glm53_block_fp8_capture` (`properties.record`), and
prints one line per case and then `GLM53-BLOCK-FP8 GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and
nothing else runs when it fails (exit 3).  vLLM runs at its defaults, as a served GLM-5.3 engine does: `engine.env.served_defaults`
clears every switch the process inherited before vLLM is imported, so DeepGEMM is on with its ue8m0 scales, the run is not batch
invariant (GLM-5.3's engine refuses `VLLM_BATCH_INVARIANT` on sm_90: no sparse-MLA backend supports it) and the experts launch vLLM's
tuned H200 configs.  A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  served     vLLM reads the switches back as served: `VLLM_USE_DEEP_GEMM`, `is_deep_gemm_e8m0_used()`, not `VLLM_BATCH_INVARIANT`, no
             `VLLM_TUNED_CONFIG_FOLDER`.
  selection  `select_fp8_moe_backend` with Fp8MoEMethod's keys (TP, no EP) selects TRITON / TritonExperts (oracle/fp8.py:113-123).
  moe        vLLM's own kernel: `make_fp8_moe_quant_config` and `make_fp8_moe_kernel` build the FusedMoEKernel `Fp8MoEMethod` builds
             (`MoEPrepareAndFinalizeNoDPEPModular` around `TritonExperts`), and `kernel.apply` runs it twice (E = 256, top_k 8, hidden
             6144) at one TP8 rank's shard (I = 256) and at whole experts (I = 2048), with uniform and hot routing (`MOE`).  Every
             `fused_moe_kernel` launch is intercepted (its operands, config, and BLOCK_SIZE_K and SWAP_AB as invoke_fused_moe_triton_kernel
             computes them, fused_moe.py:789-846) and the experts' `activation` and `moe_sum` tapped, so each link is compared on what
             the previous one wrote: the w13 launch's A / A_scale (the prepare step's `per_token_group_quant_fp8(x, 128)`, ue8m0) against
             `Fp8GroupQuantUe8m0_v1{6144,128}` of x (twin `deepgemm_twins.ue8m0_group_quant_row`); every (token, slot) row of the w13 launch
             against `MoeExpertFp8BlockGemm_v3{K=6144,N=2I,G=128,KB,SWAP,DOT}` and of the w2 launch against
             `MoeExpertFp8BlockGemmW_v3{K=I,N=6144,...}` (twin `fp8_moe_rows.moe_expert_fp8_block_row` at the launch's DOT); the
             activation's output against `SiluMul_v2{I}` of its input (twin `act_rows.silu_mul_expf_rows`); the w2 launch's A / A_scale
             against `Fp8GroupQuantUe8m0_v1{I,128}` of the activation's output; `moe_sum`'s output against `MoeSum_v1{8,6144}` of its
             input (twin `rows.moe_sum`).  The chain: the activation reads the w13 launch's words, `moe_sum` the w2 launch's, the kernel
             returns `moe_sum`'s, one launch of each per run and both runs the same words (`same_words`); `moe_sum` takes its
             vectorised kernel (`moe_sum_vec_kernel<bf16, int32, 8>`: contiguous, 16-byte aligned).  Every launch's (token, slot) ->
             expert assignment is checked against the routing, the aligned blocks' and the naive one's (M <= 8).
  mmav2      `HopperE4m3MmaV2Dot32_v1` (the decode tiles' DOT: the tuned 8x32x128 tile at M <= 3 and 8x64x128 under SWAP_AB at M 4..6
             run `mma.sync.m16n8k16` on widened e4m3) on operands that decide its model (`mmav2_probe`): the same chain at the decode
             shapes (`MMAV2_PROBE`) with every 32-wide k window of x and w1 planting +448 * +448 at k = 0 and +448 * -448 at k = 2
             (one in each of the two k16 instructions, `fp8_moe_rows.MMAV2_HALVES`) over small codes, so the two big products'
             alignment truncates the small ones and their cancellation shows how.  The w13 words must equal the twin and refute
             every model in `mmav2_alternatives` (the natural k order, the halves swapped, adder widths 24, 25, 27 and 28, Ampere's
             8 + 8 groups, one k32 group, QGMMA's step, a chain of FMAs): each must differ from the kernel on some word.
  sum        `moe_sum` on planted rows (`moe_sum_probe`: all -0, a tie into infinity, overflow, NaN payloads, infinities, ties,
             bf16 subnormals, cancellations whose result depends on the order), twice, every word against `MoeSum_v1`, and every
             model in `moe_sum_alternatives` (descending order, a pairwise tree, the first slot's start, a bf16 accumulator, FTZ,
             NaN payloads kept, truncation) refuted.
  reference  on a sample of the coordinates and rows of every link, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`)
             of `MoeExpertFp8BlockCoordinate(W)_v3{...,DOT}`, `Fp8GroupQuantUe8m0_v1`, `SiluMul_v2` and `MoeSumCoordinate_v1`, which
             the twins must equal.
  ptx        every `fused_moe_kernel` compiled in the run's Triton cache, by the e4m3 `tt.dot` of its TTIR and its warps: a wgmma tile's
             PTX has wgmma, every one `.m64nNk32.f32.e4m3.e4m3`, and no `mma.sync`; an MMAv2 tile's has `mma.sync`, every one
             `.m16n8k16.row.col.f32.f16.f16.f32`, the e4m3 widening `cvt.rn.f16x2.e4m3x2`, and no wgmma; and every launch's tile (the
             dot's M, N and K, and its warps) is among the compiled kernels.
  bind       the Programs bind at every shape and launched statics: `MoeExpertFp8BlockGemm(W)_v3` and `fp8_moe_experts`'
             `MoeExpertFp8Gemm(W)_v1{E=256,...}` with the weight-scale rows each reads, `Fp8GroupQuantUe8m0_v1` at 6144 and I,
             `SiluMul_v2{I}`, `MoeSum_v1{8,6144}`.

vLLM's tuned H200 config for GLM-5.3's TP8 expert shard (fused_moe/configs/`E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,
block_shape=[128,128].json`, looked up at the nearest batch size, fused_moe.py:1442-1447) gives M <= 3 BLOCK_SIZE_M 8 x BLOCK_SIZE_N
32, which SWAP_AB does not take, and M in 4..6 BLOCK_SIZE_M 8 x BLOCK_SIZE_N 64 under SWAP_AB (the dot's N = 8): their DOT is
`HopperE4m3MmaV2Dot32_v1`, every larger batch's and whole experts' `HopperE4m3QgmmaDot32_v1`.

GO needs every required case (`required_cases`) to pass: 0 mismatches and 0 unresolved words in every link (a twin flags a word it
cannot claim: a non-finite accumulator; these operands are finite and none should occur), the references, the chain, deterministic
reruns, every competing model refuted, and the run's served read-back, selection, PTX scan and bindings.  The verdict line counts the
expert launches by DOT.

    cd integrations/vllm; PY=/workspace/cp/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_block_fp8_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--only moe,mmav2,sum]
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import dataclasses
import glob
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
from typing import Any, Callable

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.engine.env import served_defaults
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-block-fp8-capture/v4"
RECORD = "glm53_block_fp8_capture.json"
GROUP = 128
HIDDEN = 6144
NUM_EXPERTS, TOP_K = 256, 8
#: (name, intermediate size I, tokens, hot experts: 0 routes each token to 8 of all 256); the B1 row's decode is tp8_decode, its
#: prefill tp8_prefill
MOE = (("tp8_decode", 256, 1, 0), ("tp8_small", 256, 5, 0), ("tp8_block", 256, 64, 0), ("tp8_prefill", 256, 256, 16),
       ("whole_decode", 2048, 1, 0), ("whole_hot", 2048, 16, 12))
QUICK_MOE = ("tp8_decode", "tp8_small", "whole_decode")
#: the two decode tiles, each its own case of the chain on `mmav2_probe`'s operands
MMAV2_PROBE = (("probe_decode", 256, 1, 0), ("probe_small", 256, 5, 0))
#: `moe_sum_probe`'s tokens
SUM_TOKENS = 8
PHASES = ("moe", "mmav2", "sum")
#: batch-invariant `get_default_config` (fused_moe.py:1310-1316), the tile sm_120's capture pins
BATCH_INVARIANT_CONFIG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1}
#: lane-steps (output words x K/32) per comparison job
JOB_LANE_STEPS = 4_000_000
WGMMA = re.compile(r"wgmma\.mma_async\.sync\.aligned\.m\d+n\d+k\d+\.\w+\.\w+\.\w+")
WGMMA_OK = re.compile(r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k32\.f32\.e4m3\.e4m3")
MMA_SYNC = re.compile(r"(?<![\w.])mma\.sync\.aligned\.m\d+n\d+k\d+[\w.]*")
MMA_SYNC_OK = re.compile(r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.f16\.f16\.f32")
CVT_E4M3 = re.compile(r"cvt\.rn\.f16x2\.e4m3x2")
#: an e4m3 `tl.dot` in a kernel's TTIR: (M, K) x (K, N)
TT_DOT = re.compile(r"tt\.dot [^\n]*: tensor<(\d+)x(\d+)xf8E4M3FN> \* tensor<(\d+)x(\d+)xf8E4M3FN> -> tensor<(\d+)x(\d+)xf32>")
#: the PTX each DOT's instructions are (`ptx`)
FAMILY = {"HopperE4m3QgmmaDot32_v1": "wgmma", "HopperE4m3MmaV2Dot32_v1": "mmav2"}
MMAV2 = "HopperE4m3MmaV2Dot32_v1"
KERNEL_TYPES = {"type": "FusedMoEKernel", "prepare_finalize": "MoEPrepareAndFinalizeNoDPEPModular", "experts": "TritonExperts"}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(triton_cache: Path) -> tuple[dict[str, str], dict[str, str]]:
    """vLLM's defaults (`served_defaults`, before vLLM is imported) and this run's own settings: (the env set, the switches cleared)."""
    cleared = served_defaults()
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(triton_cache)}
    os.environ.update(env)
    return env, cleared


def served() -> dict[str, Any]:
    """The switches as vLLM read them: DeepGEMM on with its ue8m0 scales (the MoE quants' `use_ue8m0`), not batch invariant, vLLM's
    own tuned configs."""
    import vllm.envs as envs
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    rec = {"VLLM_USE_DEEP_GEMM": bool(envs.VLLM_USE_DEEP_GEMM), "deep_gemm_e8m0": bool(is_deep_gemm_e8m0_used()),
           "VLLM_BATCH_INVARIANT": bool(envs.VLLM_BATCH_INVARIANT), "VLLM_TUNED_CONFIG_FOLDER": envs.VLLM_TUNED_CONFIG_FOLDER}
    rec["ok"] = (rec["VLLM_USE_DEEP_GEMM"] and rec["deep_gemm_e8m0"] and not rec["VLLM_BATCH_INVARIANT"]
                 and rec["VLLM_TUNED_CONFIG_FOLDER"] is None)
    return rec


def launch_dot(config: dict[str, Any], swap: bool):
    """The e4m3 k32 step a `fused_moe_kernel` launch's tile compiles to, the DOT its Programs bind (`fp8_moe.sm90_dot`; Triton's
    default 4 warps when the config names none)."""
    from verity.ml.definitions.registry.fp8_moe import sm90_dot

    return sm90_dot(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config.get("num_warps", 4), swap)


def dot_definition(dot_id: str):
    """The DOT a launch record names, by id."""
    from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32

    return {d.id: d for d in (HopperE4m3QgmmaDot32, HopperE4m3MmaV2Dot32)}[dot_id]


def launch_tile(x: dict[str, Any]) -> tuple[int, int, int, int]:
    """A launch record's `tl.dot` as its kernel's TTIR shows it, (M, N, K, warps): M x N = BLOCK_SIZE_N x BLOCK_SIZE_M under SWAP_AB."""
    c = x["config"]
    m, n = (c["BLOCK_SIZE_N"], c["BLOCK_SIZE_M"]) if x["SWAP"] else (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"])
    return m, n, x["KB"], c.get("num_warps", 4)


# ------------------------------------------------------------------------------------------------------------------ operands
def _e4m3(gen, shape):
    """Random e4m3 bytes, every code but NaN (0x7F / 0xFF): all exponents, subnormals and both zeros."""
    import torch

    b = torch.randint(0, 256, shape, generator=gen, device="cuda", dtype=torch.uint8)
    return torch.where((b & 0x7F) == 0x7F, b ^ 1, b).view(torch.float8_e4m3fn)


def _scales(gen, shape, lo: float, hi: float):
    import torch

    return torch.exp2(torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(lo, hi, generator=gen))


def _activation(gen, M: int, K: int):
    """bf16 rows of randn * 2^U(-3, 3); from M >= 4 on, row 1 is zero, row 2 one outlier over tiny values (most e4m3 words subnormal
    or zero), row 3 bf16 subnormals (a subnormal f32 scale)."""
    import torch

    x = torch.randn(M, K, generator=gen, device="cuda") * _scales(gen, (M, 1), -3, 3)
    if M >= 4:
        x[1] = 0
        x[2] *= 2.0 ** -9
        x[2, 7] = 2.0 ** 12
        x[3] *= 2.0 ** -130
    return x.to(torch.bfloat16).contiguous()


def mmav2_probe(gen, rows: int, cols: int, K: int, device: str = "cuda"):
    """(x (rows, K) bf16, w (cols, K) e4m3): small e4m3 codes (exponent field 0..4, any mantissa and sign) with, in every 32-wide k
    window, x = +448 at k = 0 and 2, w = +448 at k = 0 and -448 at k = 2.  x is the codes times 2^-8, so each 128-group's max is
    448 * 2^-8 and `Fp8GroupQuantUe8m0_v1` gives the scale 2^-8 and the codes back exactly."""
    import torch

    def small(shape):
        sign = torch.randint(0, 2, shape, generator=gen, device=device, dtype=torch.uint8) << 7
        exp = torch.randint(0, 5, shape, generator=gen, device=device, dtype=torch.uint8) << 3
        return sign | exp | torch.randint(0, 8, shape, generator=gen, device=device, dtype=torch.uint8)

    x, w = small((rows, K)), small((cols, K))
    x[:, 0::32] = 0x7E
    x[:, 2::32] = 0x7E
    w[:, 0::32] = 0x7E
    w[:, 2::32] = 0xFE
    xb = (x.view(torch.float8_e4m3fn).to(torch.float32) * 2.0 ** -8).to(torch.bfloat16)
    return xb.contiguous(), w.view(torch.float8_e4m3fn).contiguous()


def _u8(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.uint8).cpu().numpy()


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().to(torch.float32).cpu().numpy().view(np.uint32)


def same_words(a, b) -> bool:
    """Two runs' outputs of one launch hold the same words, compared as bits: a NaN word is unequal to itself as a float, so
    `torch.equal` calls two runs that wrote the same words different wherever one is NaN."""
    import torch

    return bool(a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _save(d: Path, **arrays: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))
        out[k] = str(d / f"{k}.npy")
    return out


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def _chunks(n: int, cost: int):
    step = max(1, JOB_LANE_STEPS // max(1, cost))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


# ------------------------------------------------------------------------------------------------------- competing models
def _halves_step(model, halves) -> Callable:
    """A k32 e4m3 step of `fp8_moe_rows.TWIN_DOTS`' signature: the bytes widened to bf16, then `model`'s k16 step over each index
    set of `halves` in turn, each from the previous one's result."""
    from verity.ml.kernels.tc import group_sum_total_batch
    from verity.ml.definitions.numerics.fp8_moe_rows import E4M3_AS_BF16, E4M3_NAN

    def step(acc, a, b):
        acc = np.asarray(acc, dtype=np.uint32)
        a, b = np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)
        nonfin = ((a & E4M3_NAN) == E4M3_NAN).any(-1) | ((b & E4M3_NAN) == E4M3_NAN).any(-1) | ((acc & 0x7F800000) == 0x7F800000)
        wa, wb = E4M3_AS_BF16[a], E4M3_AS_BF16[b]
        for h in halves:
            acc = group_sum_total_batch(model, acc, wa[:, h], wb[:, h])[0]
        return np.asarray(acc, dtype=np.uint32), nonfin

    return step


def _fma_chain(acc, a, b):
    """Thirty-two IEEE FMAs in k order, each product exact: a dot with no grouped adder."""
    from verity.ml.circuits.silicon.cast import e4m3_to_f32_word
    from verity.ml.definitions.numerics.derived_rows import fma_rn
    from verity.ml.definitions.numerics.fp8_moe_rows import E4M3_NAN

    f32 = np.array([e4m3_to_f32_word(e) for e in range(256)], dtype=np.uint32)
    acc = np.asarray(acc, dtype=np.uint32)
    a, b = np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)
    nonfin = ((a & E4M3_NAN) == E4M3_NAN).any(-1) | ((b & E4M3_NAN) == E4M3_NAN).any(-1) | ((acc & 0x7F800000) == 0x7F800000)
    for k in range(a.shape[-1]):
        acc = fma_rn(f32[a[:, k]], f32[b[:, k]], acc)
    return np.asarray(acc, dtype=np.uint32), nonfin


def mmav2_alternatives() -> dict[str, Callable]:
    """The models of `HopperE4m3MmaV2Dot32_v1` the MMAv2 probe must refute, each a k32 step: the twin is core's
    `HOPPER_BF16_M16N8K16` (one group of 16, adder width 26) over `MMAV2_HALVES` (k mod 4 < 2, then >= 2)."""
    from verity.ml.circuits.silicon.models import AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16
    from verity.ml.definitions.numerics.derived_rows import tc_dot32_e4m3
    from verity.ml.definitions.numerics.fp8_moe_rows import MMAV2_HALVES

    natural = (np.arange(16), np.arange(16, 32))
    alts = {"natural_k": _halves_step(HOPPER_BF16_M16N8K16, natural),
            "halves_swapped": _halves_step(HOPPER_BF16_M16N8K16, MMAV2_HALVES[::-1])}
    alts.update({f"width_{w}": _halves_step(dataclasses.replace(HOPPER_BF16_M16N8K16, width=w), MMAV2_HALVES) for w in (24, 25, 27, 28)})
    alts["ampere_8x2"] = _halves_step(AMPERE_BF16_M16N8K16, MMAV2_HALVES)
    alts["one_k32"] = _halves_step(dataclasses.replace(HOPPER_BF16_M16N8K16, groups=(32,)), (np.arange(32),))
    alts["qgmma_k32"] = tc_dot32_e4m3
    alts["fma_chain"] = _fma_chain
    return alts


#: `moe_sum_probe`'s planted columns, each the TOPK bf16 words of one output word: (name, slot words)
SUM_PLANTS = (("neg_zero", (0x8000,) * 8),
              ("mixed_zero", (0x8000, 0x0000) * 4),
              ("tie_into_inf", (0x7F7F, 0x7B00) + (0,) * 6),
              ("overflow", (0x7F7F, 0x7F7F) + (0,) * 6),
              ("neg_overflow", (0xFF7F, 0xFF7F, 0xFF7F) + (0,) * 5),
              ("nan_quiet", (0x7FC0,) + (0x3F80,) * 7),
              ("nan_signalling", (0x3F80, 0x7F81) + (0x3F80,) * 6),
              ("nan_negative", (0xFFC0,) + (0,) * 7),
              ("inf_minus_inf", (0x7F80, 0xFF80) + (0,) * 6),
              ("inf", (0x3F80, 0x7F80) + (0x3F80,) * 6),
              ("tie_even", (0x3F80, 0x3B80) + (0,) * 6),
              ("tie_odd", (0x3F81, 0x3B80) + (0,) * 6),
              ("bf16_accumulator", (0x3F80, 0x3B80, 0x3B80) + (0,) * 5),
              ("subnormal", (0x0001,) * 8),
              ("subnormals_to_normal", (0x0040,) * 8),
              ("cancel_ascending", (0x3F80, 0x4E80, 0xCE80) + (0,) * 5),
              ("cancel_pairs", (0x4E80, 0x3F80, 0xCE80, 0x3F80) + (0,) * 4))


def moe_sum_probe(seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(SUM_TOKENS, TOPK, HIDDEN) bf16 words: random finite words, with each of `SUM_PLANTS` written into one column of every token,
    its slots rolled by the token (so each slot order occurs); and the planted (token, column)s."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((SUM_TOKENS, TOP_K, HIDDEN)).astype(np.float32) * np.exp2(rng.integers(-40, 40, (SUM_TOKENS, TOP_K, 1)))
    y = (f.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    planted = []
    for i, (_name, words) in enumerate(SUM_PLANTS):
        for t in range(SUM_TOKENS):
            col = 64 * i + 1
            y[t, :, col] = np.roll(np.array(words, dtype=np.uint16), t)
            y[t, :, col + 8] = np.array(words, dtype=np.uint16)
            planted += [(t, col), (t, col + 8)]
    return y, planted


def moe_sum_alternatives() -> dict[str, Callable]:
    """The models of `MoeSum_v1` (`bf16_rn` of the f32 sum from +0 over ascending k) the planted rows must refute, each y (..., TOPK,
    H) bf16 words -> (..., H) bf16 words."""
    from verity.ml.definitions.numerics import derived_rows as D

    def fold(order, start_first=False, step=None, final=None):
        def f(y):
            y = D.bf16_to_f32_words(np.asarray(y, dtype=np.uint16))
            ks = list(order(y.shape[-2]))
            acc = y[..., ks[0], :] if start_first else np.zeros(y.shape[:-2] + y.shape[-1:], dtype=np.uint32)
            for k in ks[1:] if start_first else ks:
                acc = step(acc, y[..., k, :]) if step else D.add_rn(acc, y[..., k, :])
            return (final or D.f2fp_bf16)(acc).astype(np.uint16)
        return f

    def pairwise(y):
        v = [D.bf16_to_f32_words(np.asarray(y, dtype=np.uint16)[..., k, :]) for k in range(np.asarray(y).shape[-2])]
        while len(v) > 1:
            v = [D.add_rn(v[i], v[i + 1]) if i + 1 < len(v) else v[i] for i in range(0, len(v), 2)]
        return D.f2fp_bf16(v[0]).astype(np.uint16)

    def keep_payload(acc):
        acc = np.asarray(acc, dtype=np.uint32)
        nan = ((acc & 0x7F800000) == 0x7F800000) & ((acc & 0x007FFFFF) != 0)
        return np.where(nan, (acc >> 16) | 0x0040, D.f2fp_bf16(acc)).astype(np.uint16)

    asc = range
    return {"descending": fold(lambda n: range(n - 1, -1, -1)),
            "pairwise_tree": pairwise,
            "first_slot_start": fold(asc, start_first=True),
            "bf16_accumulator": fold(asc, step=lambda a, b: D.bf16_to_f32_words(D.f2fp_bf16(D.add_rn(a, b)))),
            "ftz": fold(asc, step=lambda a, b: D.ftz(D.add_rn(D.ftz(a), D.ftz(b)))),
            "nan_payload_kept": fold(asc, final=keep_payload),
            "truncated": fold(asc, final=lambda acc: (np.asarray(acc, dtype=np.uint32) >> 16).astype(np.uint16))}


# ------------------------------------------------------------------------------------------------------------------- selection
def _keys():
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    return (create_fp8_quant_key(static=True, group_shape=GroupShape(GROUP, GROUP)),
            create_fp8_quant_key(static=False, group_shape=GroupShape(1, GROUP)))


def moe_config(inter: int):
    """`FusedMoEConfig` as GLM-5.3's `FusedMoE` layer makes it on one TP rank without EP, its intermediate size the rank's shard."""
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType

    return FusedMoEConfig(num_experts=NUM_EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=inter,
                          num_local_experts=NUM_EXPERTS, num_logical_experts=NUM_EXPERTS, activation=MoEActivation.SILU, device="cuda",
                          routing_method=RoutingMethodType.DeepSeekV3, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                          in_dtype=torch.bfloat16)


def selection() -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import select_fp8_moe_backend

    wkey, akey = _keys()
    backend, cls = select_fp8_moe_backend(config=moe_config(256), weight_key=wkey, activation_key=akey, allow_vllm_cutlass=False)
    rec = {"moe": {"backend": getattr(backend, "name", str(backend)), "experts": getattr(cls, "__name__", None)}}
    rec["moe_ok"] = rec["moe"]["backend"] == "TRITON" and rec["moe"]["experts"] == "TritonExperts"
    return rec


def served_kernel(inter: int, s1, s2):
    """The FusedMoEKernel `Fp8MoEMethod` builds for the layer: the selected backend's quant config over the block scales, then
    `make_fp8_moe_kernel`."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (make_fp8_moe_kernel, make_fp8_moe_quant_config,
                                                                  select_fp8_moe_backend)

    cfg = moe_config(inter)
    wkey, akey = _keys()
    backend, cls = select_fp8_moe_backend(config=cfg, weight_key=wkey, activation_key=akey, allow_vllm_cutlass=False)
    qc = make_fp8_moe_quant_config(backend, w1_scale=s1, w2_scale=s2, a1_scale=None, a2_scale=None, block_shape=[GROUP, GROUP])
    return make_fp8_moe_kernel(qc, cfg, cls, backend)


def kernel_types(kernel) -> dict[str, str]:
    return {"type": type(kernel).__name__, "prepare_finalize": type(kernel.prepare_finalize).__name__,
            "experts": type(kernel.fused_experts).__name__}


# ------------------------------------------------------------------------------------------------------------------------- moe
class Interceptor:
    """Wraps `invoke_fused_moe_triton_kernel` in fused_moe and triton_moe: each launch's operands and statics, its output cloned
    as soon as it is written (w13's output and w2's share a workspace)."""

    def __init__(self):
        self.launches: list[dict[str, Any]] = []
        self.label = ""

    def __enter__(self):
        import importlib

        self.mods = [importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")]
        try:
            self.mods.append(importlib.import_module("vllm.model_executor.layers.fused_moe.experts.triton_moe"))
        except ImportError:
            pass
        self.inner = self.mods[0].invoke_fused_moe_triton_kernel
        for m in self.mods:
            if hasattr(m, "invoke_fused_moe_triton_kernel"):
                m.invoke_fused_moe_triton_kernel = self.tap
        return self

    def __exit__(self, *exc):
        for m in self.mods:
            if hasattr(m, "invoke_fused_moe_triton_kernel"):
                m.invoke_fused_moe_triton_kernel = self.inner

    def tap(self, A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
            config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant, block_shape=None,
            B_bias=None):
        from vllm.model_executor.layers.fused_moe.utils import enable_swap_ab

        swap = bool(enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])) if use_fp8_w8a8 else False
        kb = config["BLOCK_SIZE_K"] if block_shape is None else min(config["BLOCK_SIZE_K"], min(block_shape[0], block_shape[1]))
        self.inner(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
                   config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant, block_shape, B_bias)
        self.launches.append({"label": self.label, "A": A.clone(), "B": B, "C": C.clone(), "A_scale": A_scale.clone(), "B_scale": B_scale,
                              "topk_weights": topk_weights, "sorted_token_ids": None if sorted_token_ids is None else sorted_token_ids.clone(),
                              "expert_ids": expert_ids.clone(), "num_tokens_post_padded": num_tokens_post_padded.clone(),
                              "mul_routed_weight": bool(mul_routed_weight), "top_k": int(top_k), "config": dict(config),
                              "compute_type": compute_type, "flags": [bool(use_fp8_w8a8), bool(use_int8_w8a8), bool(use_int8_w8a16),
                                                                      bool(use_int4_w4a16), bool(per_channel_quant)],
                              "block_shape": None if block_shape is None else list(block_shape), "bias": B_bias is not None,
                              "KB": int(kb), "SWAP": swap})


def _vectorised(inp, out) -> bool:
    """`moe_sum` launches `moe_sum_vec_kernel` (csrc moe_align_sum_kernels.cu `can_vec`): hidden contiguous and a whole number of
    16-byte packs, every row offset and both pointers 16-byte aligned."""
    H = inp.shape[-1]
    return bool(inp.stride(-1) == 1 and H % 8 == 0 and inp.stride(0) % 8 == 0 and inp.stride(1) % 8 == 0
                and inp.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0)


class Taps:
    """The experts' `activation` and `moe_sum` (instance attributes over TritonExperts' methods, removed on exit): each call's input
    and output, cloned around the call."""

    def __init__(self, experts):
        self.experts, self.label = experts, ""
        self.act: list[dict[str, Any]] = []
        self.sum: list[dict[str, Any]] = []

    def __enter__(self):
        act, msum = self.experts.activation, self.experts.moe_sum

        def activation(kind, output, input, **kw):
            gu = input.clone()
            act(kind, output, input, **kw)
            self.act.append({"label": self.label, "kind": getattr(kind, "name", str(kind)), "in": gu, "out": output.clone()})

        def moe_sum(input, output):
            y = input.clone()
            msum(input, output)
            self.sum.append({"label": self.label, "in": y, "out": output.clone(), "vec": _vectorised(input, output)})

        self.experts.activation, self.experts.moe_sum = activation, moe_sum
        return self

    def __exit__(self, *exc):
        del self.experts.activation, self.experts.moe_sum


def _routing(gen, M: int, hot: int):
    import torch

    pool = torch.randperm(NUM_EXPERTS, generator=gen, device="cuda")[:hot] if hot else torch.arange(NUM_EXPERTS, device="cuda")
    ids = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:TOP_K]] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOP_K, generator=gen, device="cuda", dtype=torch.float32) + 0.05
    return ids.contiguous(), (w / w.sum(dim=1, keepdim=True) * 2.5).contiguous()


def _assignment_ok(L: dict[str, Any], ids_flat: np.ndarray) -> bool:
    """The launch's (token, slot) -> expert assignment is the routing's: without sorted ids (the naive assignment, M * top_k * 4 <=
    E) block b is pair b and its expert ids[b]; else every valid slot of every BLOCK_SIZE_M block routes to its block's expert, each
    pair once."""
    ex, bm, n = L["expert_ids"].cpu().numpy(), L["config"]["BLOCK_SIZE_M"], ids_flat.size
    ntpp = int(L["num_tokens_post_padded"].reshape(-1)[0])
    if L["sorted_token_ids"] is None:
        return bool(np.array_equal(ex, ids_flat) and ntpp == n * bm)
    st = L["sorted_token_ids"].cpu().numpy()
    seen = np.zeros(n, dtype=np.int64)
    for b in range(min(ex.size, cdiv(ntpp, bm))):
        p = st[b * bm:(b + 1) * bm]
        p = p[p < n]
        if p.size and (ex[b] < 0 or not np.all(ids_flat[p] == ex[b])):
            return False
        seen[p] += 1
    return bool(np.all(seen == 1))


def moe_cases(quick: bool) -> list[tuple[str, int, int, int]]:
    return [c for c in MOE if not quick or c[0] in QUICK_MOE]


def required_cases(quick: bool) -> list[str]:
    """The cases a GO needs: every MoE case of the run (`--quick` keeps `QUICK_MOE`), both probes and the planted sum."""
    return [f"moe/{c[0]}" for c in moe_cases(quick)] + [f"mmav2/{c[0]}" for c in MMAV2_PROBE] + ["sum/planted"]


def moe_launches(work: Path, phase: str, name: str, inter: int, M: int, hot: int, seed: int) -> tuple[list[dict], list[dict]]:
    """One case on the GPU: the served kernel twice, on random operands (phase "moe") or `mmav2_probe`'s x and w1 (phase "mmav2");
    returns (launch records, comparison jobs)."""
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from verity.ml.definitions.registry.fp8_moe import check_statics

    probe = phase == "mmav2"
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if probe:
        x, w1 = mmav2_probe(gen, M, NUM_EXPERTS * 2 * inter, HIDDEN)
        w1 = w1.view(NUM_EXPERTS, 2 * inter, HIDDEN)
    else:
        w1 = _e4m3(gen, (NUM_EXPERTS, 2 * inter, HIDDEN))
    w2 = _e4m3(gen, (NUM_EXPERTS, HIDDEN, inter))
    s1 = _scales(gen, (NUM_EXPERTS, cdiv(2 * inter, GROUP), HIDDEN // GROUP), -12, -6)
    s2 = _scales(gen, (NUM_EXPERTS, cdiv(HIDDEN, GROUP), inter // GROUP), -12, -6)
    if not probe:
        x = _activation(gen, M, HIDDEN)
    ids, tw = _routing(gen, M, hot)
    kernel = served_kernel(inter, s1, s2)
    outs = []
    with Interceptor() as icp, Taps(kernel.fused_experts) as taps:
        for run in (1, 2):
            icp.label = taps.label = f"run{run}"
            outs.append(kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=tw, topk_ids=ids, activation=MoEActivation.SILU,
                                     global_num_experts=NUM_EXPERTS, expert_map=None, apply_router_weight_on_input=False).clone())
    torch.cuda.synchronize()
    runs = {lab: ([L for L in icp.launches if L["label"] == lab], [a for a in taps.act if a["label"] == lab],
                  [s for s in taps.sum if s["label"] == lab]) for lab in ("run1", "run2")}
    counts = {lab: [len(v) for v in r] for lab, r in runs.items()}
    chain = {"phase": phase, "case": name, "launch": "chain", "I": inter, "M": M, "kernel": kernel_types(kernel), "counts": counts}
    if counts != {"run1": [2, 1, 1], "run2": [2, 1, 1]}:
        chain["ok"] = False
        return [chain], []
    (w13, w2l), (act,), (msum,) = runs["run1"]
    (w13b, w2b), (actb,), (msumb,) = runs["run2"]
    chain.update({"act_kind": act["kind"], "sum_vectorised": msum["vec"],
                  "deterministic": (same_words(outs[0], outs[1]) and same_words(act["out"], actb["out"])
                                    and same_words(msum["out"], msumb["out"])),
                  "act_reads_w13": same_words(act["in"], w13["C"].reshape(M * TOP_K, 2 * inter)),
                  "sum_reads_w2": same_words(msum["in"], w2l["C"].reshape(M, TOP_K, HIDDEN)),
                  "returns_sum": same_words(outs[0], msum["out"]),
                  "shapes_ok": (list(x.shape) == [M, HIDDEN] and list(act["out"].shape) == [M * TOP_K, inter]
                                and list(msum["out"].shape) == [M, HIDDEN] and outs[0].dtype == torch.bfloat16)})
    chain["ok"] = bool(chain["kernel"] == KERNEL_TYPES and chain["act_kind"] == "SILU" and chain["sum_vectorised"]
                       and chain["deterministic"] and chain["act_reads_w13"] and chain["sum_reads_w2"] and chain["returns_sum"]
                       and chain["shapes_ok"])
    ids_flat = ids.cpu().numpy().astype(np.int64).reshape(-1)
    tw_flat = _u32(tw).reshape(-1)
    used = np.unique(ids_flat)
    umap = np.full(NUM_EXPERTS, -1, dtype=np.int64)
    umap[used] = np.arange(used.size)
    launches, jobs = [chain], []
    for kind, L, twin in (("w13", w13, w13b), ("w2", w2l, w2b)):
        K, N = L["A"].shape[1], L["B"].shape[1]
        n_pairs = L["A"].shape[0] * L["top_k"]
        try:
            check_statics(K, GROUP, L["KB"])
            statics_known = True
        except ValueError:
            statics_known = False
        rec = {"phase": phase, "case": name, "launch": kind, "I": inter, "M": M, "hot": hot, "K": int(K), "N": int(N), "pairs": int(n_pairs),
               "config": L["config"], "KB": L["KB"], "SWAP": L["SWAP"], "block_shape": L["block_shape"], "flags": L["flags"], "bias": L["bias"],
               "mul_routed_weight": L["mul_routed_weight"], "top_k": L["top_k"], "experts_used": int(used.size),
               "deterministic": same_words(L["C"], twin["C"]), "assignment_ok": _assignment_ok(L, ids_flat),
               "dot": launch_dot(L["config"], L["SWAP"]).id}
        rec["statics_ok"] = (statics_known and L["block_shape"] == [GROUP, GROUP] and L["flags"] == [True, False, False, False, False]
                             and not L["bias"] and L["mul_routed_weight"] == (kind == "w2") and L["top_k"] == (TOP_K if kind == "w13" else 1)
                             and n_pairs == M * TOP_K)
        launches.append(rec)
        sel = torch.as_tensor(used, device="cuda")
        arrays = {"A": _u8(L["A"]), "As": _u32(L["A_scale"]), "B": _u8(L["B"].view(torch.uint8)[sel]), "Bs": _u32(L["B_scale"][sel]),
                  "ids": ids_flat, "umap": umap, "tw": tw_flat, "C": _u16(L["C"].reshape(n_pairs, N))}
        if kind == "w13":
            arrays["x"] = _u16(x)
        else:
            arrays["gu"], arrays["act"] = _u16(act["in"]), _u16(act["out"])
        job = {"phase": phase, "case": name, "launch": kind, "K": int(K), "N": int(N), "KB": L["KB"], "SWAP": L["SWAP"], "dot": rec["dot"],
               "mul": L["mul_routed_weight"], "top_k": L["top_k"], "pairs": int(n_pairs), "statics_known": statics_known,
               "alternatives": probe and kind == "w13", "paths": _save(work / f"{phase}_{name}_{kind}", **arrays)}
        for p0, p1 in _chunks(n_pairs, N * K // 32 * (1 + 10 * job["alternatives"])):
            jobs.append({**job, "pair_range": [p0, p1], "reference": p0 == 0,
                         "quant_rows": list(range(cdiv(p0, TOP_K), cdiv(p1, TOP_K))) if kind == "w13" else []})
    job = {"phase": phase, "case": name, "launch": "sum", "H": HIDDEN, "alternatives": False, "columns": [],
           "paths": _save(work / f"{phase}_{name}_sum", y=_u16(msum["in"]), out=_u16(msum["out"]))}
    jobs += [{**job, "rows": [r0, r1], "reference": r0 == 0} for r0, r1 in _chunks(M, TOP_K * HIDDEN)]
    del w1, w2
    torch.cuda.empty_cache()
    return launches, jobs


def sum_probe_launch(work: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """`moe_sum` (`vllm._custom_ops.moe_sum`, what TritonExperts.moe_sum calls) on `moe_sum_probe`'s rows, twice."""
    import torch
    from vllm import _custom_ops as ops

    yw, planted = moe_sum_probe(seed)
    y = torch.from_numpy(yw.view(np.int16)).cuda().view(torch.bfloat16).contiguous()
    outs = []
    for _ in range(2):
        out = torch.empty(SUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ops.moe_sum(y, out)
        outs.append(out)
    torch.cuda.synchronize()
    rec = {"phase": "sum", "case": "planted", "launch": "sum", "M": SUM_TOKENS, "planted": len(planted),
           "deterministic": same_words(outs[0], outs[1]), "sum_vectorised": _vectorised(y, outs[0]), "unchanged": bool(np.array_equal(
               _u16(y), yw))}
    rec["ok"] = rec["deterministic"] and rec["sum_vectorised"] and rec["unchanged"]
    job = {"phase": "sum", "case": "planted", "launch": "sum", "H": HIDDEN, "alternatives": True, "columns": planted,
           "rows": [0, SUM_TOKENS], "reference": True, "paths": _save(work / "sum_planted", y=yw, out=_u16(outs[0]))}
    return rec, job


# ----------------------------------------------------------------------------------------------------------------- comparisons
def compare_moe(job: dict[str, Any]) -> dict[str, Any]:
    """Pairs [p0, p1) of one expert launch: the GEMM twin at the launch's statics and DOT against the kernel's words (and every
    competing MMAv2 model on a probe's w13 launch), the input quant (w13) or the activation and its quant (w2) on what the kernel
    read, and the reference on a sample."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics.act_rows import silu_mul_expf_rows
    from verity.ml.definitions.numerics.deepgemm_twins import ue8m0_group_quant_row, ue8m0_group_quant_rows
    from verity.ml.definitions.numerics.fp8_moe_rows import TWIN_DOT_OF, moe_expert_fp8_block_row
    from verity.ml.definitions.registry import act as ACT
    from verity.ml.definitions.registry import deepgemm as DG
    from verity.ml.definitions.registry import fp8_moe as MO

    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None, "refuted": {}}
    out = {"phase": job["phase"], "case": job["case"], "launch": job["launch"], "pair_range": job["pair_range"]}
    a = _load(job["paths"])
    K, N, KB, SWAP, top_k = job["K"], job["N"], job["KB"], job["SWAP"], job["top_k"]
    p0, p1 = job["pair_range"]
    if job["launch"] == "w2":
        gu, act = np.asarray(a["gu"][p0:p1]), np.asarray(a["act"][p0:p1])
        bad = ~(silu_mul_expf_rows(gu)["out"] == act).all(axis=1)
        rec["act_rows"] += p1 - p0
        rec["act_mismatch"] += int(bad.sum())
        q, s = ue8m0_group_quant_rows(act, G=GROUP)
        badq = ~((q == np.asarray(a["A"][p0:p1])).all(axis=1) & (s == np.asarray(a["As"][p0:p1])).all(axis=1))
        rec["act_quant_rows"] += p1 - p0
        rec["act_quant_mismatch"] += int(badq.sum())
        if (bad.any() or badq.any()) and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"act_pair": p0 + int(np.flatnonzero(bad | badq)[0]), "act": bool(bad.any()), "act_quant": bool(badq.any())}
        if job["reference"]:
            y = evaluate(bind(ACT.SiluMul, I=act.shape[1]), gu[0].tolist())
            rec["reference_act_rows"] += 1
            rec["twin_mismatch"] += list(y) != silu_mul_expf_rows(gu[:1])["out"][0].tolist()
            y = evaluate(bind(DG.Fp8GroupQuantUe8m0, K=act.shape[1], G=GROUP), act[0].tolist())
            rec["reference_act_quant_rows"] += 1
            rec["twin_mismatch"] += list(y) != q[0].tolist() + s[0].tolist()
    for i, t in enumerate(job["quant_rows"]):
        q, s = ue8m0_group_quant_row(a["x"][t], G=GROUP)
        rec["quant_rows"] += 1
        badq = not (np.array_equal(q, a["A"][t]) and np.array_equal(s, a["As"][t]))
        rec["quant_mismatch"] += badq
        if badq and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"quant_row": t}
        if job["reference"] and i == 0:
            y = evaluate(bind(DG.Fp8GroupQuantUe8m0, K=K, G=GROUP), a["x"][t].tolist())
            rec["reference_quant_rows"] += 1
            rec["twin_mismatch"] += list(y) != q.tolist() + s.tolist()
    if not job["statics_known"]:
        return {**out, **rec, "words": (p1 - p0) * N, "unresolved": (p1 - p0) * N}
    alts = mmav2_alternatives() if job["alternatives"] else {}
    for p in range(p0, p1):
        t, u = p // top_k, int(a["umap"][int(a["ids"][p])])
        g = int(a["tw"][p]) if job["mul"] else None
        operands = (a["A"][t], a["As"][t], a["B"][u], a["Bs"][u])
        want, nonfin = moe_expert_fp8_block_row(*operands, G=GROUP, KB=KB, swap=SWAP, g=g, dot=TWIN_DOT_OF[job["dot"]])
        got = np.asarray(a["C"][p])
        bad = np.flatnonzero((want != got) & ~nonfin)
        rec["words"] += N
        rec["mismatch"] += int(bad.size)
        rec["unresolved"] += int(nonfin.sum())
        if bad.size and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"pair": p, "token": t, "expert": int(a["ids"][p]), "column": int(bad[0]), "want": int(want[bad[0]]),
                                     "kernel": int(got[bad[0]])}
        for alt, step in alts.items():
            other, _ = moe_expert_fp8_block_row(*operands, G=GROUP, KB=KB, swap=SWAP, g=g, dot=step)
            rec["refuted"][alt] = rec["refuted"].get(alt, 0) + int((other != got).sum())
        if job["reference"] and p in (p0, p1 - 1):
            fam = MO.MoeExpertFp8BlockCoordinateWV3 if job["mul"] else MO.MoeExpertFp8BlockCoordinateV3
            fn = bind(fam, K=K, G=GROUP, KB=KB, SWAP=SWAP, DOT=dot_definition(job["dot"]))
            for n in sorted({0, N - 1, (p * 7919) % N}):
                args = [a["A"][t].tolist(), a["As"][t].tolist()] + ([g] if g is not None else []) + [a["B"][u][n].tolist(),
                                                                                                     a["Bs"][u][n // GROUP].tolist()]
                rec["reference_coords"] += 1
                rec["twin_mismatch"] += evaluate(fn, *args) != (int(want[n]),)
    return {**out, **rec}


def compare_sum(job: dict[str, Any]) -> dict[str, Any]:
    """Tokens [r0, r1) of one `moe_sum`: `MoeSum_v1{TOPK,H}`'s twin against the kernel's words (and every competing sum on the
    planted rows), and `MoeSumCoordinate_v1` on a sample of the columns and every planted one."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics.rows import moe_sum
    from verity.ml.definitions.registry import moe as MOE_R

    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None, "refuted": {}}
    a = _load(job["paths"])
    r0, r1 = job["rows"]
    H = job["H"]
    y, got = np.asarray(a["y"][r0:r1]), np.asarray(a["out"][r0:r1])
    want = np.stack([moe_sum({"TOPK": TOP_K, "H": H}, [y[i]])["sum"] for i in range(r1 - r0)])
    bad = np.argwhere(want != got)
    rec["sum_rows"] += r1 - r0
    rec["sum_mismatch"] += int(bad.shape[0])
    if bad.size:
        i, c = map(int, bad[0])
        rec["first_mismatch"] = {"sum_token": r0 + i, "column": c, "want": int(want[i, c]), "kernel": int(got[i, c]),
                                 "slots": [int(v) for v in y[i, :, c]]}
    if job["alternatives"]:
        for alt, f in moe_sum_alternatives().items():
            rec["refuted"][alt] = int((f(y) != got).sum())
    if job["reference"]:
        fn = bind(MOE_R.MoeSumCoordinate, TOPK=TOP_K)
        cols = {(t, c) for t, c in job["columns"] if r0 <= t < r1} | {(t, n) for t in range(r0, r1) for n in (0, H - 1, (t * 7919) % H)}
        for t, c in sorted(cols):
            rec["reference_sum_coords"] += 1
            rec["twin_mismatch"] += evaluate(fn, [int(v) for v in y[t - r0, :, c]]) != (int(want[t - r0, c]),)
    return {"phase": job["phase"], "case": job["case"], "launch": "sum", "rows": job["rows"], **rec}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return compare_sum(job) if job["launch"] == "sum" else compare_moe(job)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "launch": job.get("launch"), "error": _error(e)}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx_family(text: str) -> str | None:
    """The DOT family a kernel's PTX runs: "wgmma" (every wgmma k32 e4m3, no mma.sync), "mmav2" (every mma.sync the f16 m16n8k16
    with an f32 accumulator, on e4m3 widened by `cvt.rn.f16x2.e4m3x2`, no wgmma), else None (neither, both, or another form: a
    kernel with neither runs its dot on FMAs)."""
    wg, ms = WGMMA.findall(text), MMA_SYNC.findall(text)
    if wg and not ms and all(WGMMA_OK.fullmatch(w) for w in wg):
        return "wgmma"
    if ms and not wg and all(MMA_SYNC_OK.fullmatch(m) for m in ms) and CVT_E4M3.search(text):
        return "mmav2"
    return None


def ptx(triton_cache: Path, tiles: set[tuple[int, int, int, int]] = frozenset()) -> dict[str, Any]:
    """Every `fused_moe_kernel` the run compiled (one Triton cache entry per specialization): its e4m3 `tt.dot` (M, N, K) from the
    TTIR, its warps from the entry's metadata, the DOT family `fp8_moe.sm90_dot` gives that tile, and the family its PTX runs
    (`ptx_family`); ok when every kernel's two agree and every launch's tile (`launch_tile`) is among the kernels."""
    from verity.ml.definitions.registry.fp8_moe import sm90_dot

    kernels: dict[str, Any] = {}
    for f in sorted(glob.glob(str(triton_cache / "**" / "fused_moe_kernel*.ptx"), recursive=True)):
        p = Path(f)
        text = p.read_text(errors="replace")
        ttir, meta = p.with_suffix(".ttir"), p.with_suffix(".json")
        dots = TT_DOT.findall(ttir.read_text(errors="replace")) if ttir.exists() else []
        warps = json.loads(meta.read_text()).get("num_warps") if meta.exists() else None
        k = {"wgmma": dict(Counter(WGMMA.findall(text))), "mma_sync": dict(Counter(MMA_SYNC.findall(text))),
             "cvt_e4m3": len(CVT_E4M3.findall(text)), "tt_dot": sorted({tuple(map(int, d)) for d in dots}), "num_warps": warps}
        shapes = {(int(m), int(n), int(kk)) for m, kk, _, n, _, _ in dots}
        k["tile"] = [*next(iter(shapes)), warps] if len(shapes) == 1 and isinstance(warps, int) else None
        k["expected"] = FAMILY[sm90_dot(k["tile"][0], k["tile"][1], warps, False).id] if k["tile"] else None
        k["observed"] = ptx_family(text)
        k["ok"] = k["expected"] is not None and k["observed"] == k["expected"]
        kernels[str(p.relative_to(triton_cache))] = k
    compiled = {tuple(k["tile"]) for k in kernels.values() if k["tile"]}
    missing = sorted(t for t in tiles if t not in compiled)
    ok = bool(kernels) and all(k["ok"] for k in kernels.values()) and not missing
    return {"files": len(kernels), "kernels": kernels, "families": dict(Counter(k["observed"] for k in kernels.values())),
            "missing_tiles": [list(t) for t in missing], "ok": ok}


def bind_check(statics: set[tuple[int, int, int, bool, str]]) -> dict[str, Any]:
    """The Programs bind at every GLM-5.3 shape: each expert launch the run made, (K, N, KB, SWAP, DOT id), as
    `MoeExpertFp8BlockGemm(W)_v3` (w13 when K is the hidden size, else w2) and as `fp8_moe_experts`' `MoeExpertFp8Gemm(W)_v1{E=256}`
    with the weight-scale rows each reads; each quant, activation and the sum at the run's sizes."""
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import act as ACT
    from verity.ml.definitions.registry import deepgemm as DG
    from verity.ml.definitions.registry import fp8_moe as MO
    from verity.ml.definitions.registry import fp8_moe_experts as MX
    from verity.ml.definitions.registry import moe as MOE_R

    out: dict[str, Any] = {}
    inters = sorted({c[1] for c in MOE + MMAV2_PROBE})
    for K in [HIDDEN] + inters:
        out[f"Fp8GroupQuantUe8m0_v1{{K={K},G={GROUP}}}"] = bool(bind(DG.Fp8GroupQuantUe8m0, K=K, G=GROUP).signature())
    for inter in inters:
        params, ret = bind(ACT.SiluMul, I=inter).signature()
        out[f"SiluMul_v2{{I={inter}}}"] = list(type_shape(dict(params)["gu"])) == [2 * inter] and list(type_shape(ret)) == [inter]
    params, ret = bind(MOE_R.MoeSum, TOPK=TOP_K, H=HIDDEN).signature()
    out[f"MoeSum_v1{{TOPK={TOP_K},H={HIDDEN}}}"] = (list(type_shape(dict(params)["y"])) == [TOP_K, HIDDEN]
                                                   and list(type_shape(ret)) == [HIDDEN])
    for K, N, KB, SWAP, dot in sorted(statics):
        w13 = K == HIDDEN
        fn = bind(MO.MoeExpertFp8BlockGemmV3 if w13 else MO.MoeExpertFp8BlockGemmWV3, K=K, N=N, G=GROUP, KB=KB, SWAP=SWAP,
                  DOT=dot_definition(dot))
        params, _ = fn.signature()
        out[fn.id] = list(type_shape(dict(params)["ws"])) == [cdiv(N, GROUP), K // GROUP]
        fx = bind(MX.MoeExpertFp8Gemm if w13 else MX.MoeExpertFp8GemmW, E=NUM_EXPERTS, K=K, N=N, G=GROUP, KB=KB, SWAP=SWAP,
                  DOT=dot_definition(dot))
        params, _ = fx.signature()
        out[fx.id] = (list(type_shape(dict(params)["w"])) == [NUM_EXPERTS, N, K]
                      and list(type_shape(dict(params)["ws"])) == [NUM_EXPERTS, cdiv(N, GROUP), K // GROUP])
    return {"bindings": out, "ok": bool(out) and all(out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
TOTALS = ("words", "mismatch", "unresolved", "reference_coords", "twin_mismatch",
          "quant_rows", "quant_mismatch", "reference_quant_rows",
          "act_rows", "act_mismatch", "reference_act_rows",
          "act_quant_rows", "act_quant_mismatch", "reference_act_quant_rows",
          "sum_rows", "sum_mismatch", "reference_sum_coords")


def _launch_ok(x: dict[str, Any]) -> bool:
    if x["launch"] in ("chain", "sum"):
        return x.get("ok") is True
    return bool(x["deterministic"] and x["statics_ok"] and x["assignment_ok"] is True)


def _totals_ok(phase: str, t: dict[str, Any]) -> bool:
    sums = t["sum_rows"] > 0 and t["sum_mismatch"] == 0 and t["reference_sum_coords"] > 0 and t["twin_mismatch"] == 0
    if phase == "sum":
        return sums
    return (sums and t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["reference_coords"] > 0
            and t["quant_rows"] > 0 and t["quant_mismatch"] == 0 and t["reference_quant_rows"] > 0
            and t["act_rows"] > 0 and t["act_mismatch"] == 0 and t["reference_act_rows"] > 0
            and t["act_quant_rows"] > 0 and t["act_quant_mismatch"] == 0 and t["reference_act_quant_rows"] > 0)


def _refuted_ok(phase: str, refuted: dict[str, int]) -> bool:
    if phase == "moe":
        return True
    names = mmav2_alternatives() if phase == "mmav2" else moe_sum_alternatives()
    return set(refuted) == set(names) and all(refuted[n] > 0 for n in names)


def _case(x: dict[str, Any]) -> str:
    return f"{x['phase']}/{x['case']}"


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's own verdict (a MoE case or a probe with its chain and two launches, or the planted sum): its launches' checks,
    its words, the competing models it refuted and the expert tiles, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(_case(x), {"launches_ok": True, "totals": {k: 0 for k in TOTALS}, "refuted": {}, "tiles": [], "dots": {}})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
        if "config" in x:
            cfg = x["config"]
            c["tiles"].append(f"{x['launch']}:{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{x['KB']}{'/swap' if x['SWAP'] else ''}"
                              f"/{FAMILY[x['dot']]}")
            c["dots"][x["launch"]] = x["dot"]
    for e in doc["errors"]:
        if e.split("/")[0] in PHASES and e not in out:
            out[e] = {"launches_ok": False, "totals": {k: 0 for k in TOTALS}, "refuted": {}, "tiles": [], "dots": {}}
    for r in doc.get("comparisons", []):
        c = out.get(_case(r))
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
            for k, v in r.get("refuted", {}).items():
                c["refuted"][k] = c["refuted"].get(k, 0) + v
    for key, c in out.items():
        phase = key.split("/")[0]
        errors = [e for e in doc["errors"] if e == key or e.startswith(f"compare/{key}/")]
        c["ok"] = (c["launches_ok"] and _totals_ok(phase, c["totals"]) and _refuted_ok(phase, c["refuted"]) and not errors
                   and (phase != "mmav2" or c["dots"].get("w13") == MMAV2))
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    got = doc.get("cases") or cases(doc)
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["served"].get("ok") is True and doc["selection"].get("moe_ok") is True
                and doc["ptx"].get("ok") is True and doc["bind"].get("ok") is True and doc["required"]
                and all(got.get(k, {}).get("ok") is True for k in doc["required"]))


def _record_launches(doc: dict[str, Any], launches: list[dict[str, Any]]) -> None:
    doc["launches"] += launches
    for x in launches:
        if x["launch"] == "chain":
            print(f"[glm53-block-fp8] {x['phase']} {x['case']:13s} chain ok={x.get('ok')} kernel={x['kernel']} counts={x['counts']} "
                  + " ".join(f"{k}={x.get(k)}" for k in ("act_kind", "sum_vectorised", "deterministic", "act_reads_w13", "sum_reads_w2",
                                                          "returns_sum", "shapes_ok")), flush=True)
        else:
            print(f"[glm53-block-fp8] {x['phase']} {x['case']:13s} {x['launch']:4s} K={x['K']:5d} N={x['N']:5d} pairs={x['pairs']:5d} "
                  f"KB={x['KB']} SWAP={x['SWAP']} config={x['config']} det={x['deterministic']} statics={x['statics_ok']} "
                  f"assignment={x['assignment_ok']} dot={x['dot']}", flush=True)


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool,
        only: set[str]) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env, cleared = environment(work / "triton_cache")
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "quick": quick,
                           "required": required_cases(quick), "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "served": {}, "selection": {}, "launches": []}
    print(f"[glm53-block-fp8] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-BLOCK-FP8 GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs: list[dict[str, Any]] = []
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        try:
            doc["served"] = served()
        except Exception as e:  # noqa: BLE001
            doc["errors"]["served"] = _error(e)
        print(f"[glm53-block-fp8] served {json.dumps(doc['served'])} cleared={sorted(cleared)}", flush=True)
        try:
            from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized

            if not is_workspace_manager_initialized():
                init_workspace_manager(torch.device("cuda"))
            doc["selection"] = selection()
        except Exception as e:  # noqa: BLE001
            doc["errors"]["selection"] = _error(e)
        print(f"[glm53-block-fp8] selection {json.dumps(doc['selection'])}", flush=True)
        todo = ([("moe", c) for c in moe_cases(quick)] if "moe" in only else []) + ([("mmav2", c) for c in MMAV2_PROBE] if "mmav2" in only else [])
        for i, (phase, (name, inter, M, hot)) in enumerate(todo):
            try:
                launches, mjobs = moe_launches(work, phase, name, inter, M, hot, 9000 + i)
                _record_launches(doc, launches)
                jobs += mjobs
            except Exception as e:  # noqa: BLE001
                doc["errors"][f"{phase}/{name}"] = _error(e)
                print(f"[glm53-block-fp8] {phase} {name} ERROR {doc['errors'][f'{phase}/{name}'][:400]}", flush=True)
        if "sum" in only:
            try:
                rec, job = sum_probe_launch(work, 9500)
                doc["launches"].append(rec)
                jobs.append(job)
                print(f"[glm53-block-fp8] sum planted ok={rec['ok']} {json.dumps(rec)}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"]["sum/planted"] = _error(e)
                print(f"[glm53-block-fp8] sum planted ERROR {doc['errors']['sum/planted'][:400]}", flush=True)
    experts = [x for x in doc["launches"] if "config" in x]
    doc["dots"] = dict(Counter(x["dot"] for x in experts))
    doc["ptx"] = ptx(work / "triton_cache", {launch_tile(x) for x in experts}) if experts else {"ok": None}
    print(f"[glm53-block-fp8] ptx {json.dumps(doc['ptx'])}", flush=True)
    try:
        doc["bind"] = bind_check({(x["K"], x["N"], x["KB"], x["SWAP"], x["dot"]) for x in experts})
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-block-fp8] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"][f"compare/{res['phase']}/{res['case']}/{res.get('launch')}"] = res["error"]
                continue
            key = (res["phase"], res["case"], res["launch"])
            agg = results.setdefault(key, {"phase": res["phase"], "case": res["case"], "launch": res["launch"], **{k: 0 for k in TOTALS},
                                           "refuted": {}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            for k, v in res["refuted"].items():
                agg["refuted"][k] = agg["refuted"].get(k, 0) + v
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        print(f"[glm53-block-fp8] words {r['phase']:5s} {r['case']:13s} {r['launch']:4s} words={r['words']} !={r['mismatch']} "
              f"unresolved={r['unresolved']} quant={r['quant_rows']}!={r['quant_mismatch']} act={r['act_rows']}!={r['act_mismatch']} "
              f"act_quant={r['act_quant_rows']}!={r['act_quant_mismatch']} sum={r['sum_rows']}!={r['sum_mismatch']} "
              f"ref={r['reference_coords']}+{r['reference_quant_rows']}+{r['reference_act_rows']}+{r['reference_act_quant_rows']}"
              f"+{r['reference_sum_coords']} twin!={r['twin_mismatch']} refuted={json.dumps(r['refuted'])} "
              f"first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-block-fp8] case {key:20s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} unresolved={t['unresolved']} quant!={t['quant_mismatch']} act={t['act_rows']}!={t['act_mismatch']} "
              f"act_quant!={t['act_quant_mismatch']} sum={t['sum_rows']}!={t['sum_mismatch']} twin!={t['twin_mismatch']} "
              f"refuted={json.dumps(c['refuted'])} {' '.join(c['tiles'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"GLM53-BLOCK-FP8 {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"dots={json.dumps(doc['dots'])} served={doc['served'].get('ok')} no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} "
          f"missing={[k for k in doc['required'] if k not in doc['cases']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's served fused-MoE chain on this Hopper GPU against its Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"MoE cases {QUICK_MOE}, the probes and the planted sum: a first signal in minutes")
    ap.add_argument("--only", default=",".join(PHASES), help=f"comma list of phases: {', '.join(PHASES)} (a GO needs all)")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, set(a.only.split(",")))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
