"""GLM-5.3-FP8's block-scaled FP8 GEMMs on Hopper against their Programs: the H200's first-hour go/no-go for binding
`fp8.ScaledMmFp8Block_v1` (CUTLASS blockwise, `CutlassFp8BlockScaledMMKernel`), `fp8_moe.MoeExpertFp8BlockGemm(W)_v3{...,DOT}` (Triton
`fused_moe_kernel`, DOT the step of the launch's tile, `fp8_moe.sm90_dot`), `fp8_moe.SiluMulFp8BlockQuant_v1`
(`silu_and_mul_per_block_quant` between the expert launches) and `fp8.Fp8GroupQuant_v1` (`_C.per_token_group_fp8_quant`) at GLM-5.3's
shapes.

One GPU process writes `<out>/glm53_block_fp8_capture.json`, the property record `glm53_block_fp8_capture` (`properties.record`), and
prints one line per case and then `GLM53-BLOCK-FP8 GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and
nothing else runs when it fails (exit 3).  vLLM runs as GLM-5.3's engine does on sm_90: `VLLM_USE_DEEP_GEMM=0` (TargetProfile
`fp8_block_gemm="cutlass"`, target_profile.py), custom ops on (the eager engine's `QuantFP8.forward_cuda`), and not batch invariant (the
engine refuses `VLLM_BATCH_INVARIANT` on sm_90: no sparse-MLA backend supports it), so the experts launch with vLLM's tuned H200
configs and each launch's Program is bound at the statics it used; `--batch-invariant` runs the fixed tile of `get_default_config`
instead.  A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `init_fp8_linear_kernel` with Fp8LinearMethod's keys for 128x128 blocks (fp8.py:281-288) selects CutlassFp8BlockScaledMMKernel
             at every dense shape, its QuantFP8 dispatching forward_cuda; `select_fp8_moe_backend` (TP, no EP) selects TRITON /
             TritonExperts (oracle/fp8.py:113-123).
  dense      GLM-5.3's nine FP8 linears as one TP8 rank runs them (`DENSE`) at M in `DENSE_M` (M % 4 != 0 takes CUTLASS's swap-AB
             kernel, scaled_mm_blockwise_sm90_fp8_dispatch.cuh:210): the selected kernel's `apply_weights`, twice, on a layer of random
             e4m3 weights and served-range scales after its `process_weights_after_loading`.  The operands `apply_block_scaled_mm`
             received: the activation's quant against `Fp8GroupQuant_v1{K,128}` (twin `derived_rows.fp8_group_quant_row`) and every
             output word against `ScaledMmFp8Block_v1{K,N,128}` (twin `scaled_mm_fp8_block_row`).
  moe        `fused_experts` (E = 256, top_k 8, hidden 6144), twice, at one TP8 rank's shard (I = 256) and at whole experts (I = 2048),
             with uniform and hot routing: every `fused_moe_kernel` launch is intercepted (its operands, config, and BLOCK_SIZE_K and
             SWAP_AB as invoke_fused_moe_triton_kernel computes them, fused_moe.py:789-846), its e4m3 step DOT = `launch_dot` of its
             tile, every (token, slot) row of the w13 launch against `MoeExpertFp8BlockGemm_v3{K=6144,N=2I,G=128,KB,SWAP,DOT}`, of the
             w2 launch against `MoeExpertFp8BlockGemmW_v3{K=I,N=6144,...}` (twin `fp8_moe_rows.moe_expert_fp8_block_row` with that
             DOT's step), and the w13 activation's quant against `Fp8GroupQuant_v1`.  TritonExperts's
             own w2 input, `silu_and_mul_per_block_quant` of the w13 output (triton_moe.py:471-485), is launched as `served_w2`, and
             its (q, scale) on every pair against `SiluMulFp8BlockQuant_v1{I,128}` of that pair's w13 output row (twin
             `fp8_moe_rows.silu_mul_fp8_block_quant_rows`): the `act` rows.
             Every launch's (token, slot) -> expert assignment is checked against the routing.
             Each case also runs the Build's own MoE input quant, `per_token_group_quant_fp8(x, 128, use_ue8m0=True)` (what
             `moe_kernel_quantize_input` launches when DeepGEMM is on, vLLM's sm_90 default; the flag passed explicitly so this
             run's `VLLM_USE_DEEP_GEMM=0` does not turn it off -- the wrapper dispatches to the same `_C.per_token_group_fp8_quant`
             launch, fp8_utils.py:620-637), twice, and compares every row against `Fp8GroupQuantUe8m0_v1{6144,128}`
             (registry.deepgemm; twin `deepgemm_twins.ue8m0_group_quant_row`): the `ue8m0` rows.
  reference  on a sample of the coordinates and rows of every launch, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of
             `ScaledMmFp8BlockCoordinate_v1`, `MoeExpertFp8BlockCoordinate(W)_v3{...,DOT}`, `Fp8GroupQuant_v1` and
             `SiluMulFp8BlockQuant_v1`, which the twins must equal.
  ptx        every `fused_moe_kernel` compiled in the run's Triton cache, by the e4m3 `tt.dot` of its TTIR and its warps: a wgmma tile's
             PTX has wgmma, every one `.m64nNk32.f32.e4m3.e4m3`, and no `mma.sync`; an MMAv2 tile's has `mma.sync`, every one
             `.m16n8k16.row.col.f32.f16.f16.f32`, the e4m3 widening `cvt.rn.f16x2.e4m3x2`, and no wgmma; and every launch's tile (the
             dot's M, N and K, and its warps) is among the compiled kernels.
  bind       the Programs bind at every shape (`ScaledMmFp8Block_v1{6144,2624,128}` reads cdiv(2624, 128) = 21 weight-scale rows).

vLLM's tuned H200 config for GLM-5.3's TP8 expert shard (fused_moe/configs/`E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,
block_shape=[128,128].json`, looked up at the nearest batch size, fused_moe.py:1442-1447) gives M <= 3 (every decode step of a batch
of one) BLOCK_SIZE_M 8 x BLOCK_SIZE_N 32, which SWAP_AB does not take (BLOCK_SIZE_N < 64), and M in 4..6 BLOCK_SIZE_M 8 x
BLOCK_SIZE_N 64, which SWAP_AB takes with the dot's N = 8: no wgmma tile either way (both took `mma.sync.m16n8k16` on f16 on the
H200, run r20261012-215811-1ef7), so their DOT is `HopperE4m3MmaV2Dot32_v1`, every larger batch's and whole experts'
`HopperE4m3QgmmaDot32_v1`.  `--wgmma-configs` instead writes those configs with every such entry widened to BLOCK_SIZE_M 16 x
BLOCK_SIZE_N 64, the shape the same file gives M = 8 (`widen_for_wgmma`), to `<out>/moe_configs/`, and runs with
`VLLM_TUNED_CONFIG_FOLDER` pointing there; an engine served with the same folder launches the same tiles, all wgmma.

GO needs 0 mismatches and 0 unresolved words (a twin flags a word it cannot claim: a non-finite accumulator; these operands are finite
and none should occur), every MoE case's `act` and `ue8m0` rows compared with 0 mismatches, deterministic reruns, the PTX scan, and with
`--batch-invariant` every expert launch at BLOCK_SIZE_K 32 without SWAP_AB.  The verdict line counts the expert launches by DOT.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_block_fp8_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--wgmma-configs] [--batch-invariant]
"""
from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-block-fp8-capture/v3"
RECORD = "glm53_block_fp8_capture.json"
GROUP = 128
HIDDEN = 6144
#: GLM-5.3's nine FP8 linears as one TP8 rank runs them, (name, K, N) (the capture map's `fp8-block-linear` taps): 64 heads, 8 per rank;
#: q_lora 2048, kv_lora 512, qk nope 192 + rope 64, v 256; the indexer's wq_b 32 x 128, replicated; dense intermediate 12288 and one
#: shared expert of 2048, column- then row-parallel.  kv_b_proj runs on the dense-MHA prefill route (its absorbed decode forms are
#: bf16), and the indexer's wk + weights_proj is bf16.
DENSE = (("fused_qkv_a_proj", 6144, 2048 + 512 + 64), ("q_b_proj", 2048, 8 * (192 + 64)), ("kv_b_proj", 512, 8 * (192 + 256)),
         ("o_proj", 8 * 256, 6144), ("indexer.wq_b", 2048, 32 * 128), ("mlp.gate_up_proj", 6144, 2 * 12288 // 8),
         ("mlp.down_proj", 12288 // 8, 6144), ("shared_experts.gate_up_proj", 6144, 2 * 2048 // 8), ("shared_experts.down_proj", 2048 // 8, 6144))
#: the B1 row's decode (1) and prefill (256) among them
DENSE_M = (1, 3, 4, 16, 128, 255, 256)
NUM_EXPERTS, TOP_K = 256, 8
#: (name, intermediate size I, tokens, hot experts: 0 routes each token to 8 of all 256); the B1 row's decode is tp8_decode, its
#: prefill tp8_prefill
MOE = (("tp8_decode", 256, 1, 0), ("tp8_small", 256, 5, 0), ("tp8_block", 256, 64, 0), ("tp8_prefill", 256, 256, 16),
       ("whole_decode", 2048, 1, 0), ("whole_hot", 2048, 16, 12))
QUICK_DENSE_M = (1, 4)
QUICK_MOE = ("tp8_decode", "tp8_small", "whole_decode")
#: batch-invariant `get_default_config` (fused_moe.py:1310-1316)
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


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(batch_invariant: bool, triton_cache: Path) -> dict[str, str]:
    """The TargetProfile `fp8_block_gemm="cutlass"` engine's env (target_profile.py `env()`), set before vLLM is imported."""
    env = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(triton_cache)}
    env["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"
    os.environ.update(env)
    return env


def launch_dot(config: dict[str, Any], swap: bool):
    """The e4m3 k32 step a `fused_moe_kernel` launch's tile compiles to, the DOT its Programs bind (`fp8_moe.sm90_dot`; Triton's
    default 4 warps when the config names none)."""
    from verity_vllm.program.registry.fp8_moe import sm90_dot

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


def wgmma_tile(config: dict[str, Any], swap: bool) -> bool:
    """Whether Triton lowers a launch's `tl.dot` to wgmma (MMAv3): `launch_dot` is `HopperE4m3QgmmaDot32_v1` (Triton 3.7.1 on the
    H200 lowered the e4m3 dot at N = 8 to `mma.sync.m16n8k16` and at N = 16 to `wgmma.m64n16k32`, run r20261012-215811-1ef7)."""
    return FAMILY[launch_dot(config, swap).id] == "wgmma"


def widen_for_wgmma(configs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list]]:
    """A tuned-config file (batch size -> config) with every entry `wgmma_tile` refuses widened to BLOCK_SIZE_N 64, so SWAP_AB
    (`enable_swap_ab` on cc 9.0: BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64, fused_moe/utils.py:628-633) makes it the dot's M, and to
    BLOCK_SIZE_M 16, the dot's N under SWAP_AB; the changed entries, (before, after)."""
    out, changed = dict(configs), {}
    for m, c in configs.items():
        if not isinstance(c, dict) or wgmma_tile(c, c["BLOCK_SIZE_M"] < 64 and c["BLOCK_SIZE_N"] >= 64):
            continue
        wide = {**c, "BLOCK_SIZE_M": max(16, c["BLOCK_SIZE_M"]), "BLOCK_SIZE_N": max(64, c["BLOCK_SIZE_N"])}
        if wide["BLOCK_SIZE_M"] < 64 and wgmma_tile(wide, True):
            out[m] = wide
            changed[m] = [c, wide]
    return out, changed


def write_wgmma_configs(folder: Path) -> dict[str, Any]:
    """vLLM's tuned fused-MoE configs on this GPU for GLM-5.3's expert shapes, through `widen_for_wgmma`, written to `folder` for
    `VLLM_TUNED_CONFIG_FOLDER` (which `get_moe_configs` reads before its own configs, fused_moe.py:1138-1163); a shape without a
    tuned file takes `get_default_config`, whose block-FP8 tiles are wgmma tiles already."""
    from vllm.model_executor.layers.fused_moe import fused_moe as FM

    folder.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"folder": str(folder), "files": {}}
    for inter in sorted({c[1] for c in MOE}):
        name = FM.get_config_file_name(NUM_EXPERTS, inter, "fp8_w8a8", [GROUP, GROUP])
        src = Path(FM.__file__).parent / "configs" / name
        if not src.exists():
            rec["files"][name] = None
            continue
        wide, changed = widen_for_wgmma(json.loads(src.read_text()))
        (folder / name).write_text(json.dumps(wide, indent=1) + "\n")
        rec["files"][name] = {"changed": changed}
    return rec


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


def _u8(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.uint8).cpu().numpy()


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().to(torch.float32).cpu().numpy().view(np.uint32)


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


# ------------------------------------------------------------------------------------------------------------------- selection
def selection() -> dict[str, Any]:
    import torch
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    act = create_fp8_quant_key(static=False, group_shape=GroupShape(1, GROUP))
    wkey = create_fp8_quant_key(static=True, group_shape=GroupShape(GROUP, GROUP))
    rec: dict[str, Any] = {"dense": {}}
    for name, K, N in DENSE:
        k = init_fp8_linear_kernel(activation_quant_key=act, weight_quant_key=wkey, input_dtype=torch.bfloat16, out_dtype=torch.bfloat16,
                                   weight_shape=(N, K), module_name=name)
        q = getattr(k, "quant_fp8", None)
        rec["dense"][name] = {"kernel": type(k).__name__, "quant_forward": getattr(getattr(q, "_forward_method", None), "__name__", None),
                              "column_major_scales": getattr(q, "column_major_scales", None), "use_ue8m0": getattr(q, "use_ue8m0", None)}
    rec["dense_ok"] = all(v["kernel"] == "CutlassFp8BlockScaledMMKernel" and v["quant_forward"] == "forward_cuda"
                          and v["column_major_scales"] is True and v["use_ue8m0"] is False for v in rec["dense"].values())
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import select_fp8_moe_backend

        cfg = FusedMoEConfig(num_experts=NUM_EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=256,
                             num_local_experts=NUM_EXPERTS, num_logical_experts=NUM_EXPERTS, activation=MoEActivation.SILU, device="cuda",
                             routing_method=RoutingMethodType.DeepSeekV3, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                             in_dtype=torch.bfloat16)
        backend, cls = select_fp8_moe_backend(config=cfg, weight_key=wkey, activation_key=act, allow_vllm_cutlass=False)
        rec["moe"] = {"backend": getattr(backend, "name", str(backend)), "experts": getattr(cls, "__name__", None)}
        rec["moe_ok"] = rec["moe"]["backend"] == "TRITON" and rec["moe"]["experts"] == "TritonExperts"
    except Exception as e:  # noqa: BLE001
        rec["moe"] = {"error": _error(e)}
        rec["moe_ok"] = False
    return rec


# ----------------------------------------------------------------------------------------------------------------------- dense
def dense_cases(quick: bool) -> list[tuple[str, int, int, int]]:
    return [(name, K, N, M) for name, K, N in DENSE for M in (QUICK_DENSE_M if quick else DENSE_M)]


def dense_launch(work: Path, name: str, K: int, N: int, M: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One dense case on the GPU: the served kernel's apply_weights, twice; returns (launch record, comparison job)."""
    import torch
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    kernel = init_fp8_linear_kernel(activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, GROUP)),
                                    weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(GROUP, GROUP)),
                                    input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(_e4m3(gen, (N, K)), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(_scales(gen, (cdiv(N, GROUP), K // GROUP), -12, -6), requires_grad=False)
    layer.input_scale = None
    kernel.process_weights_after_loading(layer)
    x = _activation(gen, M, K)
    seen: list[dict[str, Any]] = []
    inner = kernel.apply_block_scaled_mm

    def tap(A, B, As, Bs):
        seen.append({"A": A.clone(), "B": B, "As": As.contiguous().clone(), "Bs": Bs.contiguous().clone()})
        return inner(A, B, As, Bs)

    kernel.apply_block_scaled_mm = tap
    out1 = kernel.apply_weights(layer, x).clone()
    out2 = kernel.apply_weights(layer, x).clone()
    torch.cuda.synchronize()
    op = seen[0]
    launch = {"phase": "dense", "case": name, "K": K, "N": N, "M": M, "kernel": type(kernel).__name__,
              "swap_ab_expected": M % 4 != 0, "A_shape": list(op["A"].shape), "As_shape": list(op["As"].shape),
              "Bs_shape": list(op["Bs"].shape), "out_dtype": str(out1.dtype), "deterministic": bool(torch.equal(out1, out2)),
              "operands_ok": (list(op["A"].shape) == [M, K] and list(op["As"].shape) == [M, K // GROUP]
                              and list(op["B"].shape) == [N, K] and list(op["Bs"].shape) == [cdiv(N, GROUP), K // GROUP]
                              and op["A"].dtype == torch.float8_e4m3fn and out1.dtype == torch.bfloat16)}
    d = work / f"dense_{name}_M{M}"
    job = {"phase": "dense", "case": name, "K": K, "N": N, "M": M,
           "paths": _save(d, x=_u16(x), q=_u8(op["A"]), s=_u32(op["As"]), w=_u8(op["B"]), ws=_u32(op["Bs"]), out=_u16(out1))}
    return launch, job


def dense_jobs(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Row chunks of one dense launch; the reference runs on rows 0, 2 (the outlier), 3 (the subnormal scale) and M - 1."""
    M = job["M"]
    ref = sorted({r for r in (0, 2, 3, M - 1) if r < M})
    return [{**job, "rows": [r0, r1], "reference_rows": [r for r in ref if r0 <= r < r1]}
            for r0, r1 in _chunks(M, job["N"] * job["K"] // 32)]


def compare_dense(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one dense launch: quant and GEMM twins against the kernel's words, the reference on a sample."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.registry import fp8 as F

    a = _load(job["paths"])
    K, N = job["K"], job["N"]
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    r0, r1 = job["rows"]
    for r in range(r0, r1):
        q, s = DR.fp8_group_quant_row(a["x"][r], G=GROUP)
        rec["quant_rows"] += 1
        bad_q = not (np.array_equal(q, a["q"][r]) and np.array_equal(s, a["s"][r]))
        rec["quant_mismatch"] += bad_q
        want, nonfin = DR.scaled_mm_fp8_block_row(a["q"][r], a["s"][r], a["w"], a["ws"], G=GROUP)
        got = np.asarray(a["out"][r])
        bad = np.flatnonzero((want != got) & ~nonfin)
        rec["words"] += N
        rec["mismatch"] += int(bad.size)
        rec["unresolved"] += int(nonfin.sum())
        if (bad.size or bad_q) and rec["first_mismatch"] is None:
            n = int(bad[0]) if bad.size else -1
            rec["first_mismatch"] = {"row": r, "column": n, "quant": bool(bad_q),
                                     "want": int(want[n]) if bad.size else None, "kernel": int(got[n]) if bad.size else None}
        if r in job["reference_rows"]:
            fn = bind(F.ScaledMmFp8BlockCoordinate, K=K, G=GROUP)
            for n in sorted({0, N - 1, (r * 7919) % N}):
                y = evaluate(fn, a["q"][r].tolist(), a["s"][r].tolist(), a["w"][n].tolist(), a["ws"][n // GROUP].tolist())
                rec["reference_coords"] += 1
                rec["twin_mismatch"] += y != (int(want[n]),)
            y = evaluate(bind(F.Fp8GroupQuant, K=K, G=GROUP), a["x"][r].tolist())
            rec["reference_quant_rows"] += 1
            rec["twin_mismatch"] += list(y) != q.tolist() + s.tolist()
    return {"phase": "dense", "case": job["case"], "M": job["M"], "rows": job["rows"], **rec}


# ------------------------------------------------------------------------------------------------------------------------- moe
class Interceptor:
    """Wraps `invoke_fused_moe_triton_kernel` in fused_moe and triton_moe: each launch's operands and statics, its output cloned
    as soon as it is written (w13's output and w2's share `cache13`)."""

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


def _routing(gen, M: int, hot: int):
    import torch

    pool = torch.randperm(NUM_EXPERTS, generator=gen, device="cuda")[:hot] if hot else torch.arange(NUM_EXPERTS, device="cuda")
    ids = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:TOP_K]] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOP_K, generator=gen, device="cuda", dtype=torch.float32) + 0.05
    return ids.contiguous(), (w / w.sum(dim=1, keepdim=True) * 2.5).contiguous()


def _assignment_ok(L: dict[str, Any], ids_flat: np.ndarray) -> bool | None:
    """Every valid slot of every BLOCK_SIZE_M block of the launch's sorted ids routes to its block's expert."""
    if L["sorted_token_ids"] is None:
        return None
    st, ex = L["sorted_token_ids"].cpu().numpy(), L["expert_ids"].cpu().numpy()
    bm, n = L["config"]["BLOCK_SIZE_M"], ids_flat.size
    seen = np.zeros(n, dtype=np.int64)
    for b in range(min(ex.size, cdiv(int(L["num_tokens_post_padded"].reshape(-1)[0]), bm))):
        p = st[b * bm:(b + 1) * bm]
        p = p[p < n]
        if p.size and (ex[b] < 0 or not np.all(ids_flat[p] == ex[b])):
            return False
        seen[p] += 1
    return bool(np.all(seen == 1))


def moe_cases(quick: bool) -> list[tuple[str, int, int, int]]:
    return [c for c in MOE if not quick or c[0] in QUICK_MOE]


def moe_launches(work: Path, name: str, inter: int, M: int, hot: int, seed: int, batch_invariant: bool) -> tuple[list[dict], list[dict]]:
    """One MoE case on the GPU: `fused_experts` twice and the served w2; returns (launch records, comparison jobs)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from verity_vllm.program.registry.fp8_moe import check_statics

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w1 = _e4m3(gen, (NUM_EXPERTS, 2 * inter, HIDDEN))
    w2 = _e4m3(gen, (NUM_EXPERTS, HIDDEN, inter))
    s1 = _scales(gen, (NUM_EXPERTS, cdiv(2 * inter, GROUP), HIDDEN // GROUP), -12, -6)
    s2 = _scales(gen, (NUM_EXPERTS, cdiv(HIDDEN, GROUP), inter // GROUP), -12, -6)
    x = _activation(gen, M, HIDDEN)
    ids, tw = _routing(gen, M, hot)
    qc = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=[GROUP, GROUP])
    with Interceptor() as icp:
        for run in (1, 2):
            icp.label = f"run{run}"
            y = fused_experts(x, w1, w2, tw, ids, activation=MoEActivation.SILU, global_num_experts=NUM_EXPERTS, quant_config=qc)
        w13 = icp.launches[0]
        q2, a2 = ops.silu_and_mul_per_block_quant(w13["C"].view(-1, 2 * inter), group_size=GROUP, quant_dtype=torch.float8_e4m3fn)
        icp.label = "served_w2"
        c3 = torch.empty(M, TOP_K, HIDDEN, device="cuda", dtype=torch.bfloat16)
        icp.tap(q2, w2, c3, a2, s2, tw, w13["sorted_token_ids"], w13["expert_ids"], w13["num_tokens_post_padded"], True, 1, w13["config"],
                w13["compute_type"], True, False, False, False, False, [GROUP, GROUP], None)
    torch.cuda.synchronize()
    runs = {lab: [L for L in icp.launches if L["label"] == lab] for lab in ("run1", "run2", "served_w2")}
    ids_flat = ids.cpu().numpy().astype(np.int64).reshape(-1)
    tw_flat = _u32(tw).reshape(-1)
    used = np.unique(ids_flat)
    umap = np.full(NUM_EXPERTS, -1, dtype=np.int64)
    umap[used] = np.arange(used.size)
    launches, jobs = [], []
    for kind, L in (("w13", runs["run1"][0]), ("w2", runs["run1"][1]), ("served_w2", runs["served_w2"][0])):
        K, N = L["A"].shape[1], L["B"].shape[1]
        n_pairs = L["A"].shape[0] * L["top_k"]
        try:
            check_statics(K, GROUP, L["KB"])
            statics_known = True
        except ValueError:
            statics_known = False
        twin = runs["run2"][("w13", "w2").index(kind)] if kind != "served_w2" else None
        rec = {"phase": "moe", "case": name, "launch": kind, "I": inter, "M": M, "hot": hot, "K": int(K), "N": int(N), "pairs": int(n_pairs),
               "config": L["config"], "KB": L["KB"], "SWAP": L["SWAP"], "block_shape": L["block_shape"], "flags": L["flags"], "bias": L["bias"],
               "mul_routed_weight": L["mul_routed_weight"], "top_k": L["top_k"], "experts_used": int(used.size),
               "deterministic": True if twin is None else bool(torch.equal(L["C"], twin["C"])),
               "assignment_ok": _assignment_ok(L, ids_flat), "dot": launch_dot(L["config"], L["SWAP"]).id}
        rec["statics_ok"] = (statics_known and L["block_shape"] == [GROUP, GROUP] and L["flags"] == [True, False, False, False, False]
                             and not L["bias"] and L["mul_routed_weight"] == (kind != "w13") and L["top_k"] == (TOP_K if kind == "w13" else 1)
                             and n_pairs == M * TOP_K and (not batch_invariant or (L["KB"] == 32 and not L["SWAP"]
                                                                                     and L["config"] == BATCH_INVARIANT_CONFIG)))
        rec["y_shape"] = list(y.shape)
        launches.append(rec)
        d = work / f"moe_{name}_{kind}"
        sel = torch.as_tensor(used, device="cuda")
        arrays = {"A": _u8(L["A"]), "As": _u32(L["A_scale"]), "B": _u8(L["B"].view(torch.uint8)[sel]), "Bs": _u32(L["B_scale"][sel]),
                  "ids": ids_flat, "umap": umap, "tw": tw_flat,
                  "C": _u16(L["C"].reshape(n_pairs, N))}
        if kind == "w13":
            arrays["x"] = _u16(x)
        if kind == "served_w2":
            arrays["gu"] = _u16(w13["C"].reshape(n_pairs, 2 * inter))
        job = {"phase": "moe", "case": name, "launch": kind, "K": int(K), "N": int(N), "KB": L["KB"], "SWAP": L["SWAP"], "dot": rec["dot"],
               "mul": L["mul_routed_weight"], "top_k": L["top_k"], "pairs": int(n_pairs), "statics_known": statics_known,
               "paths": _save(d, **arrays)}
        for p0, p1 in _chunks(n_pairs, N * K // 32):
            jobs.append({**job, "pair_range": [p0, p1], "reference": p0 == 0,
                         "quant_rows": list(range(cdiv(p0, TOP_K), cdiv(p1, TOP_K))) if kind == "w13" else []})
    launches.append(jobs_append_ue8m0(work, name, x, jobs))
    del w1, w2
    torch.cuda.empty_cache()
    return launches, jobs


def jobs_append_ue8m0(work: Path, name: str, x, jobs: list[dict]) -> dict[str, Any]:
    """The Build's own MoE input quant on this case's x: `per_token_group_quant_fp8(x, 128, use_ue8m0=True)`, the call
    `moe_kernel_quantize_input` makes when DeepGEMM is on (vLLM's sm_90 default; `use_ue8m0 = is_deep_gemm_e8m0_used()`), with the
    flag passed explicitly so this run's `VLLM_USE_DEEP_GEMM=0` does not turn it off -- the wrapper dispatches to the same
    `_C.per_token_group_fp8_quant` launch (fp8_utils.py:620-637).  Twice (determinism on the bits), every row against
    `Fp8GroupQuantUe8m0_v1{6144,128}`."""
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    M = x.shape[0]
    q1, s1 = per_token_group_quant_fp8(x, GROUP, use_ue8m0=True)
    q2, s2 = per_token_group_quant_fp8(x, GROUP, use_ue8m0=True)
    torch.cuda.synchronize()
    rec = {"phase": "moe", "case": name, "launch": "ue8m0_quant", "M": M, "K": HIDDEN, "pairs": M,
           "deterministic": bool(torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
                                 and torch.equal(s1.view(torch.int32), s2.view(torch.int32))),
           "operands_ok": (list(q1.shape) == [M, HIDDEN] and list(s1.shape) == [M, HIDDEN // GROUP]
                           and q1.dtype == torch.float8_e4m3fn and s1.dtype == torch.float32 and s1.is_contiguous())}
    job = {"phase": "moe", "case": name, "launch": "ue8m0_quant", "K": HIDDEN,
           "paths": _save(work / f"moe_{name}_ue8m0", x=_u16(x), q=_u8(q1), s=_u32(s1))}
    for r0, r1 in _chunks(M, HIDDEN):
        jobs.append({**job, "rows": [r0, r1], "reference": r0 == 0})
    return rec


def compare_act(a: dict[str, np.ndarray], p0: int, p1: int, reference: bool, rec: dict[str, Any]) -> None:
    """The served w2 launch's A and A_scale on pairs [p0, p1), `silu_and_mul_per_block_quant`'s (q, scale), against
    `SiluMulFp8BlockQuant_v1{I,128}` of the pairs' w13 output rows `gu` (the twin), and the reference on pair p0."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels.fp8_moe_rows import silu_mul_fp8_block_quant_rows
    from verity_vllm.program.registry import fp8_moe as MO

    q, s = silu_mul_fp8_block_quant_rows(a["gu"][p0:p1], G=GROUP)
    bad = ~((q == np.asarray(a["A"][p0:p1])).all(axis=1) & (s == np.asarray(a["As"][p0:p1])).all(axis=1))
    rec["act_rows"] += p1 - p0
    rec["act_mismatch"] += int(bad.sum())
    if bad.any() and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {"act_pair": p0 + int(np.flatnonzero(bad)[0])}
    if reference:
        y = evaluate(bind(MO.SiluMulFp8BlockQuant, I=q.shape[1], G=GROUP), a["gu"][p0].tolist())
        rec["reference_act_rows"] += 1
        rec["twin_mismatch"] += list(y) != q[0].tolist() + s[0].tolist()


def compare_ue8m0(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one case's ue8m0 input quant: `Fp8GroupQuantUe8m0_v1{K,128}`'s twin (`ue8m0_group_quant_row`) against the
    kernel's (q, s) bits, and the reference on row r0."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels.deepgemm_twins import ue8m0_group_quant_row
    from verity_vllm.program.registry import deepgemm as DG

    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    a = _load(job["paths"])
    r0, r1 = job["rows"]
    for r in range(r0, r1):
        q, s = ue8m0_group_quant_row(a["x"][r], G=GROUP)
        rec["ue8m0_rows"] += 1
        bad = not (np.array_equal(q, a["q"][r]) and np.array_equal(s, a["s"][r]))
        rec["ue8m0_mismatch"] += bad
        if bad and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"ue8m0_row": r}
        if job["reference"] and r == r0:
            y = evaluate(bind(DG.Fp8GroupQuantUe8m0, K=job["K"], G=GROUP), a["x"][r].tolist())
            rec["reference_ue8m0_rows"] += 1
            rec["twin_mismatch"] += list(y) != q.tolist() + s.tolist()
    return {"phase": "moe", "case": job["case"], "launch": "ue8m0_quant", "rows": job["rows"], **rec}


def compare_moe(job: dict[str, Any]) -> dict[str, Any]:
    """Pairs [p0, p1) of one expert launch: the GEMM twin at the launch's statics and DOT against the kernel's words, the w13
    activation's quant twin, the served w2 launch's input against the activation's twin, and the reference on a sample."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.kernels.fp8_moe_rows import TWIN_DOT_OF, moe_expert_fp8_block_row
    from verity_vllm.program.registry import fp8 as F
    from verity_vllm.program.registry import fp8_moe as MO

    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    out = {"phase": "moe", "case": job["case"], "launch": job["launch"], "pair_range": job["pair_range"]}
    a = _load(job["paths"])
    if job["launch"] == "served_w2":
        compare_act(a, *job["pair_range"], job["reference"], rec)
    if not job["statics_known"]:
        p0, p1 = job["pair_range"]
        return {**out, **rec, "words": (p1 - p0) * job["N"], "unresolved": (p1 - p0) * job["N"]}
    K, N, KB, SWAP, top_k = job["K"], job["N"], job["KB"], job["SWAP"], job["top_k"]
    p0, p1 = job["pair_range"]
    for p in range(p0, p1):
        t, u = p // top_k, int(a["umap"][int(a["ids"][p])])
        g = int(a["tw"][p]) if job["mul"] else None
        want, nonfin = moe_expert_fp8_block_row(a["A"][t], a["As"][t], a["B"][u], a["Bs"][u], G=GROUP, KB=KB, swap=SWAP, g=g,
                                                dot=TWIN_DOT_OF[job["dot"]])
        got = np.asarray(a["C"][p])
        bad = np.flatnonzero((want != got) & ~nonfin)
        rec["words"] += N
        rec["mismatch"] += int(bad.size)
        rec["unresolved"] += int(nonfin.sum())
        if bad.size and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"pair": p, "token": t, "expert": int(a["ids"][p]), "column": int(bad[0]), "want": int(want[bad[0]]),
                                     "kernel": int(got[bad[0]])}
        if job["reference"] and p in (p0, p1 - 1):
            fam = MO.MoeExpertFp8BlockCoordinateWV3 if job["mul"] else MO.MoeExpertFp8BlockCoordinateV3
            fn = bind(fam, K=K, G=GROUP, KB=KB, SWAP=SWAP, DOT=dot_definition(job["dot"]))
            for n in sorted({0, N - 1, (p * 7919) % N}):
                args = [a["A"][t].tolist(), a["As"][t].tolist()] + ([g] if g is not None else []) + [a["B"][u][n].tolist(),
                                                                                                     a["Bs"][u][n // GROUP].tolist()]
                y = evaluate(fn, *args)
                rec["reference_coords"] += 1
                rec["twin_mismatch"] += y != (int(want[n]),)
    for i, t in enumerate(job["quant_rows"]):
        q, s = DR.fp8_group_quant_row(a["x"][t], G=GROUP)
        rec["quant_rows"] += 1
        rec["quant_mismatch"] += not (np.array_equal(q, a["A"][t]) and np.array_equal(s, a["As"][t]))
        if job["reference"] and i == 0:
            y = evaluate(bind(F.Fp8GroupQuant, K=K, G=GROUP), a["x"][t].tolist())
            rec["reference_quant_rows"] += 1
            rec["twin_mismatch"] += list(y) != q.tolist() + s.tolist()
    return {**out, **rec}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    try:
        if job["phase"] == "dense":
            return compare_dense(job)
        return compare_ue8m0(job) if job.get("launch") == "ue8m0_quant" else compare_moe(job)
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
    from verity_vllm.program.registry.fp8_moe import sm90_dot

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


def bind_check(statics: set[tuple[int, bool, str]]) -> dict[str, Any]:
    """The Programs bind at every GLM-5.3 shape, the expert GEMMs at every (KB, SWAP, DOT id) the run launched, with the
    weight-scale rows each reads."""
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import deepgemm as DG
    from verity_vllm.program.registry import fp8 as F
    from verity_vllm.program.registry import fp8_moe as MO

    out: dict[str, Any] = {}
    for name, K, N in DENSE:
        params, _ = bind(F.ScaledMmFp8Block, K=K, N=N, G=GROUP).signature()
        out[f"ScaledMmFp8Block_v1{{K={K},N={N},G={GROUP}}}"] = list(type_shape(dict(params)["ws"])) == [cdiv(N, GROUP), K // GROUP]
        out[f"Fp8GroupQuant_v1{{K={K},G={GROUP}}}"] = bool(bind(F.Fp8GroupQuant, K=K, G=GROUP).signature())
    out[f"Fp8GroupQuantUe8m0_v1{{K={HIDDEN},G={GROUP}}}"] = bool(bind(DG.Fp8GroupQuantUe8m0, K=HIDDEN, G=GROUP).signature())
    for inter in sorted({c[1] for c in MOE}):
        params, ret = bind(MO.SiluMulFp8BlockQuant, I=inter, G=GROUP).signature()
        out[f"SiluMulFp8BlockQuant_v1{{I={inter},G={GROUP}}}"] = (list(type_shape(dict(params)["gu"])) == [2 * inter]
                                                                  and [list(type_shape(t)) for t in ret.fields] == [[inter], [inter // GROUP]])
    for KB, SWAP, dot in sorted(statics or {(32, False, "HopperE4m3QgmmaDot32_v1")}):
        for inter in sorted({c[1] for c in MOE}):
            for fam, K, N in ((MO.MoeExpertFp8BlockGemmV3, HIDDEN, 2 * inter), (MO.MoeExpertFp8BlockGemmWV3, inter, HIDDEN)):
                fn = bind(fam, K=K, N=N, G=GROUP, KB=KB, SWAP=SWAP, DOT=dot_definition(dot))
                params, _ = fn.signature()
                out[fn.id] = list(type_shape(dict(params)["ws"])) == [cdiv(N, GROUP), K // GROUP]
    return {"bindings": out, "ok": all(out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
TOTALS = ("words", "mismatch", "unresolved", "quant_rows", "quant_mismatch", "reference_coords", "reference_quant_rows", "twin_mismatch",
          "act_rows", "act_mismatch", "reference_act_rows", "ue8m0_rows", "ue8m0_mismatch", "reference_ue8m0_rows")


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["deterministic"] and x.get("operands_ok", True) and x.get("statics_ok", True) and x.get("assignment_ok") is not False)


def _words_ok(t: dict[str, Any]) -> bool:
    return (t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["quant_rows"] > 0 and t["quant_mismatch"] == 0
            and t["reference_coords"] > 0 and t["reference_quant_rows"] > 0 and t["twin_mismatch"] == 0)


def _act_ok(t: dict[str, Any]) -> bool:
    return t["act_rows"] > 0 and t["act_mismatch"] == 0 and t["reference_act_rows"] > 0


def _ue8m0_ok(t: dict[str, Any]) -> bool:
    return t["ue8m0_rows"] > 0 and t["ue8m0_mismatch"] == 0 and t["reference_ue8m0_rows"] > 0


def _case(x: dict[str, Any]) -> str:
    return f"dense/{x['case']}/M{x['M']}" if x["phase"] == "dense" else f"moe/{x['case']}"


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's own verdict (a dense linear at one M, or one MoE case with its three launches): its launches' checks and its words,
    and the expert tiles, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(_case(x), {"launches_ok": True, "totals": {k: 0 for k in TOTALS}, "tiles": []})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
        if x["phase"] == "moe" and "config" in x:
            cfg = x["config"]
            c["tiles"].append(f"{x['launch']}:{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{x['KB']}{'/swap' if x['SWAP'] else ''}"
                              f"/{FAMILY[x['dot']]}")
    for e in doc["errors"]:
        if e.split("/")[0] in ("dense", "moe") and e not in out:
            out[e] = {"launches_ok": False, "totals": {k: 0 for k in TOTALS}, "tiles": []}
    for r in doc.get("comparisons", []):
        c = out.get(_case(r))
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
    for key, c in out.items():
        phase, name = key.split("/")[:2]
        errors = [e for e in doc["errors"] if e.startswith(key) or e.startswith(f"compare/{phase}/{name}/")]
        c["ok"] = (c["launches_ok"] and _words_ok(c["totals"]) and (phase != "moe" or (_act_ok(c["totals"]) and _ue8m0_ok(c["totals"])))
                   and not errors)
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    launches = doc["launches"]
    return (doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("dense_ok") is True and doc["selection"].get("moe_ok") is True
            and doc["ptx"].get("ok") is True and doc["bind"].get("ok") is True
            and {x["phase"] for x in launches} == {"dense", "moe"} and all(_launch_ok(x) for x in launches) and _words_ok(doc["totals"])
            and _act_ok(doc["totals"]) and _ue8m0_ok(doc["totals"]))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool, batch_invariant: bool,
        only: set[str], wgmma_configs: bool = False) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment(batch_invariant, work / "triton_cache")
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "batch_invariant": batch_invariant,
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}}
    print(f"[glm53-block-fp8] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-BLOCK-FP8 GATE-FAIL", flush=True)
        raise SystemExit(3)
    if wgmma_configs:
        # before the first MoE launch: get_moe_configs is lru_cached
        try:
            doc["moe_configs"] = write_wgmma_configs(outdir / "moe_configs")
            env["VLLM_TUNED_CONFIG_FOLDER"] = os.environ["VLLM_TUNED_CONFIG_FOLDER"] = doc["moe_configs"]["folder"]
        except Exception as e:  # noqa: BLE001
            doc["errors"]["moe_configs"] = _error(e)
        print(f"[glm53-block-fp8] moe configs {json.dumps(doc.get('moe_configs'))[:1500]}", flush=True)
    doc["selection"], doc["launches"], jobs = {}, [], []
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        try:
            doc["selection"] = selection()
        except Exception as e:  # noqa: BLE001
            doc["errors"]["selection"] = _error(e)
        print(f"[glm53-block-fp8] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
        if "dense" in only:
            for i, (name, K, N, M) in enumerate(dense_cases(quick)):
                try:
                    launch, job = dense_launch(work, name, K, N, M, 7000 + i)
                    doc["launches"].append(launch)
                    jobs += dense_jobs(job)
                    print(f"[glm53-block-fp8] dense {name:28s} K={K:5d} N={N:5d} M={M:4d} kernel={launch['kernel']} "
                          f"det={launch['deterministic']} operands={launch['operands_ok']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    doc["errors"][f"dense/{name}/M{M}"] = _error(e)
                    print(f"[glm53-block-fp8] dense {name} M={M} ERROR {doc['errors'][f'dense/{name}/M{M}'][:400]}", flush=True)
        if "moe" in only:
            for i, (name, inter, M, hot) in enumerate(moe_cases(quick)):
                try:
                    launches, mjobs = moe_launches(work, name, inter, M, hot, 9000 + i, batch_invariant)
                    doc["launches"] += launches
                    jobs += mjobs
                    for x in launches:
                        if x["launch"] == "ue8m0_quant":
                            print(f"[glm53-block-fp8] moe {name:13s} {x['launch']:11s} K={x['K']:5d} rows={x['M']:5d} "
                                  f"det={x['deterministic']} operands={x['operands_ok']}", flush=True)
                            continue
                        print(f"[glm53-block-fp8] moe {name:13s} {x['launch']:9s} K={x['K']:5d} N={x['N']:5d} pairs={x['pairs']:5d} "
                              f"KB={x['KB']} SWAP={x['SWAP']} config={x['config']} det={x['deterministic']} statics={x['statics_ok']} "
                              f"assignment={x['assignment_ok']} dot={x['dot']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    doc["errors"][f"moe/{name}"] = _error(e)
                    print(f"[glm53-block-fp8] moe {name} ERROR {doc['errors'][f'moe/{name}'][:400]}", flush=True)
    moe = [x for x in doc["launches"] if x["phase"] == "moe" and "config" in x]       # the expert launches; not the ue8m0 quant
    doc["dots"] = dict(Counter(x["dot"] for x in moe))
    doc["ptx"] = ptx(work / "triton_cache", {launch_tile(x) for x in moe}) if "moe" in only else {"ok": None}
    print(f"[glm53-block-fp8] ptx {json.dumps(doc['ptx'])}", flush=True)
    try:
        doc["bind"] = bind_check({(x["KB"], x["SWAP"], x["dot"]) for x in moe})
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
            key = (res["phase"], res["case"], res.get("launch"), res.get("M"))
            agg = results.setdefault(key, {"phase": res["phase"], "case": res["case"], "launch": res.get("launch"), "M": res.get("M"),
                                           **{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        print(f"[glm53-block-fp8] words {r['phase']:5s} {r['case']:28s} {r['launch'] or ('M=' + str(r['M'])):9s} words={r['words']} "
              f"!={r['mismatch']} unresolved={r['unresolved']} quant={r['quant_rows']}!={r['quant_mismatch']} "
              f"act={r['act_rows']}!={r['act_mismatch']} ue8m0={r['ue8m0_rows']}!={r['ue8m0_mismatch']} "
              f"ref={r['reference_coords']}+{r['reference_quant_rows']}+{r['reference_act_rows']}+{r['reference_ue8m0_rows']} "
              f"twin!={r['twin_mismatch']}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-block-fp8] case {key:40s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} unresolved={t['unresolved']} quant!={t['quant_mismatch']} act={t['act_rows']}!={t['act_mismatch']} "
              f"ue8m0={t['ue8m0_rows']}!={t['ue8m0_mismatch']} twin!={t['twin_mismatch']} "
              f"{' '.join(c['tiles'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"GLM53-BLOCK-FP8 {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"dots={json.dumps(doc['dots'])} no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} "
          f"errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's block-scaled FP8 GEMMs on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"dense M in {QUICK_DENSE_M}, MoE cases {QUICK_MOE}: a first signal in minutes")
    ap.add_argument("--batch-invariant", action="store_true",
                    help="VLLM_BATCH_INVARIANT=1: the experts launch get_default_config's fixed tile (GLM-5.3's engine refuses it on sm_90)")
    ap.add_argument("--wgmma-configs", action="store_true",
                    help="write vLLM's tuned MoE configs with every tile wgmma can't run widened (<out>/moe_configs/) and run with them")
    ap.add_argument("--only", default="dense,moe", help="comma list of phases: dense, moe")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, a.batch_invariant, set(a.only.split(",")),
              a.wgmma_configs)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
