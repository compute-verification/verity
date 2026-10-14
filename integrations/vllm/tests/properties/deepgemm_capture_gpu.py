"""DeepSeek-V4-Pro's and GLM-5.3's block-scaled FP8 linears and DeepSeek-V4's grouped `wo_a` einsum as vLLM d9105ea80 runs them on
sm_90 by default, against `registry.deepgemm`: the H200's first-hour go/no-go for binding `DeepGemmFp8Linear_v1` (M >= 32: the ue8m0
`per_token_group_quant_fp8`, then DeepGEMM `fp8_gemm_nt`), `FlashInferFp8Linear_v1` (M < 32: FlashInfer's `fp8_blockscale_gemm`,
which quantises its input itself) and `ScaledMmFp8BlockGrouped_v1` (`fp8_einsum("bhr,hdr->bhd", recipe=(1, 128, 128))`), and the
load-time weight processing whose words they bind as derived Inputs (`kernels.deepgemm_twins`).

One GPU process writes `<out>/deepgemm_capture.json`, the property record `deepgemm_capture` (`properties.record`), and
prints one line per case and then `DEEPGEMM GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and
nothing else runs when it fails (exit 3).  vLLM runs with DeepGEMM on (`VLLM_USE_DEEP_GEMM=1`, `VLLM_USE_DEEP_GEMM_E8M0=1`,
`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1`: the defaults on Hopper) and not batch invariant, so M < 32 takes FlashInfer's branch;
`--batch-invariant` sends every M to DeepGEMM's (flashinfer.py:309-310).  GLM-5.3's engine pins `VLLM_USE_DEEP_GEMM=0`
(TargetProfile `fp8_block_gemm="cutlass"`, glm53_block_fp8_capture_gpu.py); its shapes run here as a DeepGEMM-served GLM-5.3 would.
A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `init_fp8_linear_kernel` with Fp8LinearMethod's keys for 128x128 blocks (fp8.py:276-287) under a model config (DeepGEMM's
             `can_implement` reads one, deep_gemm.py:69-76) selects FlashInferFp8DeepGEMMDynamicBlockScaledKernel at every shape,
             `is_deep_gemm_e8m0_used()` holds, and `compute_fp8_einsum_recipe()` is ((1, 128, 128), False).
  mufu       `fa2_target_capture_gpu.mufu`: this GPU's `rcp.approx.ftz` table is core's `MufuRcpFtz_v1` table (FlashInfer's quantiser
             is built with -use_fast_math: its scale is MUFU.RCP + FMUL.FTZ).
  linear     each model's FP8 linears as one TP8 rank runs them (`LINEARS`) on one layer of random e4m3 weights and served-format scales
             (DeepSeek-V4: e8m0 bytes; GLM-5.3: f32, with `BAND_SCALES` planted) after the selected kernel's
             `process_weights_after_loading`, at M in `LINEAR_M`: `apply_weights`, twice, with the branch's module globals tapped
             (flashinfer.py:252-299).  The branch taken, its operands, and every output word against `DeepGemmFp8Linear_v1{K,N}`
             (the quant's (q, s) against `Fp8GroupQuantUe8m0_v1`, the GEMM over the kernel's own (q, s); twins
             `deepgemm_twins.ue8m0_group_quant_row`, `derived_rows.scaled_mm_fp8_block_row`) or `FlashInferFp8Linear_v1{K,N}` (twin
             `deepgemm_twins.flashinfer_linear_row`).
  weights    every 128x128 block of every processed weight against the twin of its processing: DeepSeek-V4's bytes unchanged and
             scales `upcast_e8m0`, GLM-5.3's `requant_weight_ue8m0` (a block whose `amax / 448` is in `requant_band` is the GPU's
             division and log2's to decide: counted, not claimed).
  einsum     DeepSeek-V4's `wo_a` at TP8 (H = 2 groups, K = 4096, N = 1024) through the model's own `deep_gemm_fp8_o_proj`
             (o_proj.py:28-73) with `fused_inv_rope_fp8_quant` and `fp8_einsum` tapped, twice, at M in `EINSUM_M`: every output word
             against `ScaledMmFp8BlockGrouped_v1{2,4096,1024,128}` over the einsum's own operands (twin
             `deepgemm_twins.scaled_mm_fp8_block_grouped_row`); its weight through the selected kernel's `is_bmm` processing.
  jit        every kernel DeepGEMM's JIT generated in the run (`DG_JIT_CACHE_DIR`'s `kernel.cu`): each GEMM `sm90_fp8_gemm_1d2d_impl`
             at BLOCK_K 128, `GemmType::Normal` (fp8_gemm_nt) or `GemmType::Batched` (fp8_einsum), bf16 out, the identity epilogue;
             otherwise only `transpose_fp32`, which puts an activation's f32 scales in TMA-aligned MN-major layout.
  reference  on a sample of the rows, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of
             `fp8.ScaledMmFp8BlockCoordinate_v1{K,128}` at three columns, `Fp8GroupQuantUe8m0_v1{K,128}` and
             `FlashInferFp8Quant1x128_v1{K}`, which the twins must equal; on a non-finite column, the coordinate's word must be the GPU's.
  bind       the Programs bind at every shape.

The activation rows (`_activation`): randn * 2^U(-3, 3); from M >= 4, row 1 zero, row 2 one outlier over tiny values, row 3 bf16
subnormals; from M >= 8, row 4 bf16 subnormals with each 128-group's lane-0 elements (x[0], x[1], x[64], x[65]) zero.  FlashInfer's
quantiser compares with `setp.lt.ftz`, so row 3's groups take lane 0's subnormal as amax and scale +inf (every word NaN: the twins
flag it non-finite, and the GPU's words there must be non-finite), and row 4's take amax 0 and scale 1 (every word zero).  DeepGEMM's
eps (1e-10) keeps both finite.

GO needs 0 mismatches and 0 unresolved words (a word the twin flags non-finite whose GPU word is finite), every reference equal, every
resolved weight block equal, deterministic reruns, the expected branch with the expected operands at every M, and the JIT kernels.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/deepgemm_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--batch-invariant] [--only PHASES]
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import multiprocessing
import os
import re
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_block_fp8_capture_gpu as B
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/deepgemm-capture/v2"
RECORD = "deepgemm_capture.json"
GROUP = 128
KERNEL = "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
#: DeepSeek-V4-Pro's dense FP8 linears as one TP8 rank runs them, (name, K, N): hidden 7168, q_lora 1536, one 512-wide KV head, 128
#: heads of 512 (16 per rank), the indexer's 64 heads of 128 (replicated), 16 output groups of o_lora 1024 (2 per rank), and the
#: shared expert's 3072 column- then row-parallel.
DSV4_LINEARS = (("fused_wqa_wkv", 7168, 1536 + 512), ("wq_b", 1536, 16 * 512), ("indexer.wq_b", 1536, 64 * 128), ("wo_b", 2 * 1024, 7168),
                ("shared_experts.gate_up_proj", 7168, 2 * 3072 // 8), ("shared_experts.down_proj", 3072 // 8, 7168))
#: model -> (its linears, its checkpoint's scale format, its `model_type`)
MODELS = {"dsv4": (DSV4_LINEARS, "e8m0", "deepseek_v4"), "glm53": (B.DENSE, "f32", "glm_moe_dsa")}
#: DeepSeek-V4's `wo_a` at TP8: (H local groups, K = 8 heads x 512, N = o_lora_rank), its heads per group and head split
EINSUM = (2, 8 * 512, 1024)
HEADS_PER_GROUP, NOPE, ROPE, MAX_POS = 8, 448, 64, 4096
#: either side of the branch point (M < 32), the decode batch of one and a prefill of 256
LINEAR_M = (1, 8, 16, 31, 32, 64, 256)
EINSUM_M = (1, 31, 64)
QUICK_LINEAR_M, QUICK_EINSUM_M = (1, 32), (1,)
#: DeepSeek-V4's e8m0 scale bytes: 2^-15 .. 2^-3
E8M0_LO, E8M0_HI = 112, 124
#: GLM-5.3 block scales planted on the first blocks (each with a 448 in it, so amax / 448 is about the scale): a power of two, one and
#: three ulps above one, one ulp below one, all four in `requant_band`, and one outside it
BAND_SCALES = (0x3A800000, 0x3B000001, 0x3B000003, 0x3B7FFFFF, 0x3C000010)
#: the weight-block rows per weights comparison job
JOB_BLOCK_ROWS = 8
PHASES = ("mufu", "linear", "einsum")
JIT = re.compile(r"&(\w+)<(.*?)>\);", re.S)
GEMM_IMPL = "sm90_fp8_gemm_1d2d_impl"
#: the one layout kernel sm_90's path may JIT: an f32 1x128 activation scale made MN-major and TMA-aligned (layout.hpp:40-42)
LAYOUT_IMPL = "transpose_fp32"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(batch_invariant: bool, work: Path) -> dict[str, str]:
    """DeepGEMM on with its ue8m0 scales and FlashInfer's M < 32 branch (vllm/envs.py defaults on Hopper, set explicitly), set before
    vLLM is imported; DeepGEMM's JIT cache in the scratch directory (vLLM keeps a set `DG_JIT_CACHE_DIR`, deep_gemm.py:266-270)."""
    env = {"VLLM_USE_DEEP_GEMM": "1", "VLLM_USE_DEEP_GEMM_E8M0": "1", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "1",
           "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_BATCH_INVARIANT": "1" if batch_invariant else "0",
           "TRITON_CACHE_DIR": str(work / "triton_cache"), "DG_JIT_CACHE_DIR": str(work / "deep_gemm")}
    os.environ.update(env)
    return env


@contextlib.contextmanager
def model_config(cfg, model_type: str):
    """``cfg.model_config`` a stub for the duration: DeepGEMM's `can_implement` refuses without one and reads only
    `hf_text_config.model_type` (for its Blackwell exclusions); `dtype` is what Fp8LinearMethod's input dtype would be."""
    import torch

    stub = types.SimpleNamespace(hf_text_config=types.SimpleNamespace(model_type=model_type), dtype=torch.bfloat16)
    old = cfg.model_config
    object.__setattr__(cfg, "model_config", stub)
    try:
        yield
    finally:
        object.__setattr__(cfg, "model_config", old)


# ------------------------------------------------------------------------------------------------------------------ operands
def _activation(gen, M: int, K: int):
    """`glm53_block_fp8_capture_gpu._activation`, and from M >= 8 row 4: row 3's subnormals with each group's lane-0 elements zero."""
    import torch

    x = B._activation(gen, M, K)
    if M >= 8:
        x[4] = x[3]
        x[4].view(-1, GROUP)[:, torch.tensor([0, 1, 64, 65], device=x.device)] = 0
    return x.contiguous()


def plant_band(w, ws) -> list[list[int]]:
    """`BAND_SCALES` on the first weight blocks in row-major order, each block holding a 448 (0x7E) at its first word; the blocks."""
    import torch

    kb = ws.shape[1]
    blocks = [[i // kb, i % kb] for i in range(min(len(BAND_SCALES), ws.numel()))]
    for (bn, bk), word in zip(blocks, BAND_SCALES):
        ws[bn, bk] = torch.tensor([word], dtype=torch.int32).view(torch.float32)[0]
        w.view(torch.uint8)[bn * GROUP, bk * GROUP] = 0x7E
    return blocks


def _weights(gen, fmt: str, N: int, K: int):
    """Random e4m3 weights [N, K] and their block scales [cdiv(N, 128), K/128] in the checkpoint's format."""
    import torch

    w = B._e4m3(gen, (N, K))
    if fmt == "e8m0":
        ws = torch.randint(E8M0_LO, E8M0_HI + 1, (cdiv(N, GROUP), K // GROUP), generator=gen, device="cuda", dtype=torch.uint8)
        return w, ws.view(torch.float8_e8m0fnu), []
    ws = B._scales(gen, (cdiv(N, GROUP), K // GROUP), -12, -6)
    return w, ws, plant_band(w, ws)


def _raw_scales(ws) -> np.ndarray:
    import torch

    return B._u8(ws) if ws.dtype == torch.float8_e8m0fnu else B._u32(ws)


def _kernel(K: int, N: int):
    import torch
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    return init_fp8_linear_kernel(activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, GROUP)),
                                  weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(GROUP, GROUP)),
                                  input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K))


def _layer(w, ws, bmm: int = 0):
    import torch

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(ws, requires_grad=False)
    layer.input_scale = None
    layer.weight_block_size = [GROUP, GROUP]
    if bmm:
        layer.is_bmm, layer.bmm_batch_size = True, bmm
    return layer


# ------------------------------------------------------------------------------------------------------------------- selection
def selection(cfg) -> dict[str, Any]:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    rec: dict[str, Any] = {"linears": {}}
    for model, (shapes, _, model_type) in MODELS.items():
        with model_config(cfg, model_type):
            for name, K, N in shapes + ((("wo_a", EINSUM[1], EINSUM[0] * EINSUM[2]),) if model == "dsv4" else ()):
                k = _kernel(K, N)
                rec["linears"][f"{model}/{name}"] = {"kernel": type(k).__name__,
                                                     "use_deep_gemm_e8m0": getattr(getattr(k, "fallback", None), "use_deep_gemm_e8m0", None)}
    recipe, tma = compute_fp8_einsum_recipe()
    rec["e8m0_used"] = bool(is_deep_gemm_e8m0_used())
    rec["einsum_recipe"], rec["tma_aligned_scales"] = list(recipe), bool(tma)
    rec["ok"] = (rec["e8m0_used"] and rec["einsum_recipe"] == [1, GROUP, GROUP] and not rec["tma_aligned_scales"]
                 and all(v["kernel"] == KERNEL and v["use_deep_gemm_e8m0"] is True for v in rec["linears"].values()))
    return rec


def mufu_rcp_ok(m: dict[str, Any]) -> bool:
    """Every build of the MUFU probe measured core's rcp table (ex2 is FA2's concern, not this capture's)."""
    builds = m.get("builds") or {}
    return bool(builds) and all(b.get("rcp_sha256") == m["library"]["rcp"] and b.get("verify", {}).get("rcp_mismatches") == 0
                                for b in builds.values())


# ------------------------------------------------------------------------------------------------------------------------ taps
class Taps:
    """Wraps module globals of vLLM modules: each call's name, keyword arguments, operands and outputs (cloned when written)."""

    def __init__(self, module: str, names: tuple[str, ...]):
        self.module, self.names, self.calls = module, names, []

    def __enter__(self):
        import importlib

        self.mod = importlib.import_module(self.module)
        self.inner = {n: getattr(self.mod, n) for n in self.names}
        for n in self.names:
            setattr(self.mod, n, self._tap(n))
        return self

    def __exit__(self, *exc):
        for n, f in self.inner.items():
            setattr(self.mod, n, f)

    def _tap(self, name: str):
        inner = self.inner[name]

        def tap(*args, **kw):
            out = inner(*args, **kw)
            self.calls.append({"op": name, "args": args, "kw": kw, "out": _clone(out)})
            return out
        return tap


def _clone(out):
    if isinstance(out, tuple):
        return tuple(_clone(o) for o in out)
    return out.contiguous().clone() if hasattr(out, "clone") else out


LINEAR_TAPS = ("vllm.model_executor.kernels.linear.scaled_mm.flashinfer",
               ("per_token_group_quant_fp8", "fp8_gemm_nt", "flashinfer_fp8_blockscale_gemm"))
EINSUM_TAPS = ("vllm.models.deepseek_v4.nvidia.ops.o_proj", ("fused_inv_rope_fp8_quant", "fp8_einsum"))
BRANCHES = {("per_token_group_quant_fp8", "fp8_gemm_nt"): "deepgemm", ("flashinfer_fp8_blockscale_gemm",): "flashinfer"}


def branch_of(calls: list[dict[str, Any]]) -> str:
    ops = tuple(c["op"] for c in calls)
    return BRANCHES.get(ops, "other:" + ",".join(ops))


def _same(a, b) -> bool:
    return a.data_ptr() == b.data_ptr() and tuple(a.shape) == tuple(b.shape)


def _equal(a, b) -> bool:
    """Bit for bit: `torch.equal` compares values, and FlashInfer's all-NaN rows (row 3) would never equal themselves."""
    import torch

    ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and torch.equal(a.contiguous().view(ints[a.element_size()]), b.contiguous().view(ints[b.element_size()])))


def branch_operands_ok(calls: list[dict[str, Any]], branch: str, x, layer, out) -> bool:
    """The branch's calls read the layer's processed weights and the input, with flashinfer.py:252-299's arguments, and the op's
    output is the served output."""
    import torch

    w, ws = layer.weight, layer.weight_scale_inv
    if branch == "flashinfer":
        (c,) = calls
        kw = c["kw"]
        return (_same(kw["input"], x) and _same(kw["weight"], w) and _same(kw["weight_scale"], ws) and kw["out_dtype"] == torch.bfloat16
                and _equal(c["out"], out))
    if branch == "deepgemm":
        quant, gemm = calls
        q, s = quant["out"]
        (a, sa), (b, sb), d = gemm["args"][:3]
        return (_same(quant["args"][0], x) and quant["kw"] == {"group_size": GROUP, "column_major_scales": True, "use_ue8m0": True}
                and _equal(a, q) and _equal(sa, s) and _same(b, w) and _same(sb, ws)
                and gemm["kw"] == {"is_deep_gemm_e8m0_used": True} and q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
                and list(q.shape) == list(x.shape) and list(s.shape) == [x.shape[0], x.shape[1] // GROUP] and _equal(d, out))
    return False


# ---------------------------------------------------------------------------------------------------------------------- linear
def linear_cases() -> list[tuple[str, str, int, int]]:
    return [(model, name, K, N) for model, (shapes, _, _) in MODELS.items() for name, K, N in shapes]


def linear_launches(work: Path, model: str, name: str, K: int, N: int, Ms: tuple[int, ...], seed: int,
                    batch_invariant: bool) -> tuple[list[dict], list[dict]]:
    """One linear on the GPU: its layer built and processed once, then at every M `apply_weights` twice with the branches tapped;
    returns (launch records, comparison jobs)."""
    import torch

    fmt = MODELS[model][1]
    kernel = _kernel(K, N)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w, ws, band = _weights(gen, fmt, N, K)
    raw = {"raw_w": B._u8(w), "raw_ws": _raw_scales(ws)}
    layer = _layer(w, ws)
    kernel.process_weights_after_loading(layer)
    pw, pws = layer.weight, layer.weight_scale_inv
    d = work / f"linear_{model}_{name}"
    weights = B._save(d, w=B._u8(pw), ws=B._u32(pws), **raw)
    wrec = {"phase": "weights", "model": model, "case": name, "K": K, "N": N, "fmt": fmt, "band_planted": band,
            "w_shape": list(pw.shape), "ws_shape": list(pws.shape), "ws_dtype": str(pws.dtype),
            "operands_ok": (list(pw.shape) == [N, K] and pw.dtype == torch.float8_e4m3fn and list(pws.shape) == [cdiv(N, GROUP), K // GROUP]
                            and pws.dtype == torch.float32), "deterministic": True}
    launches, jobs = [wrec], [{"phase": "weights", "model": model, "case": name, "K": K, "N": N, "fmt": fmt, "paths": weights,
                               "block_rows": [b0, b1]} for b0, b1 in _block_chunks(cdiv(N, GROUP))]
    for M in Ms:
        x = _activation(gen, M, K)
        with Taps(*LINEAR_TAPS) as tp:
            out1 = kernel.apply_weights(layer, x).clone()
            n1 = len(tp.calls)
            out2 = kernel.apply_weights(layer, x).clone()
        torch.cuda.synchronize()
        calls1, calls2 = tp.calls[:n1], tp.calls[n1:]
        branch, expected = branch_of(calls1), "deepgemm" if batch_invariant or M >= 32 else "flashinfer"
        launch = {"phase": "linear", "model": model, "case": name, "K": K, "N": N, "M": M, "kernel": type(kernel).__name__,
                  "branch": branch, "branch_expected": expected, "branch_ok": branch == expected and branch_of(calls2) == branch,
                  "deterministic": _equal(out1, out2), "out_dtype": str(out1.dtype)}
        launch["operands_ok"] = launch["branch_ok"] and branch_operands_ok(calls1, branch, x, layer, out1) and out1.dtype == torch.bfloat16
        launches.append(launch)
        arrays = {"x": B._u16(x), "out": B._u16(out1)}
        if branch == "deepgemm":
            q, s = calls1[0]["out"]
            arrays |= {"q": B._u8(q), "s": B._u32(s)}
        job = {"phase": "linear", "model": model, "case": name, "K": K, "N": N, "M": M, "branch": branch,
               "paths": {**B._save(d / f"M{M}", **arrays), "w": weights["w"], "ws": weights["ws"]}}
        jobs += row_jobs(job, N * K // 32)
    del w, ws, layer
    torch.cuda.empty_cache()
    return launches, jobs


def _block_chunks(nb: int) -> list[tuple[int, int]]:
    return [(b, min(nb, b + JOB_BLOCK_ROWS)) for b in range(0, nb, JOB_BLOCK_ROWS)]


def reference_rows(M: int) -> list[int]:
    return sorted({r for r in (0, 2, 3, 4, M - 1) if r < M})


def row_jobs(job: dict[str, Any], cost: int) -> list[dict[str, Any]]:
    """Row chunks of one launch whose rows cost ``cost`` lane steps each; the reference runs on rows 0, 2, 3, 4 and M - 1."""
    ref = reference_rows(job["M"])
    return [{**job, "rows": [r0, r1], "reference_rows": [r for r in ref if r0 <= r < r1]} for r0, r1 in B._chunks(job["M"], cost)]


def _tally(rec: dict[str, Any], want: np.ndarray, nf: np.ndarray, got: np.ndarray, where: dict[str, Any]) -> None:
    """One row's words: a finite twin word must be the GPU's; a non-finite one (the twin claims no payload) must meet a non-finite
    GPU word, else it is unresolved."""
    got = np.asarray(got, dtype=np.uint16)
    bad = np.flatnonzero((want != got) & ~nf)
    gnf = (got & 0x7F80) == 0x7F80
    rec["words"] += int(want.size)
    rec["mismatch"] += int(bad.size)
    rec["nonfinite_words"] += int((nf & gnf).sum())
    rec["unresolved"] += int((nf & ~gnf).sum())
    if bad.size and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {**where, "column": int(bad[0]), "want": int(want[bad[0]]), "kernel": int(got[bad[0]])}


def _reference_coords(rec: dict[str, Any], K: int, xq, sx, w, ws, want, nf, got, columns) -> None:
    """`ScaledMmFp8BlockCoordinate_v1{K,128}` at ``columns``: the twin's word where it is finite, else the GPU's."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.registry import fp8 as F

    fn = bind(F.ScaledMmFp8BlockCoordinate, K=K, G=GROUP)
    for n in sorted(columns):
        y = evaluate(fn, np.asarray(xq).tolist(), np.asarray(sx).tolist(), np.asarray(w[n]).tolist(), np.asarray(ws[n // GROUP]).tolist())
        rec["reference_coords"] += 1
        if nf[n]:
            rec["nonfinite_reference"] += 1
            rec["nonfinite_mismatch"] += y != (int(got[n]),)
        else:
            rec["twin_mismatch"] += y != (int(want[n]),)


def compare_linear(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one linear launch: DeepGEMM's branch, the quant twin against the kernel's (q, s) and the GEMM twin over them;
    FlashInfer's, the linear's twin; the references on the sampled rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.kernels import deepgemm_twins as T
    from verity_vllm.program.kernels import derived_rows as R
    from verity_vllm.program.registry import deepgemm as DG

    a = B._load(job["paths"])
    K, N, deep = job["K"], job["N"], job["branch"] == "deepgemm"
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    r0, r1 = job["rows"]
    for r in range(r0, r1):
        x = np.asarray(a["x"][r])
        if deep:
            q, s = T.ue8m0_group_quant_row(x)
            bad_q = not (np.array_equal(q, a["q"][r]) and np.array_equal(s, a["s"][r]))
            rec["quant_rows"] += 1
            rec["quant_mismatch"] += bad_q
            if bad_q and rec["first_mismatch"] is None:
                rec["first_mismatch"] = {"row": r, "quant": True}
            xq, sx = np.asarray(a["q"][r]), np.asarray(a["s"][r])
        else:
            q, s = xq, sx = T.flashinfer_quant_1x128_row(x)
        want, nf = R.scaled_mm_fp8_block_row(xq, sx, a["w"], a["ws"], G=GROUP)
        got = np.asarray(a["out"][r])
        _tally(rec, want, nf, got, {"row": r})
        if r in job["reference_rows"]:
            _reference_coords(rec, K, xq, sx, a["w"], a["ws"], want, nf, got, {0, N - 1, (r * 7919) % N})
            fn = bind(DG.Fp8GroupQuantUe8m0, K=K, G=GROUP) if deep else bind(DG.FlashInferFp8Quant1x128, K=K)
            rec["reference_quant_rows"] += 1
            rec["twin_mismatch"] += list(evaluate(fn, x.tolist())) != q.tolist() + s.tolist()
    return {"phase": "linear", "model": job["model"], "case": job["case"], "M": job["M"], "rows": job["rows"], **rec}


def compare_weights(job: dict[str, Any]) -> dict[str, Any]:
    """Weight-block rows [b0, b1) of one processed weight against the twin of its processing, block by block."""
    from verity_vllm.program.kernels import deepgemm_twins as T

    a = B._load(job["paths"])
    N, b0, b1 = job["N"], *job["block_rows"]
    r0, r1 = b0 * GROUP, min(N, b1 * GROUP)
    raw_w = np.asarray(a["raw_w"][r0:r1])
    if job["fmt"] == "e8m0":
        want_w, want_ws = raw_w, T.upcast_e8m0(a["raw_ws"][b0:b1])
        unres = np.zeros(want_ws.shape, dtype=bool)
    else:
        want_w, want_ws, unres = T.requant_weight_ue8m0(raw_w, np.asarray(a["raw_ws"][b0:b1]))
    nb, kb = want_ws.shape
    diff = np.zeros((nb * GROUP, kb * GROUP), dtype=bool)
    diff[:r1 - r0] = np.asarray(a["w"][r0:r1]) != want_w
    bad = diff.reshape(nb, GROUP, kb, GROUP).any(axis=(1, 3)) | (np.asarray(a["ws"][b0:b1]) != want_ws)
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    rec["weight_blocks"] = int((~unres).sum())
    rec["weight_mismatch"] = int((bad & ~unres).sum())
    rec["band_blocks"] = int(unres.sum())
    rec["band_agree"] = int((~bad & unres).sum())
    hit = np.argwhere(bad & ~unres)
    if hit.size:
        rec["first_mismatch"] = {"block": [b0 + int(hit[0][0]), int(hit[0][1])]}
    return {"phase": "weights", "model": job["model"], "case": job["case"], "block_rows": job["block_rows"], **rec}


# ---------------------------------------------------------------------------------------------------------------------- einsum
def einsum_launches(work: Path, Ms: tuple[int, ...], seed: int) -> tuple[list[dict], list[dict]]:
    """DeepSeek-V4's `wo_a` on the GPU: its weight [H*N, K] with e8m0 scales through the selected kernel's `is_bmm` processing, then
    at every M `deep_gemm_fp8_o_proj` (wo_b the identity), twice, with its quant and einsum tapped."""
    import torch
    from vllm.models.deepseek_v4.nvidia.ops import o_proj as OP

    H, K, N = EINSUM
    kernel = _kernel(K, H * N)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w, ws, _ = _weights(gen, "e8m0", H * N, K)
    raw = {"raw_w": B._u8(w), "raw_ws": _raw_scales(ws)}
    layer = _layer(w, ws, bmm=H)
    kernel.process_weights_after_loading(layer)
    pw, pws = layer.weight, layer.weight_scale_inv
    d, nb = work / "einsum_wo_a", cdiv(N, GROUP)
    weights = B._save(d, w=B._u8(pw).reshape(H * N, K), ws=B._u32(pws).reshape(H * nb, K // GROUP), **raw)
    wrec = {"phase": "weights", "model": "dsv4", "case": "wo_a", "K": K, "N": H * N, "fmt": "e8m0", "band_planted": [],
            "w_shape": list(pw.shape), "ws_shape": list(pws.shape), "ws_dtype": str(pws.dtype), "deterministic": True,
            "operands_ok": (list(pw.shape) == [H, N, K] and pw.dtype == torch.float8_e4m3fn and list(pws.shape) == [H, nb, K // GROUP]
                            and pws.dtype == torch.float32)}
    launches = [wrec]
    jobs = [{"phase": "weights", "model": "dsv4", "case": "wo_a", "K": K, "N": H * N, "fmt": "e8m0", "paths": weights,
             "block_rows": [b0, b1]} for b0, b1 in _block_chunks(H * nb)]
    recipe, tma = OP.compute_fp8_einsum_recipe()
    angles = torch.rand(MAX_POS, ROPE // 2, generator=gen, device="cuda", dtype=torch.float32) * 6.2831855
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).contiguous()
    for M in Ms:
        o = (torch.randn(M, H * HEADS_PER_GROUP, NOPE + ROPE, generator=gen, device="cuda")
             * B._scales(gen, (M, 1, 1), -3, 3)).to(torch.bfloat16).contiguous()
        pos = torch.randint(0, MAX_POS, (M,), generator=gen, device="cuda", dtype=torch.int64)
        with Taps(*EINSUM_TAPS) as tp:
            z1 = OP.deep_gemm_fp8_o_proj(o, pos, cos_sin, layer, torch.nn.Identity(), n_groups=H, heads_per_group=HEADS_PER_GROUP,
                                         nope_dim=NOPE, rope_dim=ROPE, o_lora_rank=N, einsum_recipe=recipe, tma_aligned_scales=tma).clone()
            n1 = len(tp.calls)
            z2 = OP.deep_gemm_fp8_o_proj(o, pos, cos_sin, layer, torch.nn.Identity(), n_groups=H, heads_per_group=HEADS_PER_GROUP,
                                         nope_dim=NOPE, rope_dim=ROPE, o_lora_rank=N, einsum_recipe=recipe, tma_aligned_scales=tma).clone()
        torch.cuda.synchronize()
        quant, ein = tp.calls[:n1] if n1 == 2 else (None, None)
        launch = {"phase": "einsum", "model": "dsv4", "case": "wo_a", "H": H, "K": K, "N": N, "M": M, "recipe": list(recipe),
                  "calls": [c["op"] for c in tp.calls], "deterministic": _equal(z1, z2)}
        ok = quant is not None and quant["op"] == "fused_inv_rope_fp8_quant" and ein["op"] == "fp8_einsum"
        if ok:
            xq, sx = quant["out"]
            (a, sa), (b, sb), zz = ein["args"][1], ein["args"][2], ein["args"][3]
            ok = (ein["args"][0] == "bhr,hdr->bhd" and tuple(ein["kw"].get("recipe", ())) == (1, GROUP, GROUP) and _equal(a, xq)
                  and _equal(sa, sx) and _same(b, pw) and _same(sb, pws) and list(xq.shape) == [M, H, K]
                  and xq.dtype == torch.float8_e4m3fn and list(sx.shape) == [M, H, K // GROUP] and sx.dtype == torch.float32
                  and _equal(zz.reshape(M, H * N), z1) and z1.dtype == torch.bfloat16 and n1 * 2 == len(tp.calls))
        launch["operands_ok"] = bool(ok)
        launches.append(launch)
        if ok:
            job = {"phase": "einsum", "model": "dsv4", "case": "wo_a", "H": H, "K": K, "N": N, "M": M,
                   "paths": {**B._save(d / f"M{M}", xq=B._u8(xq), sx=B._u32(sx), z=B._u16(z1).reshape(M, H, N)),
                             "w": weights["w"], "ws": weights["ws"]}}
            jobs += row_jobs(job, H * N * K // 32)
    del w, ws, layer
    torch.cuda.empty_cache()
    return launches, jobs


def compare_einsum(job: dict[str, Any]) -> dict[str, Any]:
    """Tokens [r0, r1) of one einsum launch: every group's words against the grouped twin over the einsum's own operands, the
    reference on the sampled tokens."""
    from verity_vllm.program.kernels import deepgemm_twins as T

    a = B._load(job["paths"])
    H, K, N = job["H"], job["K"], job["N"]
    w = np.asarray(a["w"]).reshape(H, N, K)
    ws = np.asarray(a["ws"]).reshape(H, cdiv(N, GROUP), K // GROUP)
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    r0, r1 = job["rows"]
    for t in range(r0, r1):
        want, nf = T.scaled_mm_fp8_block_grouped_row(a["xq"][t], a["sx"][t], w, ws, G=GROUP)
        for h in range(H):
            got = np.asarray(a["z"][t, h])
            _tally(rec, want[h], nf[h], got, {"row": t, "group": h})
            if t in job["reference_rows"]:
                _reference_coords(rec, K, a["xq"][t, h], a["sx"][t, h], w[h], ws[h], want[h], nf[h], got, {0, N - 1, (t * 7919 + h) % N})
    return {"phase": "einsum", "model": job["model"], "case": job["case"], "M": job["M"], "rows": job["rows"], **rec}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"linear": compare_linear, "weights": compare_weights, "einsum": compare_einsum}[job["phase"]](job)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "model": job["model"], "case": job["case"], "M": job.get("M"), "error": B._error(e)}


# ------------------------------------------------------------------------------------------------------------------- jit and bind
def jit(cache: Path) -> dict[str, Any]:
    """Every kernel DeepGEMM's JIT generated under ``cache`` (one directory per kernel, its `kernel.cu`), as its template and its
    arguments (sm90_fp8_gemm_1d2d.hpp:34-68): each GEMM `sm90_fp8_gemm_1d2d_impl` at BLOCK_K 128, Normal or Batched, bf16 out and
    the identity epilogue; besides them only `transpose_fp32` (smxx_layout.hpp:25-37)."""
    kernels = {}
    for f in sorted(glob.glob(str(cache / "**" / "kernel.cu"), recursive=True)):
        m = JIT.search(Path(f).read_text(errors="replace"))
        name, args = (m.group(1), [s.strip() for s in m.group(2).split(",")]) if m else (None, [])
        k = {"impl": name, "args": args}
        k["ok"] = name == LAYOUT_IMPL or (name == GEMM_IMPL and len(args) == 20 and args[7] == str(GROUP)
                                          and args[17] in ("GemmType::Normal", "GemmType::Batched")
                                          and args[18] == "cutlass::bfloat16_t" and args[19] == "epilogue::transform::EpilogueIdentity")
        kernels[str(Path(f).relative_to(cache))] = k
    gemms = [k for k in kernels.values() if k["impl"] == GEMM_IMPL]
    return {"files": len(kernels), "gemms": len(gemms), "layouts": sum(k["impl"] == LAYOUT_IMPL for k in kernels.values()),
            "kernels": kernels, "gemm_types": sorted({k["args"][17] for k in gemms if k["ok"]}),
            "ok": bool(gemms) and all(k["ok"] for k in kernels.values())}


def bind_check() -> dict[str, Any]:
    """The Programs bind at every shape, with the weight-scale rows each reads."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity_vllm.program.registry import deepgemm as DG

    out: dict[str, Any] = {}
    for shapes, _, _ in MODELS.values():
        for _, K, N in shapes:
            for fam in (DG.DeepGemmFp8Linear, DG.FlashInferFp8Linear):
                params, _ = bind(fam, K=K, N=N).signature()
                out[f"{fam.name}_v1{{K={K},N={N}}}"] = list(type_shape(dict(params)["ws"])) == [cdiv(N, GROUP), K // GROUP]
            for key, fn in ((f"Fp8GroupQuantUe8m0_v1{{K={K},G={GROUP}}}", bind(DG.Fp8GroupQuantUe8m0, K=K, G=GROUP)),
                            (f"FlashInferFp8Quant1x128_v1{{K={K}}}", bind(DG.FlashInferFp8Quant1x128, K=K))):
                _, ret = fn.signature()
                out[key] = [list(type_shape(t)) for t in ret.fields] == [[K], [K // GROUP]]
    H, K, N = EINSUM
    params, ret = bind(DG.ScaledMmFp8BlockGrouped, H=H, K=K, N=N, G=GROUP).signature()
    p = dict(params)
    out[f"ScaledMmFp8BlockGrouped_v1{{H={H},K={K},N={N},G={GROUP}}}"] = (
        list(type_shape(p["xq"])) == [H, K] and list(type_shape(p["sx"])) == [H, K // GROUP] and list(type_shape(p["w"])) == [H, N, K]
        and list(type_shape(p["ws"])) == [H, N // GROUP, K // GROUP] and list(type_shape(ret)) == [H, N])
    return {"bindings": out, "ok": all(out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
TOTALS = ("words", "mismatch", "unresolved", "nonfinite_words", "quant_rows", "quant_mismatch", "reference_coords",
          "reference_quant_rows", "twin_mismatch", "nonfinite_reference", "nonfinite_mismatch", "weight_blocks", "weight_mismatch",
          "band_blocks", "band_agree")


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["deterministic"] and x.get("operands_ok") is True and x.get("branch_ok", True))


def _words_ok(t: dict[str, Any]) -> bool:
    return (t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["quant_mismatch"] == 0 and t["reference_coords"] > 0
            and t["twin_mismatch"] == 0 and t["nonfinite_mismatch"] == 0)


def _weights_ok(t: dict[str, Any]) -> bool:
    return t["weight_blocks"] > 0 and t["weight_mismatch"] == 0


def _case(x: dict[str, Any]) -> str:
    return f"{x['phase']}/{x['model']}/{x['case']}" + ("" if x["phase"] == "weights" else f"/M{x['M']}")


def _case_ok(key: str, c: dict[str, Any], errors: list[str]) -> bool:
    t, phase = c["totals"], key.split("/")[0]
    if phase == "weights":
        good = _weights_ok(t)
    else:
        good = _words_ok(t) and (phase == "einsum" or t["reference_quant_rows"] > 0) and (c["branch"] != "deepgemm" or t["quant_rows"] > 0)
    return c["launches_ok"] and good and not errors


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's own verdict (a linear or the einsum at one M, or one weight's processing): its launch's checks and its words,
    so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(_case(x), {"launches_ok": True, "branch": x.get("branch"), "totals": {k: 0 for k in TOTALS}})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
    for e in doc["errors"]:
        if e.split("/")[0] in ("linear", "einsum", "weights") and e not in out:
            out[e] = {"launches_ok": False, "branch": None, "totals": {k: 0 for k in TOTALS}}
    for r in doc.get("comparisons", []):
        c = out.get(_case(r))
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
    for key, c in out.items():
        errors = [e for e in doc["errors"] if e in (key, f"compare/{key}")]
        c["ok"] = _case_ok(key, c, errors)
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    launches, t = doc["launches"], doc["totals"]
    return (doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["mufu"].get("rcp_ok") is True
            and doc["jit"].get("ok") is True and doc["jit"].get("gemm_types") == ["GemmType::Batched", "GemmType::Normal"]
            and doc["bind"].get("ok") is True and {x["phase"] for x in launches} == {"linear", "weights", "einsum"}
            and {x["branch"] for x in launches if x["phase"] == "linear"} == ({"deepgemm"} if doc["batch_invariant"] else {"deepgemm", "flashinfer"})
            and all(_launch_ok(x) for x in launches) and _words_ok(t) and _weights_ok(t) and t["quant_rows"] > 0
            and t["reference_quant_rows"] > 0 and bool(doc["cases"]) and all(c["ok"] for c in doc["cases"].values()))


def _print_case(key: str, c: dict[str, Any]) -> None:
    t = c["totals"]
    print(f"[deepgemm] case {key:52s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} branch={c['branch']} "
          f"words={t['words']} !={t['mismatch']} unresolved={t['unresolved']} nonfinite={t['nonfinite_words']} quant={t['quant_rows']}"
          f"!={t['quant_mismatch']} ref={t['reference_coords']}+{t['reference_quant_rows']} twin!={t['twin_mismatch']} "
          f"nf_ref={t['nonfinite_reference']}!={t['nonfinite_mismatch']} blocks={t['weight_blocks']}!={t['weight_mismatch']} "
          f"band={t['band_blocks']}~{t['band_agree']}", flush=True)


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool, batch_invariant: bool,
        only: set[str]) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment(batch_invariant, work)
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "batch_invariant": batch_invariant,
                           "phases": sorted(only), "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms),
                           "errors": {}}
    print(f"[deepgemm] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DEEPGEMM GATE-FAIL", flush=True)
        raise SystemExit(3)
    doc["mufu"] = {"rcp_ok": None}
    if "mufu" in only:
        try:
            doc["mufu"] = C.mufu(work, "sm_" + doc["gate"]["torch_capability"].replace(".", ""))
            doc["mufu"]["rcp_ok"] = mufu_rcp_ok(doc["mufu"])
            for b in doc["mufu"]["builds"].values():
                if b.get("tables_kept"):
                    os.replace(b["tables_kept"], outdir / Path(b["tables_kept"]).name)
        except Exception as e:  # noqa: BLE001
            doc["errors"]["mufu"] = B._error(e)
    print(f"[deepgemm] mufu rcp_ok={doc['mufu'].get('rcp_ok')}", flush=True)
    cfg = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    doc["selection"], doc["launches"], jobs = {}, [], []
    with set_current_vllm_config(cfg):
        try:
            doc["selection"] = selection(cfg)
        except Exception as e:  # noqa: BLE001
            doc["errors"]["selection"] = B._error(e)
        print(f"[deepgemm] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
        if "linear" in only:
            for i, (model, name, K, N) in enumerate(linear_cases()):
                try:
                    with model_config(cfg, MODELS[model][2]):
                        launches, ljobs = linear_launches(work, model, name, K, N, QUICK_LINEAR_M if quick else LINEAR_M, 7100 + i,
                                                          batch_invariant)
                    doc["launches"] += launches
                    jobs += ljobs
                    for x in launches[1:]:
                        print(f"[deepgemm] linear {model}/{name:28s} K={K:5d} N={N:5d} M={x['M']:4d} branch={x['branch']} "
                              f"det={x['deterministic']} operands={x['operands_ok']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    doc["errors"][f"linear/{model}/{name}"] = B._error(e)
                    print(f"[deepgemm] linear {model}/{name} ERROR {doc['errors'][f'linear/{model}/{name}'][:400]}", flush=True)
        if "einsum" in only:
            try:
                with model_config(cfg, MODELS["dsv4"][2]):
                    launches, ejobs = einsum_launches(work, QUICK_EINSUM_M if quick else EINSUM_M, 9100)
                doc["launches"] += launches
                jobs += ejobs
                for x in launches[1:]:
                    print(f"[deepgemm] einsum wo_a M={x['M']:4d} calls={x['calls']} det={x['deterministic']} "
                          f"operands={x['operands_ok']}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"]["einsum/dsv4/wo_a"] = B._error(e)
                print(f"[deepgemm] einsum ERROR {doc['errors']['einsum/dsv4/wo_a'][:400]}", flush=True)
    doc["jit"] = jit(Path(env["DG_JIT_CACHE_DIR"]))
    print(f"[deepgemm] jit files={doc['jit']['files']} gemms={doc['jit']['gemms']} layouts={doc['jit']['layouts']} "
          f"types={doc['jit']['gemm_types']} ok={doc['jit']['ok']}", flush=True)
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = B._error(e)
        doc["bind"] = {"ok": False}
    print(f"[deepgemm] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            key = _case(res) if res["phase"] == "weights" or res.get("M") is not None else f"{res['phase']}/{res['model']}/{res['case']}"
            if "error" in res:
                doc["errors"][f"compare/{key}"] = res["error"]
                continue
            agg = results.setdefault(key, {"phase": res["phase"], "model": res["model"], "case": res["case"], "M": res.get("M"),
                                           **{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        _print_case(key, c)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DEEPGEMM {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepGEMM's and FlashInfer's sm_90 block-FP8 linears and the wo_a einsum against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, JIT caches); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"linear M in {QUICK_LINEAR_M}, einsum M in {QUICK_EINSUM_M}: a first signal")
    ap.add_argument("--batch-invariant", action="store_true", help="VLLM_BATCH_INVARIANT=1: every M takes DeepGEMM's branch")
    ap.add_argument("--only", default=",".join(PHASES), help=f"comma list of phases: {', '.join(PHASES)} (a GO needs all three)")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, a.batch_invariant, set(a.only.split(",")))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
