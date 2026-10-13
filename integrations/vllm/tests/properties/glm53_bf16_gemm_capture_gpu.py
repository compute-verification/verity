"""GLM-5.3's and DeepSeek-V4-Pro's bf16 GEMMs on Hopper against their Programs: the H200's go/no-go for binding the MoE router gate
(`GateLinear`: cuteDSL `ll_bf16_gemm` at M <= 16, `ll_bf16.LlBf16DotprodGemm_v1` or `LlBf16SplitKGemm_v1`; past it cuBLAS
``torch.mm(x, W.T, out_dtype=float32)``, `cublas_bf16.GemmF32Out_v1`), GLM-5.3's MLA absorbs (``torch.bmm`` against the served
`W_UK_T` and `W_UV`, `cublas_bf16.BatchedGemm_v1`), DeepSeek-V4-Pro's compressor GEMMs (the same ``torch.mm`` on bf16,
`GemmF32Out_v1`), and each model's lm_head shard and indexer projection (``F.linear``, core's `Gemm_v2`), at their TP8 shapes
(`MODELS`) at vLLM's default cuBLAS configuration, with the former split-K-off route kept as a named comparison variant.

One GPU process writes `<out>/glm53_bf16_gemm_capture.json`, the property record `glm53_bf16_gemm_capture` (`properties.record`),
and prints one line per case and then `GLM53-BF16-GEMM GO` or `NO-GO` with each model's verdict.  `ok` (GO) = every check below,
for every model run.  The device gate runs first and nothing else runs when it fails (exit 3).  A case that raises records its
error and fails the verdict without stopping the others.

The split-K-off variant (`SWITCH_ENV` and `switch()`) is vLLM's batch-invariant cuBLAS configuration on cc 9.0 without the rest of that mode
(determinism/batch_invariant.py:1076-1080 and 1127-1138): CUBLAS_WORKSPACE_CONFIG=:16:8 and CUBLASLT_WORKSPACE_SIZE=1 before torch
creates a cuBLAS handle, fp16 and bf16 reduced-precision reductions off with split-K off (``(False, False)``, torch >= 2.10), and
cuBLASLt preferred.  vLLM's Triton bmm override is not part of it: `torch.bmm` stays on cuBLAS.  `--served-default` runs without
the variant. By default the run checks the variant first and vLLM's served default in a second process; the outer verdict needs
both, so `served_default` is first-class rather than timing-only.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  each model's `GateLinear(K, N)` with a bf16 weight and f32 out allows tier 1 (ll_bf16_gemm) and tier 4 (cuBLAS) and
             neither fp32 tier; `dispatch_unquantized_gemm()` is `default_unquantized_gemm` (F.linear); the switch's flags read back.
  router     `GateLinear.forward` at M in the model's `router_m`, twice: every `LLBf16Gemm.dispatch` it makes is tapped and must be
             `bf16_gemm_check.dispatch_sm90`'s kernel at the gate's (K, N) (GLM-5.3, (6144, 256): M <= 4 dotprod at BS 128, 5..16
             split-K at 6, or 8 at M 9 and 11; DeepSeek-V4-Pro, (7168, 384): M 1, 2 dotprod at BS 256, whose K loop ends in a tail
             tile, M 3, 4 at BS 128, M 5 split-K at 6, M 6, 8, 12 at 8, the rest to 16 at 7; past 16 no ll_bf16 call), and every f32
             word against `gate_rows`.  The dotprod's contraction is decided here: from M = 4 on, row 3 is 2^-122 times a normal
             row (and `fma-probe` at M 1 is only such a row), so its products fall below f32's normal range, where FFMA and
             FMUL-then-FADD differ (on about a third of its words); `fma` is the one of the two whose model every dotprod word of
             the run equals, and the probe must tell them apart.
  linear     `dispatch_unquantized_gemm()` (F.linear) of each model's `linear` (GLM-5.3: `lm_head.shard` K 6144, N 19360 and
             `indexer.wk_weights_proj` N 160; DeepSeek-V4-Pro: `lm_head.shard` K 7168, N 16160 and `indexer.weights_proj` N 64) at M
             in `LINEAR_M`: every bf16 word against `linear_rows`.
  f32out     DeepSeek-V4-Pro's compressor GEMMs, ``torch.mm(x, fused_wkv_wgate.weight.T, out_dtype=torch.float32)`` at K 7168 and N
             2048 (ratio-4 layers' compressor), 1024 (ratio-128's) and 512 (the indexer's compressor), at M in `LINEAR_M`: every
             f32 word against `f32out_rows`.
  bmm        GLM-5.3's `q_nope_absorb` (``torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)``, 8 heads, K 192, N 512) and
             `v_absorb` (``torch.bmm(x, W_UV, out=...)``, K 512, N 256) at M in `LINEAR_M`, on `W_UK_T` / `W_UV` laid out as
             `MLAAttention.process_weights_after_loading` leaves them (views of the dequantized kv_b_proj weight's transpose,
             mla_attention.py:1154-1252): every word against `bmm_rows`.
  reference  on rows 0, 3 and M - 1 of every launch, three coordinates each, the plain-integer reference
             (`verity.ml.circuits.evaluation.evaluate`) of the word Definition the launch binds, which the model must equal.
  bind       every Definition binds at each model's statics.
  timing     each case's kernel time, the median of `REPS` CUDA-event-timed launches.
  profile    every CUDA kernel name and launch grid, the separate cuBLASLt reduction kernel, grid.z as `SPLIT_K`, and the runtime
             cuBLAS version. Each default split-K case includes a crafted cancellation row fit against the stated
             contiguous/f32/forward hypothesis and strided, reverse, tree and bf16-partial alternatives.

GO needs, for every model and both variants, every case run, 0 mismatches on every word, `fma` decided, every launch deterministic
and on its expected kernel, every profiled split-K launch and unique hypothesis fit pinned, and the references equal.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_bf16_gemm_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--no-default]
        [--models glm53,dsv4-pro] [--only router,linear,f32out,bmm]
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-bf16-gemm-capture/v3"
RECORD = "glm53_bf16_gemm_capture.json"
#: GLM-5.3's MLA at one TP8 rank: heads, the nope and rope query widths, the latent rank and the value width
NHL, QK_NOPE, QK_ROPE, KV_LORA, V_HEAD = 8, 192, 64, 512, 256
#: (name, K, N), NHL heads each
BMM = (("q_nope_absorb", QK_NOPE, KV_LORA), ("v_absorb", KV_LORA, V_HEAD))
#: the strides of `W_UK_T` ([NHL, 192, 512], row-major per head) and `W_UV` ([NHL, 512, 256], column-major per head) as views of the
#: dequantized kv_b_proj weight's transpose
SERVED_STRIDE = {"q_nope_absorb": [(QK_NOPE + V_HEAD) * KV_LORA, KV_LORA, 1], "v_absorb": [(QK_NOPE + V_HEAD) * KV_LORA, 1, KV_LORA]}


class Model(NamedTuple):
    """One model's GEMMs at one TP8 rank: the gate's K (hidden) and N (routed experts), the router's M (every kernel its
    `dispatch_sm90` names, and past 16), and the (name, K, N) of its bf16-out linears, its f32-out GEMMs and its absorbs."""
    hidden: int
    experts: int
    router_m: tuple[int, ...]
    quick_router_m: tuple[int, ...]
    linear: tuple[tuple[str, int, int], ...]
    f32out: tuple[tuple[str, int, int], ...]
    bmm: tuple[tuple[str, int, int], ...]


#: GLM-5.3 (vocab 154880 / 8; the indexer's wk 128 + weights 32) and DeepSeek-V4-Pro (vocab 129280 / 8; the indexer's 64 heads; the
#: compressors' fused kv and score rows, 2 * coff * head dim: 2 * 2 * 512 at ratio 4, 2 * 512 at ratio 128, 2 * 2 * 128 the indexer's)
MODELS = {
    "glm53": Model(6144, 256, (1, 2, 3, 4, 5, 8, 9, 11, 12, 16, 17, 32, 128, 256), (1, 4, 5, 9, 17),
                   (("lm_head.shard", 6144, 154880 // 8), ("indexer.wk_weights_proj", 6144, 128 + 32)), (), BMM),
    "dsv4-pro": Model(7168, 384, (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 17, 32, 128, 256), (1, 3, 5, 6, 7, 17),
                      (("lm_head.shard", 7168, 129280 // 8), ("indexer.weights_proj", 7168, 64)),
                      (("compressor.kv_score.c4", 7168, 4 * 512), ("compressor.kv_score.c128", 7168, 2 * 512),
                       ("indexer.compressor.kv_score", 7168, 4 * 128)), ()),
}
LINEAR_M, QUICK_M = (1, 4, 16, 128, 256), (1, 16)
#: the M = 1 dotprod row whose products are all below f32's normal range
PROBE = "fma-probe"
STRESS_SCALE = 2.0 ** -122
SWITCH_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
REPS = 20
#: lane-steps (output words x K/16) per comparison job
JOB_LANE_STEPS = 4_000_000
PHASES = ("router", "linear", "f32out", "bmm")


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(served_default: bool, triton_cache: Path) -> dict[str, str]:
    """The env, set before torch is imported: the switch's cuBLAS variables, or (`served_default`) none of them."""
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_BATCH_INVARIANT": "0", "TRITON_CACHE_DIR": str(triton_cache)}
    if served_default:
        for k in SWITCH_ENV:
            os.environ.pop(k, None)
    else:
        env.update(SWITCH_ENV)
    os.environ.update(env)
    return env


def switch(served_default: bool) -> dict[str, Any]:
    """The switch's torch flags (unless `served_default`), set before the first GEMM, read back."""
    import torch

    m = torch.backends.cuda.matmul
    if not served_default:
        m.allow_fp16_reduced_precision_reduction = (False, False)
        m.allow_bf16_reduced_precision_reduction = (False, False)
        torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    rec = {"torch": torch.__version__, "allow_fp16_reduced_precision_reduction": repr(m.allow_fp16_reduced_precision_reduction),
           "allow_bf16_reduced_precision_reduction": repr(m.allow_bf16_reduced_precision_reduction),
           "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
           "env": {k: os.environ.get(k) for k in SWITCH_ENV}, "cublas_version": cublas_version()}
    rec["ok"] = served_default or (rec["env"] == SWITCH_ENV and "cublaslt" in rec["preferred_blas_library"].lower()
                                   and all(v in ("(False, False)", "False") for v in (rec["allow_fp16_reduced_precision_reduction"],
                                                                                     rec["allow_bf16_reduced_precision_reduction"])))
    return rec


# ------------------------------------------------------------------------------------------------------------------ operands
def _bf16(gen, shape, scale: float):
    import torch

    return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)


def _activation(gen, M: int, K: int, probe: bool = False):
    """bf16 rows of randn * 2^U(-3, 3); from M >= 4 on, row 1 is zero, row 2 one outlier over tiny values and row 3 2^-122 times a
    normal row (its products with the weights are f32 subnormals); `probe`: every row the last kind."""
    import torch

    x = torch.randn(M, K, generator=gen, device="cuda") * torch.exp2(torch.empty(M, 1, device="cuda").uniform_(-3, 3, generator=gen))
    if probe:
        x = torch.randn(M, K, generator=gen, device="cuda") * STRESS_SCALE
    elif M >= 4:
        x[1] = 0
        x[2] *= 2.0 ** -9
        x[2, 7] = 2.0 ** 12
        x[3] = torch.randn(K, generator=gen, device="cuda") * STRESS_SCALE
    return x.to(torch.bfloat16).contiguous()


def splitk_probe_words(K: int, splits: int, output: str, N: int = 32, seed: int = 20261013) -> tuple[np.ndarray, np.ndarray, dict]:
    """Craft finite cancellation rows that distinguish every modeled split-K hypothesis at these statics."""
    from verity_vllm.program.registry import bf16_gemm_check as BC

    rng = np.random.default_rng(seed + K + splits + N + (output == "bf16"))
    values = np.array([s * 2.0 ** e for e in (-20, -10, -1, 0, 1, 10, 20, 30, 50) for s in (-1, 1)] + [0.0], np.float32)
    pool = (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    x = np.full((1, K), 0x3F80, np.uint16)
    w = np.zeros((N, K), np.uint16)
    w[:, 0:K:16] = rng.choice(pool, (N, K // 16))
    models = BC.cublas_splitk_hypotheses(x, w, splits, out=output)
    candidate = models[BC.CUBLAS_SPLITK_HYPOTHESIS]
    differ = {name: int(np.count_nonzero(candidate != words)) for name, words in models.items()}
    if any(v == 0 for name, v in differ.items() if name != BC.CUBLAS_SPLITK_HYPOTHESIS):
        raise ValueError(f"split-K probe does not distinguish every hypothesis: K={K}, splits={splits}, output={output}, {differ}")
    return x[0], w, differ


def _install_splitk_probe(x, w, splits: int, output: str) -> dict[str, Any]:
    """Install the crafted probe in one token row and as many leading W rows as this shape has."""
    import torch

    n = min(32, w.shape[0])
    px, pw, differ = splitk_probe_words(x.shape[1], splits, output, n)
    x[0].copy_(torch.from_numpy(px.view(np.int16)).view(torch.bfloat16).to(x.device))
    w[:n].copy_(torch.from_numpy(pw.view(np.int16)).view(torch.bfloat16).to(w.device))
    return {"columns": n, "discriminator": differ}


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


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


def _time_us(fn) -> float:
    """The median of `REPS` launches of `fn`, each timed by CUDA events, after three warm ones."""
    import torch

    for _ in range(3):
        fn()
    times = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return float(np.median(times))


def _grid(args: dict[str, Any]) -> list[int] | None:
    """A Kineto trace event's grid, accepting the spellings emitted by torch 2.13 and older profilers."""
    grid = args.get("grid")
    if isinstance(grid, (list, tuple)) and len(grid) == 3:
        return [int(v) for v in grid]
    if isinstance(grid, str):
        try:
            values = [int(v.strip()) for v in grid.strip("()[]").split(",")]
            return values if len(values) == 3 else None
        except ValueError:
            pass
    keys = next((ks for ks in (("gridX", "gridY", "gridZ"), ("grid_x", "grid_y", "grid_z"),
                               ("Grid X", "Grid Y", "Grid Z")) if all(k in args for k in ks)), None)
    return [int(args[k]) for k in keys] if keys else None


def _profile_record(trace: dict[str, Any]) -> dict[str, Any]:
    """Kernel names, grids, the main split-K count and any separate reduction kernel from a Kineto Chrome trace."""
    launches = []
    for e in trace.get("traceEvents", []):
        args = e.get("args") or {}
        grid = _grid(args)
        cat = str(e.get("cat", "")).lower()
        if grid is None or ("kernel" not in cat and "kernel" not in str(args.get("kernel type", "")).lower()):
            continue
        block_keys = next((ks for ks in (("blockX", "blockY", "blockZ"), ("block_x", "block_y", "block_z"),
                                         ("Block X", "Block Y", "Block Z")) if all(k in args for k in ks)), None)
        launches.append({"name": str(e.get("name", "")), "grid": grid,
                         "block": [int(args[k]) for k in block_keys] if block_keys else None})
    # `LLBf16SplitK` has its split count in the dispatch record and reduces its cluster ranks in-kernel.  This field describes
    # cuBLAS's otherwise opaque split-K route, whose main kernel is followed by a separate reduction kernel.
    main = [x for x in launches if "splitk" in x["name"].lower() and "reduce" not in x["name"].lower()
            and "cute_dsl" not in x["name"].lower()]
    reductions = [x for x in launches if "splitkreduce" in x["name"].lower()]
    split_k = main[-1]["grid"][2] if main and main[-1]["grid"][2] > 1 else None
    return {"launches": launches, "split_k": split_k, "reduction_kernels": reductions,
            "ok": not main or (split_k is not None and bool(reductions))}


def _kernels(fn, trace_path: Path) -> dict[str, Any]:
    """One launch's CUDA kernels and launch geometry, by Kineto's Chrome trace."""
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    return _profile_record(json.loads(trace_path.read_text()))


def cublas_version() -> int | None:
    """The runtime cuBLAS version selected by this process, via cuBLAS's public API."""
    paths = [p for p in (ctypes.util.find_library("cublas"), "libcublas.so.12", "libcublas.so") if p]
    lib = next((lib for p in paths if (lib := _load_library(p)) is not None), None)
    if lib is None:
        return None
    lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cublasCreate_v2.restype = ctypes.c_int
    lib.cublasGetVersion_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.cublasGetVersion_v2.restype = ctypes.c_int
    lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cublasDestroy_v2.restype = ctypes.c_int
    handle, version = ctypes.c_void_p(), ctypes.c_int()
    if lib.cublasCreate_v2(ctypes.byref(handle)) != 0:
        return None
    try:
        return int(version.value) if lib.cublasGetVersion_v2(handle, ctypes.byref(version)) == 0 else None
    finally:
        lib.cublasDestroy_v2(handle)


def _load_library(path: str):
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


# ------------------------------------------------------------------------------------------------------------------- selection
def make_gate(vllm_config, model: str):
    import torch
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from verity_vllm.program.frontend.vllm_meta import ensure_distributed

    ensure_distributed(vllm_config)                                                # ReplicatedLinear reads the TP group's rank
    m = MODELS[model]
    return GateLinear(m.hidden, m.experts, bias=False, out_dtype=torch.float32, params_dtype=torch.bfloat16,
                      prefix="model.layers.3.mlp.gate").cuda()


def selection(gate) -> dict[str, Any]:
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    rec = {"gate": {k: bool(getattr(gate, k, None)) for k in ("allow_ll_bf16_gemm", "allow_cublas_router_gemm", "allow_fp32_router_gemm",
                                                              "allow_bf16x3_router_gemm")},
           "weight_dtype": str(gate.weight.dtype), "linear_gemm": getattr(dispatch_unquantized_gemm(), "__name__", None)}
    rec["ok"] = (rec["gate"] == {"allow_ll_bf16_gemm": True, "allow_cublas_router_gemm": True, "allow_fp32_router_gemm": False,
                                 "allow_bf16x3_router_gemm": False}
                 and rec["weight_dtype"] == "torch.bfloat16" and rec["linear_gemm"] == "default_unquantized_gemm")
    return rec


# ----------------------------------------------------------------------------------------------------------------------- cases
def cases(quick: bool, only: set[str], models: list[str]) -> list[tuple[str, str, str, int]]:
    """(model, phase, case, M) for every launch."""
    out = []
    for model in models:
        m = MODELS[model]
        if "router" in only:
            out += [(model, "router", "mlp.gate", M) for M in (m.quick_router_m if quick else m.router_m)] + [(model, "router", PROBE, 1)]
        for phase in ("linear", "f32out", "bmm"):
            if phase in only:
                out += [(model, phase, name, M) for name, _, _ in getattr(m, phase) for M in (QUICK_M if quick else LINEAR_M)]
    return out


def case_key(model: str, phase: str, case: str, M: int) -> str:
    return f"{model}/{phase}/{case}/M{M}"


def expected_kernel(M: int, model: str = "glm53") -> dict[str, Any]:
    """The model's gate's kernel at M as `dispatch_sm90` names it, in the record's form."""
    from verity_vllm.program.registry import bf16_gemm_check as BC

    m = MODELS[model]
    kind, st = BC.dispatch_sm90(M, m.hidden, m.experts)
    return {"kind": kind, **st}


def absorb_weights(kv_b):
    """`W_UK_T` and `W_UV` from the dequantized kv_b_proj weight [NHL (192 + 256), 512] as `MLAAttention.process_weights_after_loading`
    makes them: views of its transpose, (L, N, P) permuted to (N, P, L) and (L, N, V) to (N, L, V)."""
    w_uk, w_uv = kv_b.T.view(KV_LORA, NHL, QK_NOPE + V_HEAD).split([QK_NOPE, V_HEAD], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)


class Weights:
    """The run's weights, each made and saved once: each model's gate W, each linear's and f32-out GEMM's W, and GLM-5.3's
    dequantized kv_b_proj weight with `W_UK_T` and `W_UV` as views of its transpose."""

    def __init__(self, work: Path, seed: int):
        self.work, self.seed, self.made = work, seed, {}

    def get(self, model: str, name: str):
        key = (model, name)
        if key not in self.made:
            import torch

            m = MODELS[model]
            gen = torch.Generator(device="cuda").manual_seed(self.seed + len(self.made))
            if name == "mlp.gate":
                w = _bf16(gen, (m.experts, m.hidden), 2.0 ** -6)
                self.made[key] = (w, _save(self.work / model / "w_gate", w=_u16(w)))
            elif name in {n for n, _, _ in m.linear + m.f32out}:
                K, N = next((k, n) for nm, k, n in m.linear + m.f32out if nm == name)
                w = _bf16(gen, (N, K), 2.0 ** -6)
                self.made[key] = (w, _save(self.work / model / f"w_{name}", w=_u16(w)))
            else:
                uk_t, uv = absorb_weights(_bf16(gen, (NHL * (QK_NOPE + V_HEAD), KV_LORA), 2.0 ** -5))
                self.made[model, "q_nope_absorb"] = (uk_t, _save(self.work / model / "w_q_nope_absorb", b=_u16(uk_t)))
                self.made[model, "v_absorb"] = (uv, _save(self.work / model / "w_v_absorb", b=_u16(uv)))
        return self.made[key]


class DispatchTap:
    """Records every `LLBf16Gemm.dispatch` of vLLM's two ll_bf16 instances (the M = 1 PDL one and the other)."""

    def __enter__(self):
        from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16 as L

        self.keys: list[dict[str, Any]] = []
        self.objs = [L.ll_bf16_gemm_kernel, L.ll_bf16_gemm_c1_pdl_kernel]
        for o in self.objs:
            inner = o.dispatch

            def tap(*, M, K, N, _inner=inner, _pdl=o is L.ll_bf16_gemm_c1_pdl_kernel):
                key = _inner(M=M, K=K, N=N)
                self.keys.append({"M": M, "K": K, "N": N, "pdl": _pdl, "backend": key.backend, "bs": key.bs, "split_k": key.split_k,
                                  "num_stages": key.num_stages})
                return key

            o.dispatch = tap
        return self

    def __exit__(self, *exc):
        for o in self.objs:
            o.__dict__.pop("dispatch", None)


def _kernel_of(keys: list[dict[str, Any]]) -> dict[str, Any]:
    """The ll_bf16 kernel one launch dispatched, in `expected_kernel`'s form; cuBLAS when it made no ll_bf16 call."""
    if not keys:
        return {"kind": "cublas"}
    k = keys[-1]
    if k["backend"] == "dotprod":
        return {"kind": "dotprod", "BS": k["bs"], "VEC": 8}
    return {"kind": "splitk", "SPLIT_K": k["split_k"], "TILE_K": 256, "WARPS": 4}


def _cublas_split_k(job: dict[str, Any]) -> int | None:
    """The profiler's cuBLAS split count, never the split count of the independently described cuteDSL kernel."""
    split_k = job.get("split_k")
    return split_k if split_k and (job.get("kernel") or {}).get("kind") != "splitk" else None


def launch(work: Path, weights: Weights, gates: dict[str, Any], model: str, phase: str, name: str, M: int,
           seed: int, served_default: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """One case on the GPU, twice, timed and profiled; returns (launch record, comparison job)."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    rec: dict[str, Any] = {"model": model, "phase": phase, "case": name, "M": M}
    d = work / model / f"{phase}_{name}_M{M}"
    probe_w = None
    if phase == "router":
        m, gate = MODELS[model], gates[model]
        w, wpath = weights.get(model, "mlp.gate")
        gate.weight.data.copy_(w)
        x = _activation(gen, M, m.hidden, probe=name == PROBE)
        probe_w = gate.weight.data
        with DispatchTap() as tap:
            outs = [gate(x)[0].clone() for _ in range(2)]
        rec.update(K=m.hidden, N=m.experts, kernel=_kernel_of(tap.keys), expected=expected_kernel(M, model), dispatches=tap.keys)
        rec["dispatch_ok"] = (rec["kernel"] == rec["expected"] and len(tap.keys) == (0 if M > 16 else 2)
                              and all(k["pdl"] == (M == 1) and (k["K"], k["N"]) == (m.hidden, m.experts) for k in tap.keys))
        rec["operands_ok"] = outs[0].dtype == torch.float32 and list(outs[0].shape) == [M, m.experts]
        fn = lambda: gate(x)                                                       # noqa: E731
        paths = {**wpath, **_save(d, x=_u16(x), w=_u16(gate.weight.data), out=_u32(outs[0]))}
    elif phase in ("linear", "f32out"):
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

        w, wpath = weights.get(model, name)
        K, N = w.shape[1], w.shape[0]
        x = _activation(gen, M, K)
        probe_w = w
        if phase == "linear":
            gemm, layer = dispatch_unquantized_gemm(), torch.nn.Module()
            fn = lambda: gemm(layer, x, w, None)                                   # noqa: E731
        else:
            fn = lambda: torch.mm(x, w.T, out_dtype=torch.float32)                 # noqa: E731
        outs = [fn().clone() for _ in range(2)]
        dtype = torch.bfloat16 if phase == "linear" else torch.float32
        rec.update(K=K, N=N, operands_ok=outs[0].dtype == dtype and list(outs[0].shape) == [M, N])
        paths = {**wpath, **_save(d, x=_u16(x), w=_u16(w), out=(_u16 if phase == "linear" else _u32)(outs[0]))}
    else:
        b, wpath = weights.get(model, name)
        K, N = b.shape[1], b.shape[2]
        if name == "q_nope_absorb":
            q = _activation(gen, M, NHL * (QK_NOPE + QK_ROPE)).view(M, NHL, QK_NOPE + QK_ROPE)
            a = q.split([QK_NOPE, QK_ROPE], dim=-1)[0].transpose(0, 1)
            run = lambda: torch.bmm(a, b).transpose(0, 1)                         # noqa: E731
            outs = [run().transpose(0, 1).clone() for _ in range(2)]
            fn = run
        else:
            a = _activation(gen, M, NHL * KV_LORA).view(M, NHL, KV_LORA).transpose(0, 1)
            buf = torch.empty(M, NHL * V_HEAD, device="cuda", dtype=torch.bfloat16)
            out = buf.view(M, NHL, V_HEAD).transpose(0, 1)
            run = lambda: torch.bmm(a, b, out=out)                                 # noqa: E731
            outs = []
            for _ in range(2):
                run()
                outs.append(out.clone())
            fn = run
        rec.update(K=K, N=N, b_stride=list(b.stride()), a_stride=list(a.stride()))
        rec["operands_ok"] = (list(outs[0].shape) == [NHL, M, N] and outs[0].dtype == torch.bfloat16
                              and rec["b_stride"] == SERVED_STRIDE[name])
        paths = {**wpath, **_save(d, a=_u16(a), out=_u16(outs[0]))}
    torch.cuda.synchronize()
    rec["deterministic"] = bool(torch.equal(outs[0], outs[1]))
    rec["us"] = _time_us(fn)
    try:
        rec["profile"] = _kernels(fn, d / "profile.json")
    except Exception as e:  # noqa: BLE001
        rec["profile"] = {"error": _error(e), "ok": False}
    probe_paths, probe = None, None
    profiled_split_k = rec["profile"].get("split_k")
    if served_default and probe_w is not None and _cublas_split_k({"split_k": profiled_split_k, "kernel": rec.get("kernel")}):
        n = min(32, probe_w.shape[0])
        old_w = probe_w[:n].clone()
        output = "f32" if phase in ("router", "f32out") else "bf16"
        probe = _install_splitk_probe(x, probe_w, profiled_split_k, output)
        probe_out = fn()
        torch.cuda.synchronize()
        probe_paths = _save(d / "splitk_probe", x=_u16(x[0:1]), w=_u16(probe_w[:n]),
                            out=(_u32 if output == "f32" else _u16)(probe_out[0:1, :n]))
        probe_w[:n].copy_(old_w)
    rec["splitk_probe"] = probe
    job = {"model": model, "phase": phase, "case": name, "M": M, "K": rec["K"], "N": rec["N"], "kernel": rec.get("kernel"),
           "paths": paths, "split_k": profiled_split_k, "probe_paths": probe_paths}
    return rec, job


def jobs_of(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Row chunks of one launch; the reference runs on rows 0, 3 (the stress row) and M - 1."""
    M = job["M"]
    ref = sorted({r for r in (0, 3, M - 1) if r < M})
    per_row = job["N"] * job["K"] // 16 * (NHL if job["phase"] == "bmm" else 1)
    return [{**job, "rows": [r0, r1], "reference_rows": [r for r in ref if r0 <= r < r1]} for r0, r1 in _chunks(M, per_row)]


def dump_default_operands(outdir: Path, jobs: list[dict[str, Any]], launches: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve each cuBLAS split-K launch's exact operands and output for offline fitting; one W per case, shared across its M."""
    launch_by_case = {_case(x): x for x in launches}
    first_job: dict[str, dict[str, Any]] = {}
    for job in jobs:
        first_job.setdefault(_case(job), job)
    root, entries = outdir / "default_operands", {}
    for key, job in first_job.items():
        launch = launch_by_case[key]
        profile = launch.get("profile") or {}
        if not profile.get("reduction_kernels"):
            continue
        case_root = root / job["model"] / job["phase"] / job["case"]
        row_root = case_root / f"M{job['M']}"
        saved = {}
        for name, raw in job["paths"].items():
            dst = (case_root if name in ("w", "b") else row_root) / f"{name}.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(raw, dst)
            saved[name] = str(dst.relative_to(outdir))
        entries[key] = {"K": job["K"], "N": job["N"], "kernel": job.get("kernel"), "profile": profile, "paths": saved}
    manifest = {"schema": "verity-vllm/glm53-bf16-gemm-default-operands/v1", "cases": entries}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return {"manifest": str(path.relative_to(outdir)), "cases": len(entries)}


# ---------------------------------------------------------------------------------------------------------------------- compare
TOTALS = ("words", "mismatch", "dotprod_words", "fma1_mismatch", "fma0_mismatch", "fma_differ", "reference_coords", "twin_mismatch")


def _reference(job: dict[str, Any], a: dict[str, np.ndarray], r: int, n: int, fma: int | None) -> tuple[int, ...]:
    """The plain-integer reference of the word Definition at one output coordinate (row r, column n; head n // N for bmm)."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.ops import gemm as CG
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import cublas_bf16 as CB
    from verity_vllm.program.registry import ll_bf16 as LL

    hop = word_of(BG.HopperBF16WgmmaDot16)
    K, N = job["K"], job["N"]
    if job["phase"] == "bmm":
        h, c = divmod(n, N)
        return evaluate(bind(CB.BatchedGemmWord, B=1, K=K, N=1, DOT=hop), a["a"][h, r].tolist(), np.asarray(a["b"][h][:, c]).tolist())
    x = a["x"][r].tolist()
    w = a["w"][n].tolist()
    cublas_split_k = _cublas_split_k(job)
    if cublas_split_k:
        target = CB.CublasSplitKGemmF32OutWord if job["phase"] in ("router", "f32out") else CB.CublasSplitKGemmWord
        return evaluate(bind(target, K=K, N=1, SPLIT_K=cublas_split_k, DOT=hop), x, w)
    if job["phase"] == "linear":
        return evaluate(bind(CG.Gemm, K=K, N=1, DOT=hop), x, w)
    k = job.get("kernel") or {}
    if k.get("kind") == "dotprod":
        return evaluate(bind(LL.LlBf16DotprodDotWord, K=K, BS=k["BS"], VEC=k["VEC"], FMA=fma), x, w)
    if k.get("kind") == "splitk":
        return evaluate(bind(LL.LlBf16SplitKDotWord, K=K, SPLIT_K=k["SPLIT_K"], TILE_K=k["TILE_K"], WARPS=k["WARPS"], DOT=hop), x, w)
    return evaluate(bind(CB.GemmF32OutWord, K=K, N=1, DOT=hop), x, w)


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one launch: the kernel's words against the model of the Definition it binds (a dotprod gate launch against
    both contractions), and the reference on a sample."""
    try:
        return _compare(job)
    except Exception as e:  # noqa: BLE001
        return {"model": job["model"], "phase": job["phase"], "case": job["case"], "M": job["M"], "error": _error(e)}


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity_vllm.program.registry import bf16_gemm_check as BC
    from verity_vllm.program.registry import mla_check as MK

    a = _load(job["paths"])
    r0, r1 = job["rows"]
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    k = job.get("kernel") or {}
    if job["phase"] == "bmm":
        got = np.asarray(a["out"][:, r0:r1]).transpose(1, 0, 2).reshape(r1 - r0, -1)
        want = {None: BC.bmm_rows(np.asarray(a["a"][:, r0:r1]), np.asarray(a["b"])).transpose(1, 0, 2).reshape(r1 - r0, -1)}
    else:
        x, w, got = np.asarray(a["x"][r0:r1]), np.asarray(a["w"]), np.asarray(a["out"][r0:r1])
        cublas_split_k = _cublas_split_k(job)
        if cublas_split_k:
            output = "f32" if job["phase"] in ("router", "f32out") else "bf16"
            want = {None: BC.cublas_splitk_rows(x, w, cublas_split_k, out=output)}
        elif job["phase"] == "linear":
            want = {None: BC.linear_rows(x, w)}
        elif k.get("kind") == "dotprod":
            want = {f: BC.dotprod_rows(x, w, BS=k["BS"], VEC=k["VEC"], FMA=f) for f in (1, 0)}
        elif k.get("kind") == "splitk":
            want = {None: BC.splitk_rows(x, w, SPLIT_K=k["SPLIT_K"], TILE_K=k["TILE_K"], WARPS=k["WARPS"])}
        else:
            want = {None: BC.f32out_rows(x, w)}
    f32 = job["phase"] in ("router", "f32out")
    for f, wv in want.items():
        bad = ~MK._words_equal(got, wv, f32_words=f32)
        if f is None:
            rec["words"] += got.size
            rec["mismatch"] += int(bad.sum())
        else:
            rec["dotprod_words"] += got.size if f == 1 else 0
            rec[f"fma{f}_mismatch"] += int(bad.sum())
        if bad.any() and rec["first_mismatch"] is None and f in (None, 1):
            i, j = (int(v[0]) for v in np.nonzero(bad))
            rec["first_mismatch"] = {"row": r0 + i, "column": j, "fma": f, "want": int(wv[i, j]), "kernel": int(got[i, j])}
    if len(want) == 2:
        rec["fma_differ"] += int((~MK._words_equal(want[1], want[0], f32_words=True)).sum())
    cols = got.shape[1]
    for r in job["reference_rows"]:
        for n in sorted({0, cols - 1, (r * 7919) % cols}):
            for f, wv in want.items():
                y = _reference(job, a, r, n, f)
                rec["reference_coords"] += 1
                rec["twin_mismatch"] += not bool(MK._words_equal(np.asarray(y[0], wv.dtype), wv[r - r0, n], f32_words=f32))
    hypotheses = None
    cublas_split_k = _cublas_split_k(job)
    if cublas_split_k and r0 == 0 and job.get("probe_paths"):
        probe = _load(job["probe_paths"])
        output = "f32" if job["phase"] in ("router", "f32out") else "bf16"
        models = BC.cublas_splitk_hypotheses(np.asarray(probe["x"]), np.asarray(probe["w"]), cublas_split_k, out=output)
        actual = np.asarray(probe["out"])
        hypotheses = {
            name: int((~MK._words_equal(actual, words, f32_words=output == "f32")).sum())
            for name, words in models.items()
        }
    return {"model": job["model"], "phase": job["phase"], "case": job["case"], "M": job["M"], "rows": job["rows"],
            "hypothesis_mismatch": hypotheses, **rec}


# ------------------------------------------------------------------------------------------------------------------------ bind
def bind_check(models: list[str] = list(MODELS), launches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every Definition binds at each model's statics, including each default split count read from the profiler."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.ops import gemm as CG
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import bf16_gemm_check as BC
    from verity_vllm.program.registry import cublas_bf16 as CB
    from verity_vllm.program.registry import ll_bf16 as LL

    hop = word_of(BG.HopperBF16WgmmaDot16)
    targets = []
    for model in models:
        m = MODELS[model]
        H, E = m.hidden, m.experts
        gate = [[H], [E, H], [E]]
        kernels = [expected_kernel(M, model) for M in range(1, BC.LL_MAX_M + 1)]
        targets.append((bind(CB.GemmF32OutWord, K=H, N=E, DOT=hop), gate))
        for bs in sorted({k["BS"] for k in kernels if k["kind"] == "dotprod"}):
            targets += [(bind(LL.LlBf16DotprodGemmWord, N=E, K=H, BS=bs, VEC=BC.DOTPROD_VEC, FMA=f), gate) for f in (0, 1)]
        for s in sorted({k["SPLIT_K"] for k in kernels if k["kind"] == "splitk"}):
            targets.append((bind(LL.LlBf16SplitKGemmWord, N=E, K=H, SPLIT_K=s, TILE_K=BC.SPLITK_TILE_K, WARPS=BC.SPLITK_WARPS,
                                 DOT=hop), gate))
        targets += [(bind(CG.Gemm, K=K, N=N, DOT=hop), [[K], [N, K], [N]]) for _, K, N in m.linear]
        targets += [(bind(CB.GemmF32OutWord, K=K, N=N, DOT=hop), [[K], [N, K], [N]]) for _, K, N in m.f32out]
        targets += [(bind(CB.BatchedGemmWord, B=NHL, K=K, N=N, DOT=hop), [[NHL, K], [NHL, N, K], [NHL, N]]) for _, K, N in m.bmm]
    for x in launches or []:
        splits = x.get("profile", {}).get("split_k")
        if not splits:
            continue
        target = CB.CublasSplitKGemmF32OutWord if x["phase"] in ("router", "f32out") else CB.CublasSplitKGemmWord
        targets.append((bind(target, K=x["K"], N=x["N"], SPLIT_K=splits, DOT=hop), [[x["K"]], [x["N"], x["K"]], [x["N"]]]))
    out = {}
    for t, want in targets:
        params, ret = t.signature()
        out[t.id] = [list(type_shape(ty)) for _, ty in params] + [list(type_shape(ret))] == want
    return {"bindings": out, "ok": all(out.values())}


# -------------------------------------------------------------------------------------------------------------------- verdict
def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["deterministic"] and x.get("operands_ok", True) and x.get("dispatch_ok", True)
                and x.get("profile", {}).get("ok", True))


def fma_of(t: dict[str, Any]) -> int | None:
    """The dotprod's contraction the words decide: the one model every dotprod word equals, when the probe told them apart."""
    if t["dotprod_words"] == 0 or t["fma_differ"] == 0:
        return None
    ok = [f for f in (1, 0) if t[f"fma{f}_mismatch"] == 0]
    return ok[0] if len(ok) == 1 else None


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] + t["dotprod_words"] > 0 and t["mismatch"] == 0 and t["reference_coords"] > 0 and t["twin_mismatch"] == 0


def _case(x: dict[str, Any]) -> str:
    return case_key(x["model"], x["phase"], x["case"], x["M"])


def case_verdicts(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each launch's own verdict, so a NO-GO names what failed: its checks and its words, a dotprod launch's at the run's `fma`."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        out[_case(x)] = {"launch_ok": _launch_ok(x), "totals": {k: 0 for k in TOTALS}, "us": x.get("us"),
                         "kernel": x.get("kernel", {}).get("kind") if x["phase"] == "router" else x["phase"]}
    for r in doc.get("comparisons", []):
        c = out.get(_case(r))
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
    fma = doc.get("fma")
    failed = {"/".join(p[:4]) for p in (e.split("/") for e in doc["errors"]) if len(p) >= 4 and p[1] in PHASES}
    split = {_case(x) for x in doc["launches"] if x.get("profile", {}).get("split_k")}
    for key, c in out.items():
        t = c["totals"]
        fma_ok = t["dotprod_words"] == 0 or (fma is not None and t[f"fma{fma}_mismatch"] == 0)
        c["hypothesis_ok"] = key not in split or doc.get("splitk_hypotheses", {}).get(key, {}).get("ok") is True
        c["ok"] = c["launch_ok"] and _words_ok(t) and fma_ok and c["hypothesis_ok"] and key not in failed
    for key in sorted(failed - set(out)):
        out[key] = {"launch_ok": False, "totals": {k: 0 for k in TOTALS}, "ok": False}
    return out


def model_verdicts(doc: dict[str, Any]) -> dict[str, bool]:
    """Each model's verdict: every one of its expected cases launched and GO."""
    return {m: all(k in doc["cases"] and doc["cases"][k]["ok"] for k in doc["expected"] if k.startswith(m + "/"))
            for m in doc["models"]}


def _verdict(doc: dict[str, Any]) -> bool:
    return (doc["gate"]["ok"] and not doc["errors"] and doc["switch"].get("ok") is True
            and isinstance(doc["switch"].get("cublas_version"), int)
            and all(s.get("ok") is True for s in doc["selection"].values()) and bool(doc["selection"])
            and doc["bind"].get("ok") is True and {_case(x) for x in doc["launches"]} == set(doc["expected"])
            and all(_launch_ok(x) for x in doc["launches"]) and _words_ok(doc["totals"]) and doc.get("fma") is not None
            and all(c["ok"] for c in doc["cases"].values()) and all(model_verdicts(doc).values())
            and (not doc.get("require_served_default") or doc.get("served_default", {}).get("ok") is True))


# --------------------------------------------------------------------------------------------------------------------- default
def default_run(a: argparse.Namespace, outdir: Path) -> dict[str, Any]:
    """The first-class served-default capture in its own process (the variant is process-wide and set before the first handle)."""
    sub = outdir / "served_default"
    argv = [sys.executable, str(Path(__file__).resolve()), "--served-default", "--no-default", "--out", str(sub),
            "--workers", str(a.workers), "--only", a.only, "--models", a.models, "--min-driver", str(a.min_driver),
            "--want-cc", a.want_cc, "--want-sms", str(a.want_sms)]
    argv += (["--scratch", a.scratch] if a.scratch else []) + (["--quick"] if a.quick else [])
    argv += ["--dump-operands"] if a.dump_operands else []
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    (outdir / "served_default.log").write_text(proc.stdout + proc.stderr)
    doc = json.loads((sub / RECORD).read_text())
    return {"returncode": proc.returncode, "ok": doc["ok"], "digest": doc["digest"], "errors": sorted(doc["errors"]),
            "model_verdicts": doc.get("model_verdicts"), "cublas_version": doc.get("switch", {}).get("cublas_version"),
            "profiles": {_case(x): x.get("profile") for x in doc["launches"]},
            "splitk_hypotheses": doc.get("splitk_hypotheses"),
            "changed": sorted(k for k, c in doc["cases"].items() if not c["ok"]), "us": {k: c.get("us") for k, c in doc["cases"].items()}}


def _cost(doc: dict[str, Any]) -> dict[str, Any]:
    """Each case's kernel time under the switch over the default's."""
    base = doc.get("served_default", {}).get("us") or {}
    return {k: round(c["us"] / base[k], 3) for k, c in doc["cases"].items() if c.get("us") and base.get(k)}


# ------------------------------------------------------------------------------------------------------------------------- run
def run(a: argparse.Namespace) -> dict[str, Any]:
    only, models = set(a.only.split(",")), a.models.split(",")
    unknown = sorted(set(models) - set(MODELS)) + sorted(only - set(PHASES))
    if unknown:
        raise SystemExit(f"unknown models or phases: {unknown} (models {sorted(MODELS)}, phases {list(PHASES)})")
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    env = environment(a.served_default, work / "triton_cache")
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from verity_vllm.program.registry import bf16_gemm_check as BC

    tag = "GLM53-BF16-GEMM" + (" (served default)" if a.served_default else "")
    todo = cases(a.quick, only, models)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": a.quick, "served_default": a.served_default,
                           "variant": "served_default" if a.served_default else "split_k_off",
                           "require_served_default": not a.served_default and not a.no_default,
                           "models": models, "expected": [case_key(*c) for c in todo], "environment": G._environment(),
                           "gate": C.gate(a.min_driver, a.want_cc, a.want_sms), "errors": {}}
    print(f"[glm53-bf16-gemm] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print(f"{tag} GATE-FAIL", flush=True)
        raise SystemExit(3)
    try:
        doc["switch"] = switch(a.served_default)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["switch"] = _error(e)
        doc["switch"] = {"ok": False}
    print(f"[glm53-bf16-gemm] switch {json.dumps(doc['switch'])}", flush=True)
    doc["selection"], doc["launches"], jobs = {}, [], []
    vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(vllm_config):
        gates: dict[str, Any] = {}
        for model in models:
            try:
                gates[model] = make_gate(vllm_config, model)
                doc["selection"][model] = selection(gates[model])
            except Exception as e:  # noqa: BLE001
                doc["errors"][f"selection/{model}"] = _error(e)
                doc["selection"][model] = {"ok": False}
            print(f"[glm53-bf16-gemm] selection {model} {json.dumps(doc['selection'][model])}", flush=True)
        weights = Weights(work, 4000)
        for i, (model, phase, name, M) in enumerate(todo):
            key = case_key(model, phase, name, M)
            try:
                launch_rec, job = launch(work, weights, gates, model, phase, name, M, 7000 + i, a.served_default)
                doc["launches"].append(launch_rec)
                jobs += jobs_of(job)
                print(f"[glm53-bf16-gemm] {key:48s} K={launch_rec['K']:5d} N={launch_rec['N']:5d} kernel={launch_rec.get('kernel')} "
                      f"det={launch_rec['deterministic']} dispatch={launch_rec.get('dispatch_ok')} operands={launch_rec['operands_ok']} "
                      f"us={launch_rec['us']:.1f}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"][key] = _error(e)
                print(f"[glm53-bf16-gemm] {key} ERROR {doc['errors'][key][:400]}", flush=True)
        del weights
        torch.cuda.empty_cache()
    try:
        doc["bind"] = bind_check(models, doc["launches"])
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-bf16-gemm] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {a.workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            key = (res["model"], res["phase"], res["case"], res["M"])
            if "error" in res:
                doc["errors"][case_key(*key) + "/compare"] = res["error"]
                continue
            agg = results.setdefault(key, {"model": key[0], "phase": key[1], "case": key[2], "M": key[3], **{k: 0 for k in TOTALS},
                                           "first_mismatch": None, "hypothesis_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
            agg["hypothesis_mismatch"] = agg["hypothesis_mismatch"] or res.get("hypothesis_mismatch")
    doc["comparisons"] = list(results.values())
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["splitk_hypotheses"] = {}
    for r in doc["comparisons"]:
        if r.get("hypothesis_mismatch"):
            fit = sorted(k for k, v in r["hypothesis_mismatch"].items() if v == 0)
            doc["splitk_hypotheses"][_case(r)] = {
                "hypothesis": BC.CUBLAS_SPLITK_HYPOTHESIS,
                "mismatch": r["hypothesis_mismatch"],
                "fits": fit,
                "ok": fit == [BC.CUBLAS_SPLITK_HYPOTHESIS],
            }
    if a.dump_operands:
        doc["operand_dump"] = dump_default_operands(outdir, jobs, doc["launches"])
    doc["fma"] = fma_of(doc["totals"])
    doc["cases"] = case_verdicts(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-bf16-gemm] case {key:48s} {'GO' if c['ok'] else 'NO-GO':5s} {c.get('kernel')} words={t['words']}!={t['mismatch']} "
              f"dotprod={t['dotprod_words']} fma1!={t['fma1_mismatch']} fma0!={t['fma0_mismatch']} differ={t['fma_differ']} "
              f"ref={t['reference_coords']} twin!={t['twin_mismatch']} us={c.get('us')}", flush=True)
    if not a.served_default and not a.no_default:
        try:
            doc["served_default"] = default_run(a, outdir)
            doc["switch_cost"] = _cost(doc)
        except Exception as e:  # noqa: BLE001
            doc["served_default"] = {"error": _error(e)}
        print(f"[glm53-bf16-gemm] served default {json.dumps(doc['served_default'])[:1500]}", flush=True)
        print(f"[glm53-bf16-gemm] switch cost (us switched / default) {json.dumps(doc.get('switch_cost'))}", flush=True)
    doc["model_verdicts"] = model_verdicts(doc)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    verdicts = " ".join(f"{m}={'GO' if ok else 'NO-GO'}" for m, ok in doc["model_verdicts"].items())
    default_verdict = (" standalone" if a.served_default or a.no_default else
                       f" served_default={'GO' if doc.get('served_default', {}).get('ok') else 'NO-GO'}")
    print(f"{tag} {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} {verdicts} fma={doc['fma']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}{default_verdict}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's and DeepSeek-V4-Pro's bf16 GEMMs on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"each model's quick router M, the others at M in {QUICK_M}: a first signal in minutes")
    ap.add_argument("--served-default", action="store_true", help="torch's default cuBLAS configuration instead of the split-K-off switch")
    ap.add_argument("--no-default", action="store_true", help="skip the second process under torch's default configuration")
    ap.add_argument("--dump-operands", action="store_true",
                    help="preserve operands and outputs for launches with a separate split-K reduction, deduplicating W across M")
    ap.add_argument("--models", default=",".join(MODELS), help=f"comma list of models: {', '.join(MODELS)}")
    ap.add_argument("--only", default=",".join(PHASES), help=f"comma list of phases: {', '.join(PHASES)}")
    a = ap.parse_args(argv)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
