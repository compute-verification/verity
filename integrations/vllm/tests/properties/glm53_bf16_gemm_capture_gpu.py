"""GLM-5.3's bf16 GEMMs on Hopper against their Programs: the H200's first-hour go/no-go for binding the MoE router gate
(`GateLinear`: cuteDSL `ll_bf16_gemm` at M <= 16, `ll_bf16.LlBf16DotprodGemm_v1` or `LlBf16SplitKGemm_v1`; past it cuBLAS
``torch.mm(x, W.T, out_dtype=float32)``, `cublas_bf16.GemmF32Out_v1`), MLA's absorbs (``torch.bmm`` against the served `W_UK_T` and
`W_UV`, `cublas_bf16.BatchedGemm_v1`), and the lm_head shard and the indexer's `wk_weights_proj` (``F.linear``, core's `Gemm_v2`), at
GLM-5.3's TP8 shapes under the split-K-off cuBLAS switch.

One GPU process writes `<out>/glm53_bf16_gemm_capture.json`, the property record `glm53_bf16_gemm_capture` (`properties.record`),
and prints one line per case and then `GLM53-BF16-GEMM GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first
and nothing else runs when it fails (exit 3).  A phase that raises records its error and fails the verdict without stopping the
others.

The switch (`SWITCH_ENV` and `switch()`) is vLLM's batch-invariant cuBLAS configuration on cc 9.0 without the rest of that mode
(determinism/batch_invariant.py:1076-1080 and 1127-1138): CUBLAS_WORKSPACE_CONFIG=:16:8 and CUBLASLT_WORKSPACE_SIZE=1 before torch
creates a cuBLAS handle, fp16 and bf16 reduced-precision reductions off with split-K off (``(False, False)``, torch >= 2.10), and
cuBLASLt preferred.  vLLM's Triton bmm override is not part of it: `torch.bmm` stays on cuBLAS.  The served engine does not set it
yet (the capture map's `cublas-bf16-gemm` `when`); `--served-default` runs without it, and by default the run ends with that
configuration in a second process (`served_default`: which cases it changes, and each case's kernel time either way, the switch's
cost), which does not enter the verdict.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `GateLinear(6144, 256)` with a bf16 weight and f32 out allows tier 1 (ll_bf16_gemm) and tier 4 (cuBLAS) and neither fp32
             tier; `dispatch_unquantized_gemm()` is `default_unquantized_gemm` (F.linear); the switch's flags read back as set.
  router     `GateLinear.forward` at M in `ROUTER_M`, twice: every `LLBf16Gemm.dispatch` it makes is tapped and must be
             `bf16_gemm_check.dispatch_sm90`'s kernel (M <= 4 dotprod at BS 128, 5..16 split-K at 6, or 8 at M 9 and 11; past 16 no
             ll_bf16 call), and every f32 word against `gate_rows`.  The dotprod's contraction is decided here: from M = 4 on, row 3
             is 2^-122 times a normal row (and `fma-probe` at M 1 is only such a row), so its products fall below f32's normal range,
             where FFMA and FMUL-then-FADD differ (on about a third of its words); `fma` is the one of the two whose model every
             dotprod word equals, and the probe must tell them apart.
  linear     `dispatch_unquantized_gemm()` (F.linear) of `lm_head.shard` (K 6144, N 19360) and `indexer.wk_weights_proj` (K 6144,
             N 160) at M in `LINEAR_M`: every bf16 word against `linear_rows`.
  bmm        `q_nope_absorb` (``torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)``, 8 heads, K 192, N 512) and `v_absorb`
             (``torch.bmm(x, W_UV, out=...)``, K 512, N 256) at M in `BMM_M`, on `W_UK_T` / `W_UV` laid out as
             `MLAAttention.process_weights_after_loading` leaves them (views of the dequantized kv_b_proj weight's transpose,
             mla_attention.py:1154-1252): every word against `bmm_rows`.
  reference  on rows 0, 3 and M - 1 of every launch, three coordinates each, the plain-integer reference
             (`verity.ml.circuits.evaluation.evaluate`) of the word Definition the launch binds, which the model must equal.
  bind       every Definition binds at GLM-5.3's statics.
  timing     each case's kernel time, the median of `REPS` CUDA-event-timed launches.

GO needs 0 mismatches on every word, `fma` decided, every launch deterministic and on its expected kernel, and the references equal.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_bf16_gemm_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--no-default] [--only router,linear,bmm]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-bf16-gemm-capture/v1"
RECORD = "glm53_bf16_gemm_capture.json"
HIDDEN, EXPERTS = 6144, 256
NHL, QK_NOPE, QK_ROPE, KV_LORA, V_HEAD = 8, 192, 64, 512, 256
#: (name, K, N): the capture map's `cublas-bf16-gemm` linears at one TP8 rank (vocab 154880 / 8; the indexer's wk 128 + weights 32)
LINEAR = (("lm_head.shard", HIDDEN, 154880 // 8), ("indexer.wk_weights_proj", HIDDEN, 128 + 32))
#: (name, K, N), NHL heads each
BMM = (("q_nope_absorb", QK_NOPE, KV_LORA), ("v_absorb", KV_LORA, V_HEAD))
#: the strides of `W_UK_T` ([NHL, 192, 512], row-major per head) and `W_UV` ([NHL, 512, 256], column-major per head) as views of the
#: dequantized kv_b_proj weight's transpose
SERVED_STRIDE = {"q_nope_absorb": [(QK_NOPE + V_HEAD) * KV_LORA, KV_LORA, 1], "v_absorb": [(QK_NOPE + V_HEAD) * KV_LORA, 1, KV_LORA]}
#: every kernel the gate dispatches on sm_90: dotprod (1..4), split-K at the default 6 (5, 8, 12, 16) and the table's 8 (9, 11), cuBLAS
ROUTER_M = (1, 2, 3, 4, 5, 8, 9, 11, 12, 16, 17, 32, 128, 256)
LINEAR_M = BMM_M = (1, 4, 16, 128, 256)
QUICK_ROUTER_M, QUICK_M = (1, 4, 5, 9, 17), (1, 16)
#: the M = 1 dotprod row whose products are all below f32's normal range
PROBE = "fma-probe"
STRESS_SCALE = 2.0 ** -122
SWITCH_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
REPS = 20
#: lane-steps (output words x K/16) per comparison job
JOB_LANE_STEPS = 4_000_000
PHASES = ("router", "linear", "bmm")


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
           "env": {k: os.environ.get(k) for k in SWITCH_ENV}}
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


def _kernels(fn) -> list[str]:
    """The CUDA kernels one launch of `fn` runs, by the profiler's names."""
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")})


# ------------------------------------------------------------------------------------------------------------------- selection
def make_gate(vllm_config):
    import torch
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from verity_vllm.program.frontend.vllm_meta import ensure_distributed

    ensure_distributed(vllm_config)                                                # ReplicatedLinear reads the TP group's rank
    return GateLinear(HIDDEN, EXPERTS, bias=False, out_dtype=torch.float32, params_dtype=torch.bfloat16, prefix="model.layers.3.mlp.gate").cuda()


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
def cases(quick: bool, only: set[str]) -> list[tuple[str, str, int]]:
    """(phase, case, M) for every launch."""
    out = []
    if "router" in only:
        out += [("router", "mlp.gate", M) for M in (QUICK_ROUTER_M if quick else ROUTER_M)] + [("router", PROBE, 1)]
    if "linear" in only:
        out += [("linear", name, M) for name, _, _ in LINEAR for M in (QUICK_M if quick else LINEAR_M)]
    if "bmm" in only:
        out += [("bmm", name, M) for name, _, _ in BMM for M in (QUICK_M if quick else BMM_M)]
    return out


def expected_kernel(M: int) -> dict[str, Any]:
    """The gate's kernel at M as `dispatch_sm90` names it, in the record's form."""
    from verity_vllm.program.registry import bf16_gemm_check as BC

    kind, st = BC.dispatch_sm90(M, HIDDEN, EXPERTS)
    return {"kind": kind, **st}


def absorb_weights(kv_b):
    """`W_UK_T` and `W_UV` from the dequantized kv_b_proj weight [NHL (192 + 256), 512] as `MLAAttention.process_weights_after_loading`
    makes them: views of its transpose, (L, N, P) permuted to (N, P, L) and (L, N, V) to (N, L, V)."""
    w_uk, w_uv = kv_b.T.view(KV_LORA, NHL, QK_NOPE + V_HEAD).split([QK_NOPE, V_HEAD], dim=-1)
    return w_uk.permute(1, 2, 0), w_uv.transpose(0, 1)


class Weights:
    """The run's weights, each made and saved once: the gate's W, each linear's W, and the dequantized kv_b_proj weight with
    `W_UK_T` and `W_UV` as views of its transpose."""

    def __init__(self, work: Path, seed: int):
        self.work, self.seed, self.made = work, seed, {}

    def get(self, name: str):
        if name not in self.made:
            import torch

            gen = torch.Generator(device="cuda").manual_seed(self.seed + len(self.made))
            if name == "mlp.gate":
                w = _bf16(gen, (EXPERTS, HIDDEN), 2.0 ** -6)
                self.made[name] = (w, _save(self.work / "w_gate", w=_u16(w)))
            elif name in {n for n, _, _ in LINEAR}:
                K, N = next((k, n) for nm, k, n in LINEAR if nm == name)
                w = _bf16(gen, (N, K), 2.0 ** -6)
                self.made[name] = (w, _save(self.work / f"w_{name}", w=_u16(w)))
            else:
                uk_t, uv = absorb_weights(_bf16(gen, (NHL * (QK_NOPE + V_HEAD), KV_LORA), 2.0 ** -5))
                self.made["q_nope_absorb"] = (uk_t, _save(self.work / "w_q_nope_absorb", b=_u16(uk_t)))
                self.made["v_absorb"] = (uv, _save(self.work / "w_v_absorb", b=_u16(uv)))
        return self.made[name]


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


def launch(work: Path, weights: Weights, gate, phase: str, name: str, M: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One case on the GPU, twice, timed and profiled; returns (launch record, comparison job)."""
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device="cuda").manual_seed(seed)
    rec: dict[str, Any] = {"phase": phase, "case": name, "M": M}
    d = work / f"{phase}_{name}_M{M}"
    if phase == "router":
        w, wpath = weights.get("mlp.gate")
        gate.weight.data.copy_(w)
        x = _activation(gen, M, HIDDEN, probe=name == PROBE)
        with DispatchTap() as tap:
            outs = [gate(x)[0].clone() for _ in range(2)]
        rec.update(K=HIDDEN, N=EXPERTS, kernel=_kernel_of(tap.keys), expected=expected_kernel(M), dispatches=tap.keys)
        rec["dispatch_ok"] = (rec["kernel"] == rec["expected"] and len(tap.keys) == (0 if M > 16 else 2)
                              and all(k["pdl"] == (M == 1) for k in tap.keys))
        rec["operands_ok"] = outs[0].dtype == torch.float32 and list(outs[0].shape) == [M, EXPERTS]
        fn = lambda: gate(x)                                                       # noqa: E731
        paths = {**wpath, **_save(d, x=_u16(x), out=_u32(outs[0]))}
    elif phase == "linear":
        from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

        gemm = dispatch_unquantized_gemm()
        w, wpath = weights.get(name)
        K, N = w.shape[1], w.shape[0]
        x = _activation(gen, M, K)
        layer = torch.nn.Module()
        outs = [gemm(layer, x, w, None).clone() for _ in range(2)]
        rec.update(K=K, N=N, operands_ok=outs[0].dtype == torch.bfloat16 and list(outs[0].shape) == [M, N])
        fn = lambda: gemm(layer, x, w, None)                                       # noqa: E731
        paths = {**wpath, **_save(d, x=_u16(x), out=_u16(outs[0]))}
    else:
        b, wpath = weights.get(name)
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
        rec["kernels"] = _kernels(fn)
    except Exception as e:  # noqa: BLE001
        rec["kernels"] = {"error": _error(e)}
    job = {"phase": phase, "case": name, "M": M, "K": rec["K"], "N": rec["N"], "kernel": rec.get("kernel"), "paths": paths}
    return rec, job


def jobs_of(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Row chunks of one launch; the reference runs on rows 0, 3 (the stress row) and M - 1."""
    M = job["M"]
    ref = sorted({r for r in (0, 3, M - 1) if r < M})
    per_row = job["N"] * job["K"] // 16 * (NHL if job["phase"] == "bmm" else 1)
    return [{**job, "rows": [r0, r1], "reference_rows": [r for r in ref if r0 <= r < r1]} for r0, r1 in _chunks(M, per_row)]


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
    if job["phase"] == "linear":
        return evaluate(bind(CG.Gemm, K=K, N=1, DOT=hop), x, w)
    k = job["kernel"]
    if k["kind"] == "dotprod":
        return evaluate(bind(LL.LlBf16DotprodDotWord, K=K, BS=k["BS"], VEC=k["VEC"], FMA=fma), x, w)
    if k["kind"] == "splitk":
        return evaluate(bind(LL.LlBf16SplitKDotWord, K=K, SPLIT_K=k["SPLIT_K"], TILE_K=k["TILE_K"], WARPS=k["WARPS"], DOT=hop), x, w)
    return evaluate(bind(CB.GemmF32OutWord, K=K, N=1, DOT=hop), x, w)


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one launch: the kernel's words against the model of the Definition it binds (a dotprod gate launch against
    both contractions), and the reference on a sample."""
    try:
        return _compare(job)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "M": job["M"], "error": _error(e)}


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity_vllm.program.registry import bf16_gemm_check as BC
    from verity_vllm.program.registry import mla_check as MK

    a = _load(job["paths"])
    r0, r1 = job["rows"]
    N = job["N"]
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    k = job.get("kernel") or {}
    if job["phase"] == "bmm":
        got = np.asarray(a["out"][:, r0:r1]).transpose(1, 0, 2).reshape(r1 - r0, -1)
        want = {None: BC.bmm_rows(np.asarray(a["a"][:, r0:r1]), np.asarray(a["b"])).transpose(1, 0, 2).reshape(r1 - r0, -1)}
    else:
        x, w, got = np.asarray(a["x"][r0:r1]), np.asarray(a["w"]), np.asarray(a["out"][r0:r1])
        if job["phase"] == "linear":
            want = {None: BC.linear_rows(x, w)}
        elif k.get("kind") == "dotprod":
            want = {f: BC.dotprod_rows(x, w, BS=k["BS"], VEC=k["VEC"], FMA=f) for f in (1, 0)}
        elif k.get("kind") == "splitk":
            want = {None: BC.splitk_rows(x, w, SPLIT_K=k["SPLIT_K"], TILE_K=k["TILE_K"], WARPS=k["WARPS"])}
        else:
            want = {None: BC.f32out_rows(x, w)}
    f32 = job["phase"] == "router"
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
    return {"phase": job["phase"], "case": job["case"], "M": job["M"], "rows": job["rows"], **rec}


# ------------------------------------------------------------------------------------------------------------------------ bind
def bind_check() -> dict[str, Any]:
    """Every Definition binds at GLM-5.3's statics, its operands at the served shapes."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.circuits.ops import gemm as CG
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import bf16_gemm_check as BC
    from verity_vllm.program.registry import cublas_bf16 as CB
    from verity_vllm.program.registry import ll_bf16 as LL

    hop = word_of(BG.HopperBF16WgmmaDot16)
    gate = [[HIDDEN], [EXPERTS, HIDDEN], [EXPERTS]]
    targets = [(bind(CB.GemmF32OutWord, K=HIDDEN, N=EXPERTS, DOT=hop), gate)]
    targets += [(bind(LL.LlBf16DotprodGemmWord, N=EXPERTS, K=HIDDEN, BS=BC.DOTPROD_BS, VEC=BC.DOTPROD_VEC, FMA=f), gate) for f in (0, 1)]
    for s in sorted({BC.SPLITK_DEFAULT, *BC.SM90_SPLITK[(HIDDEN, EXPERTS)].values()}):
        targets.append((bind(LL.LlBf16SplitKGemmWord, N=EXPERTS, K=HIDDEN, SPLIT_K=s, TILE_K=BC.SPLITK_TILE_K, WARPS=BC.SPLITK_WARPS,
                             DOT=hop), gate))
    targets += [(bind(CG.Gemm, K=K, N=N, DOT=hop), [[K], [N, K], [N]]) for _, K, N in LINEAR]
    targets += [(bind(CB.BatchedGemmWord, B=NHL, K=K, N=N, DOT=hop), [[NHL, K], [NHL, N, K], [NHL, N]]) for _, K, N in BMM]
    out = {}
    for t, want in targets:
        params, ret = t.signature()
        out[t.id] = [list(type_shape(ty)) for _, ty in params] + [list(type_shape(ret))] == want
    return {"bindings": out, "ok": all(out.values())}


# -------------------------------------------------------------------------------------------------------------------- verdict
def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["deterministic"] and x.get("operands_ok", True) and x.get("dispatch_ok", True))


def fma_of(t: dict[str, Any]) -> int | None:
    """The dotprod's contraction the words decide: the one model every dotprod word equals, when the probe told them apart."""
    if t["dotprod_words"] == 0 or t["fma_differ"] == 0:
        return None
    ok = [f for f in (1, 0) if t[f"fma{f}_mismatch"] == 0]
    return ok[0] if len(ok) == 1 else None


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] + t["dotprod_words"] > 0 and t["mismatch"] == 0 and t["reference_coords"] > 0 and t["twin_mismatch"] == 0


def _case(x: dict[str, Any]) -> str:
    return f"{x['phase']}/{x['case']}/M{x['M']}"


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
    failed = {"/".join(e.split("/")[:3]) for e in doc["errors"] if e.split("/")[0] in PHASES}
    for key, c in out.items():
        t = c["totals"]
        fma_ok = t["dotprod_words"] == 0 or (fma is not None and t[f"fma{fma}_mismatch"] == 0)
        c["ok"] = c["launch_ok"] and _words_ok(t) and fma_ok and key not in failed
    for key in sorted(failed - set(out)):
        out[key] = {"launch_ok": False, "totals": {k: 0 for k in TOTALS}, "ok": False}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    return (doc["gate"]["ok"] and not doc["errors"] and doc["switch"].get("ok") is True and doc["selection"].get("ok") is True
            and doc["bind"].get("ok") is True and {x["phase"] for x in doc["launches"]} == set(PHASES)
            and all(_launch_ok(x) for x in doc["launches"]) and _words_ok(doc["totals"]) and doc.get("fma") is not None
            and all(c["ok"] for c in doc["cases"].values()))


# --------------------------------------------------------------------------------------------------------------------- default
def default_run(a: argparse.Namespace, outdir: Path) -> dict[str, Any]:
    """This capture under torch's default cuBLAS configuration, in a process of its own (the switch is process-wide and set before
    the first cuBLAS handle): which cases' words it changes, and each case's kernel time either way."""
    sub = outdir / "served_default"
    argv = [sys.executable, str(Path(__file__).resolve()), "--served-default", "--no-default", "--out", str(sub), "--workers", str(a.workers),
            "--only", a.only, "--min-driver", str(a.min_driver), "--want-cc", a.want_cc, "--want-sms", str(a.want_sms)]
    argv += (["--scratch", a.scratch] if a.scratch else []) + (["--quick"] if a.quick else [])
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    (outdir / "served_default.log").write_text(proc.stdout + proc.stderr)
    doc = json.loads((sub / RECORD).read_text())
    return {"returncode": proc.returncode, "ok": doc["ok"], "digest": doc["digest"], "errors": sorted(doc["errors"]),
            "changed": sorted(k for k, c in doc["cases"].items() if not c["ok"]), "us": {k: c.get("us") for k, c in doc["cases"].items()}}


def _cost(doc: dict[str, Any]) -> dict[str, Any]:
    """Each case's kernel time under the switch over the default's."""
    base = doc.get("served_default", {}).get("us") or {}
    return {k: round(c["us"] / base[k], 3) for k, c in doc["cases"].items() if c.get("us") and base.get(k)}


# ------------------------------------------------------------------------------------------------------------------------- run
def run(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    only = set(a.only.split(","))
    env = environment(a.served_default, work / "triton_cache")
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    tag = "GLM53-BF16-GEMM" + (" (served default)" if a.served_default else "")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": a.quick, "served_default": a.served_default,
                           "environment": G._environment(), "gate": C.gate(a.min_driver, a.want_cc, a.want_sms), "errors": {}}
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
    doc["selection"], doc["launches"], jobs = {"ok": False}, [], []
    vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(vllm_config):
        gate = None
        try:
            gate = make_gate(vllm_config)
            doc["selection"] = selection(gate)
        except Exception as e:  # noqa: BLE001
            doc["errors"]["selection"] = _error(e)
        print(f"[glm53-bf16-gemm] selection {json.dumps(doc['selection'])}", flush=True)
        weights = Weights(work, 4000)
        for i, (phase, name, M) in enumerate(cases(a.quick, only)):
            key = f"{phase}/{name}/M{M}"
            try:
                launch_rec, job = launch(work, weights, gate, phase, name, M, 7000 + i)
                doc["launches"].append(launch_rec)
                jobs += jobs_of(job)
                print(f"[glm53-bf16-gemm] {key:36s} K={launch_rec['K']:5d} N={launch_rec['N']:5d} kernel={launch_rec.get('kernel')} "
                      f"det={launch_rec['deterministic']} dispatch={launch_rec.get('dispatch_ok')} operands={launch_rec['operands_ok']} "
                      f"us={launch_rec['us']:.1f}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"][key] = _error(e)
                print(f"[glm53-bf16-gemm] {key} ERROR {doc['errors'][key][:400]}", flush=True)
        del weights
        torch.cuda.empty_cache()
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-bf16-gemm] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {a.workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"][f"{res['phase']}/{res['case']}/M{res['M']}/compare"] = res["error"]
                continue
            agg = results.setdefault((res["phase"], res["case"], res["M"]), {"phase": res["phase"], "case": res["case"], "M": res["M"],
                                                                             **{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["fma"] = fma_of(doc["totals"])
    doc["cases"] = case_verdicts(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-bf16-gemm] case {key:36s} {'GO' if c['ok'] else 'NO-GO':5s} {c.get('kernel')} words={t['words']}!={t['mismatch']} "
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
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"{tag} {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} fma={doc['fma']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's bf16 GEMMs on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"gate M in {QUICK_ROUTER_M}, the others at M in {QUICK_M}: a first signal in minutes")
    ap.add_argument("--served-default", action="store_true", help="torch's default cuBLAS configuration instead of the split-K-off switch")
    ap.add_argument("--no-default", action="store_true", help="skip the second process under torch's default configuration")
    ap.add_argument("--only", default=",".join(PHASES), help="comma list of phases: router, linear, bmm")
    a = ap.parse_args(argv)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
