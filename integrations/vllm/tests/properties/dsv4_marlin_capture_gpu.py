"""DeepSeek-V4-Pro's Marlin W4A16 MXFP4 expert GEMMs on Hopper against their Programs: the H200's go/no-go for binding
`marlin_moe.MarlinMoeExpertGemm(W)_v1` (and through them `MarlinMoeSchedule_v1`, `MarlinMxfp4Coordinate(W)_v1`) to the two
`ops.moe_wna16_marlin_gemm` launches of `fused_marlin_moe` at one TP8 rank's shapes (E = 384, top_k 6, hidden 7168, I = 384).

One GPU process writes `<out>/dsv4_marlin_capture.json`, the property record `dsv4_marlin_capture` (`properties.record`), and prints
one line per case and then `DSV4-MARLIN GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else
runs when it fails (exit 3).  A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `select_deepseek_v4_mxfp4_moe_backend` on DeepSeek-V4-Pro's FusedMoEConfig (oracle/mxfp4.py:656) selects MARLIN /
             MarlinExperts, `get_marlin_input_dtype()` is None (W4A16: VLLM_MARLIN_INPUT_DTYPE unset) and Marlin's round-up leaves
             (7168, 384) as they are (oracle/mxfp4.py:746-751).
  binary     the SASS and PTX archs `_moe_C` carries (information: an `8.0+PTX` build has no sm_90 SASS, and the driver JIT-compiles
             it, so the kernel's registers, and with them blocks_per_sm, are the driver's).
  auto       on one layer of random logical weights (every e2m1 code; scale bytes 118..127, and experts 0..7 with 30% of their scale
             bytes from `EDGE_BYTES`: +0, +inf, overflow) repacked by `prepare_moe_mxfp4_layer_for_marlin`, `fused_marlin_moe` runs
             each case twice (`CASES`: decode batches of 1 to 16 tokens over experts 8..383, 16 tokens over 12 hot experts, and 8
             over the edge experts only); every `moe_wna16_marlin_gemm` launch is intercepted, its statics checked against the served
             path's (moe block 8, fp32 reduce, no atomic add, top_k and mul_topk_weights per launch, no bias, zeros, act order or
             global scale), its (pair -> block, expert) assignment against the routing, and its words between the two runs.
  kernel     the launch's thread config and blocks per SM are ops.cu's choice (`determine_exec_config`, ops.cu:476-490), invisible
             from the host and, on fp32 sums rounded once to bf16, almost never visible in the words: one torch.profiler session
             over every auto and sweep launch names each launch's kernel, `marlin_moe_wna16::Marlin<a, b, c, s, threads,
             thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, stages, group_blocks, is_zp_float>`, and its grid
             (sms * blocks_per_sm, ops.cu:488-533); the launch's config is (16 thread_k_blocks, 16 thread_n_blocks, grid / sms),
             a small-batch config at 1..4 blocks per SM, the same in both runs, over the served types, one m block of 8, 4 stages
             and groups of 2 k blocks.
  sweep      `fused_marlin_moe` once at every decode batch M of 1..16 under the same session: the served config of each launch kind
             at each M, which every auto launch's must be; `model` (information) the per-config limits (`MR.auto_config`'s
             `allow`, each in 0..4) under which ops.cu's rule gives every swept config.
  forced     on `FORCED` cases, each launch again at every candidate, forced (`thread_k`, `thread_n`, `blocks_per_sm`), into an output
             filled with `SENTINEL`, on the probe input (`probe`: random signs and mantissas over 4 binades, each row scaled by a
             power of two so the median exact column sum over its experts is 2^127.5, where fp32 partial sums overflow in some
             summation orders and not others): a launch ops.cu refuses, or that writes no word (a failed launch: its CUDA error is
             read and cleared), is recorded refused; an accepted one is compared word for word at its config, and the accepted
             configs' own words must fall in more than one class (the probe tells summation orders apart, so the twin's schedule,
             warp tree and fold are checked where they show).
  words      every word of every auto launch against the twin (`kernels.marlin_moe_rows.launch`) at its kernel's config, and of every
             accepted forced launch at its own.
  reference  on each auto launch, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of `MarlinMoeSchedule_v1`
             at the served statics on its first blocks, `MarlinMxfp4Coordinate(W)_v1` on sampled coordinates, and for the w2 launch
             `MarlinMoeExpertCoordinateW_v1{E=384}` on one column through the 384-expert gather, which the twin must equal.
  bind       `MarlinMoeExpertGemm(W)_v1` bind at every served config, and each names its Boolean version.

GO needs every launch at the served statics, deterministic and assigned as routed, every auto launch's kernel read and at its M's
swept config, a swept config per kind at every M of 1..16, at least one accepted forced launch per kind and more than one word class
wherever two were accepted (when the phase runs), 0 mismatched words, and every reference evaluation equal to the twin.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_marlin_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--skip-forced]
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import inspect
import itertools
import json
import multiprocessing
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties.glm53_block_fp8_capture_gpu import _activation, _error, _load, _save, _u16, _u32, cdiv
from verity_vllm.program.kernels import marlin_moe_rows as MR
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-marlin-capture/v1"
RECORD = "dsv4_marlin_capture.json"
#: one TP8 rank: hidden 7168, moe_intermediate 3072 / 8, 384 routed experts, top-6; DeepSeek-V4-Pro's swiglu_limit
HIDDEN, INTER, NUM_EXPERTS, TOP_K, SWIGLU_LIMIT = 7168, 384, 384, 6, 10.0
GROUP = 32
MOE_BLOCK = 8
#: (name, tokens, pool): `uniform` routes over experts EDGE_EXPERTS.., `hot` over 12 of them, `edge` over the edge experts only
CASES = (("decode1", 1, "uniform"), ("decode2", 2, "uniform"), ("decode3", 3, "uniform"), ("decode4", 4, "uniform"),
         ("decode8", 8, "uniform"), ("decode16", 16, "uniform"), ("hot16", 16, "hot"), ("edge8", 8, "edge"))
QUICK = ("decode1", "decode4", "hot16", "edge8")
FORCED, QUICK_FORCED = ("decode1", "hot16"), ("hot16",)
HOT = 12
#: experts 0..7 carry scale bytes from EDGE_BYTES on EDGE_FRACTION of their groups: +0 (code 0), the smallest, ones around 1, the
#: largest (6 * 2^127 overflows bf16) and 255 (+inf, so a +-0 code reads 0 * inf = NaN)
EDGE_EXPERTS, EDGE_FRACTION = 8, 0.3
EDGE_BYTES = (0, 1, 2, 125, 126, 127, 128, 252, 253, 254, 255)
SCALE_BYTES = (118, 128)
#: a bf16 NaN no launch writes (a NaN word out of the kernel is 0x7FFF)
SENTINEL = 0x7FA5
SERVED_M = tuple(range(1, 17))
CANDIDATES = tuple((tk, tn, b) for tk, tn, _ in MR.SMALL_BATCH_CONFIGS for b in (1, 2, 3, 4))
#: the probe's rows span PROBE_BINADES binades below 2; each is scaled so its median |exact column sum| is 2^PROBE_LOG2
PROBE_BINADES, PROBE_LOG2 = 4, 127.5
MARLIN = "marlin_moe_wna16::Marlin<"
#: the template statics ops.cu instantiates for this path: one m block (moe block 8 < 16), 4 stages (cc >= 8.0), group 32 = 2 k blocks
KERNEL_STATICS = {"thread_m_blocks": 1, "m_block_size_8": True, "stages": 4, "group_blocks": GROUP // 16, "is_zp_float": False}
TEMPLATE = ("a_type", "b_type", "c_type", "s_type", "threads", "thread_m_blocks", "thread_n_blocks", "thread_k_blocks", "m_block_size_8",
            "stages", "group_blocks", "is_zp_float")
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
#: coordinate-k steps (output words x K) per comparison job
JOB_COORD_K = 100_000_000
#: the launch arguments a W4A16 MXFP4 launch leaves None
ABSENT = ("b_bias", "a_scales", "global_scale", "b_qzeros", "g_idx", "perm")


def shape(kind: str) -> tuple[int, int]:
    """(K, N) of a launch kind: w13 [gate || up] over the hidden state, w2 down over the activation."""
    return (HIDDEN, 2 * INTER) if kind == "w13" else (INTER, HIDDEN)


def prob_m(kind: str, M: int) -> int:
    return M if kind == "w13" else M * TOP_K


def launch_top_k(kind: str) -> int:
    return TOP_K if kind == "w13" else 1


# ------------------------------------------------------------------------------------------------------------------------- env
def environment() -> dict[str, Any]:
    """DeepSeek-V4-Pro's serve env as it bears on the experts (the capture map's `target.env`), set before vLLM is imported; the
    serve leaves VLLM_MARLIN_INPUT_DTYPE unset (W4A16), so a value this shell holds is dropped and recorded."""
    env: dict[str, Any] = {"VLLM_BATCH_INVARIANT": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    os.environ.update(env)
    env["unset"] = {"VLLM_MARLIN_INPUT_DTYPE": os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)}
    return env


# ------------------------------------------------------------------------------------------------------------------- selection
def selection() -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        mxfp4_round_up_hidden_size_and_intermediate_size,
        select_deepseek_v4_mxfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_input_dtype

    cfg = FusedMoEConfig(num_experts=NUM_EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=INTER,
                         num_local_experts=NUM_EXPERTS, num_logical_experts=NUM_EXPERTS, activation=MoEActivation.SILU, device="cuda",
                         routing_method=RoutingMethodType.DeepseekV4, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                         in_dtype=torch.bfloat16, swiglu_limit=SWIGLU_LIMIT)
    backend, cls = select_deepseek_v4_mxfp4_moe_backend(cfg)
    rec = {"backend": getattr(backend, "name", str(backend)), "experts": getattr(cls, "__name__", None),
           "input_dtype": str(get_marlin_input_dtype()),
           "rounded": list(mxfp4_round_up_hidden_size_and_intermediate_size(Mxfp4MoeBackend.MARLIN, HIDDEN, INTER))}
    rec["ok"] = (rec["backend"] == "MARLIN" and rec["experts"] == "MarlinExperts" and rec["input_dtype"] == "None"
                 and rec["rounded"] == [HIDDEN, INTER])
    return rec


def binary() -> dict[str, Any]:
    """Information only: an error here is recorded in the phase, not the verdict."""
    import vllm

    so = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_moe_C*.so"))
    if not so:
        return {"error": "no vllm/_moe_C*.so"}
    try:
        rec = C._archs(so[0])
    except Exception as e:  # noqa: BLE001
        return {"so": so[0], "error": _error(e)}
    rec["sm90_sass"] = "90" in rec["sass_archs"] or "90a" in rec["sass_archs"]
    return rec


# ------------------------------------------------------------------------------------------------------------------------- weights
def weights(work: Path, seed: int) -> dict[str, Any]:
    """One layer's logical weights on the GPU, Marlin's repack of them, and the logical codes and scale bytes saved for the workers:
    codes [E][N][K] (every e2m1 code), scale bytes [E][N][K / 32] in SCALE_BYTES, the edge experts' partly from EDGE_BYTES; packed
    as the checkpoint is, two codes a byte, the even k in the low nibble (GPTQ's order, which `gptq_marlin_repack` reads)."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    gen = torch.Generator(device="cuda").manual_seed(seed)
    edge = torch.tensor(EDGE_BYTES, device="cuda", dtype=torch.uint8)
    logical, paths = {}, {}
    for kind in ("w13", "w2"):
        K, N = shape(kind)
        q = torch.randint(0, 16, (NUM_EXPERTS, N, K), generator=gen, device="cuda", dtype=torch.uint8)
        s = torch.randint(*SCALE_BYTES, (NUM_EXPERTS, N, K // GROUP), generator=gen, device="cuda", dtype=torch.uint8)
        pick = edge[torch.randint(0, len(EDGE_BYTES), (EDGE_EXPERTS, N, K // GROUP), generator=gen, device="cuda")]
        hit = torch.rand((EDGE_EXPERTS, N, K // GROUP), generator=gen, device="cuda") < EDGE_FRACTION
        s[:EDGE_EXPERTS] = torch.where(hit, pick, s[:EDGE_EXPERTS])
        logical[kind] = (q, s)
        paths[kind] = _save(work / f"weights_{kind}", wq=q.cpu().numpy(), ws=s.cpu().numpy())
    pack = {k: (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous() for k, (q, _) in logical.items()}
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    w13, w2, s13, s2, _, _ = prepare_moe_mxfp4_layer_for_marlin(layer, pack["w13"], pack["w2"], logical["w13"][1], logical["w2"][1],
                                                                None, None)
    del pack, logical
    torch.cuda.empty_cache()
    return {"w13": w13, "w2": w2, "s13": s13, "s2": s2, "paths": paths,
            "info": {"w13": list(w13.shape), "w2": list(w2.shape), "s13": [list(s13.shape), str(s13.dtype)],
                     "s2": [list(s2.shape), str(s2.dtype)]}}


# ------------------------------------------------------------------------------------------------------------------------- launches
class Interceptor:
    """Wraps `vllm._custom_ops.moe_wna16_marlin_gemm`, which `fused_marlin_moe` calls through the module: each launch's bound
    arguments, its input, routing tensors and output cloned as soon as it returns (the w2 launch writes over the w13 launch's
    output: both live in intermediate_cache13)."""

    def __enter__(self):
        from vllm import _custom_ops as ops

        self.ops, self.inner = ops, ops.moe_wna16_marlin_gemm
        self.sig = inspect.signature(self.inner)
        self.launches: list[dict[str, Any]] = []
        self.label = ""
        ops.moe_wna16_marlin_gemm = self.tap
        return self

    def __exit__(self, *exc):
        self.ops.moe_wna16_marlin_gemm = self.inner

    def tap(self, *args, **kwargs):
        import torch

        b = self.sig.bind(*args, **kwargs)
        b.apply_defaults()
        call = dict(b.arguments)
        out = self.inner(**call)
        keep = ("input", "sorted_token_ids", "expert_ids", "num_tokens_past_padded", "topk_weights")
        self.launches.append({
            "label": self.label, "output": out.clone(), **{k: call[k].clone() for k in keep},
            "call": {**call, **{k: call[k].clone() for k in keep}},
            "args": {k: (str(v) if k == "b_q_type" else v) for k, v in call.items() if not torch.is_tensor(v) and v is not None},
            "absent": sorted(k for k in ABSENT if call[k] is None),
            "dtypes": {k: str(call[k].dtype) for k in ("input", "b_scales", "topk_weights", "b_qweight")} | {"output": str(out.dtype)},
            "workspace": int(call["workspace"].numel())})
        return out


def kind_of(L: dict[str, Any]) -> str:
    return "w13" if L["args"].get("size_n") == 2 * INTER else "w2"


# ------------------------------------------------------------------------------------------------------------------------- kernels
def _template_value(s: str) -> int | bool:
    s = re.sub(r"^\([^)]*\)", "", s.strip())
    if s in ("true", "false"):
        return s == "true"
    m = re.match(r"-?\d+", s)
    if m is None:
        raise ValueError(f"template argument {s!r}")
    return int(m.group())


def kernel_template(name: str) -> dict[str, Any] | None:
    """The template arguments of a demangled `marlin_moe_wna16::Marlin<...>(...)` (kernel.h:27-45), by `TEMPLATE`'s names; None when
    the name is another kernel's or does not parse."""
    i = name.find(MARLIN)
    if i < 0:
        return None
    i += len(MARLIN)
    depth, j = 1, i
    while depth and j < len(name):
        depth += {"<": 1, ">": -1}.get(name[j], 0)
        j += 1
    args = name[i:j - 1].split(",")
    try:
        return dict(zip(TEMPLATE, (_template_value(a) for a in args), strict=True)) if not depth else None
    except ValueError:
        return None


def marlin_kernels(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Every Marlin MoE kernel of a torch.profiler chrome trace, in launch order (one stream): its template, grid, block, registers."""
    ev = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel" and MARLIN in e.get("name", "")]
    out = []
    for e in sorted(ev, key=lambda e: (e.get("ts", 0), e.get("args", {}).get("correlation", 0))):
        a = e.get("args", {})
        out.append({"name": e["name"][:600], "template": kernel_template(e["name"]), "grid": a.get("grid"), "block": a.get("block"),
                    "registers": a.get("registers per thread"), "shared": a.get("shared memory")})
    return out


def kernel_config(k: dict[str, Any] | None, sms: int, types: list[int] | None) -> tuple[list[int] | None, bool]:
    """(config, ok) of one launch's kernel: config [TK, TN, blocks per SM] = [16 thread_k_blocks, 16 thread_n_blocks, grid / sms]; ok
    when it is a small-batch config at 1..4 blocks per SM, launched at threads x 1 x 1 on a grid of sms * b x 1 x 1, with
    KERNEL_STATICS and, when given, the served [a, b, c, s] type ids."""
    t = (k or {}).get("template")
    if not t:
        return None, False
    grid, block = list(k.get("grid") or [0, 0, 0]), list(k.get("block") or [0, 0, 0])
    tk, tn = 16 * t["thread_k_blocks"], 16 * t["thread_n_blocks"]
    b = grid[0] // sms if sms and grid[0] % sms == 0 else 0
    ok = ((tk, tn, t["threads"]) in MR.SMALL_BATCH_CONFIGS and 1 <= b <= 4 and grid[1:] == [1, 1] and block == [t["threads"], 1, 1]
          and all(t[n] == v for n, v in KERNEL_STATICS.items())
          and (types is None or [t["a_type"], t["b_type"], t["c_type"], t["s_type"]] == types))
    return [tk, tn, b], bool(ok)


def served_types() -> list[int]:
    """The [a, b, c, s] ScalarType ids ops.cu instantiates for bf16 activations over e2m1 weights with e8m0 scales and a bf16 output."""
    from vllm.scalar_type import scalar_types

    return [scalar_types.bfloat16.id, scalar_types.float4_e2m1f.id, scalar_types.bfloat16.id, scalar_types.float8_e8m0fnu.id]


# ------------------------------------------------------------------------------------------------------------------------- probe
def dequant64(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The exact weights [..., K] of e2m1 codes over e8m0 scale bytes [..., K / 32] (byte 0 reads +0 and 255 +inf, as the bf16 view)."""
    s = np.asarray(s).astype(np.float64)
    sc = np.where(s == 255, np.inf, np.where(s == 0, 0.0, np.exp2(s - 127)))
    return E2M1[np.asarray(q)] * np.repeat(sc, GROUP, axis=-1)


def probe(rng: np.random.Generator, row_experts: list[np.ndarray], wq, ws) -> tuple[np.ndarray, list[int]]:
    """bf16 words [rows][K]: random signs and mantissas over the PROBE_BINADES binades below 2, row r then times 2^t_r so the median
    |exact column sum| over the columns of its experts (`row_experts[r]`, into wq [E][N][K] and ws [E][N][K / 32]) is 2^PROBE_LOG2,
    where fp32 partial sums overflow in some summation orders and not in others; returns (words, t)."""
    K, rows = wq.shape[-1], len(row_experts)
    sign = rng.integers(0, 2, (rows, K)).astype(np.int64) << 15
    exp = 127 - rng.integers(0, PROBE_BINADES, (rows, K)).astype(np.int64)
    man = rng.integers(0, 128, (rows, K)).astype(np.int64)
    val = ((sign | (exp << 7) | man).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    sums: list[list[np.ndarray]] = [[] for _ in range(rows)]
    for e in sorted({int(v) for es in row_experts for v in es}):
        w = dequant64(wq[e], ws[e])
        for r, es in enumerate(row_experts):
            sums[r] += [w @ val[r]] * int(np.count_nonzero(np.asarray(es) == e))
    shifts = []
    for r in range(rows):
        s = np.abs(np.concatenate(sums[r])) if sums[r] else np.zeros(0)
        s = s[np.isfinite(s)]
        m = float(np.median(s)) if s.size else 0.0
        shifts.append(int(np.clip(np.round(PROBE_LOG2 - np.log2(m)), PROBE_BINADES - 127, 127)) if m > 0 else 0)
    x = sign | ((exp + np.array(shifts, dtype=np.int64)[:, None]) << 7) | man
    return x.astype(np.uint16), shifts


def _routing(gen, M: int, pool):
    import torch

    ids = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:TOP_K]] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOP_K, generator=gen, device="cuda", dtype=torch.float32) + 0.05
    return ids.contiguous(), (w / w.sum(dim=1, keepdim=True) * 2.5).contiguous()


def _pool(gen, kind: str):
    import torch

    if kind == "edge":
        return torch.arange(EDGE_EXPERTS, device="cuda")
    pool = torch.arange(EDGE_EXPERTS, NUM_EXPERTS, device="cuda")
    return pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:HOT]] if kind == "hot" else pool


def assignment(st: np.ndarray, ex: np.ndarray, ntpp: int, ids_flat: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, int]:
    """(ok, blk, e, par) of one launch from `moe_align_block_size`'s outputs: each pair's moe block (its position in the sorted ids
    over 8) and that block's expert, and par = num_tokens_post_padded / 8; ok when every pair sits in exactly one block below par,
    whose expert routes it."""
    n = ids_flat.size
    blk, e, seen = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    par = ntpp // MOE_BLOCK
    for b in range(min(int(ex.size), par)):
        p = st[b * MOE_BLOCK:(b + 1) * MOE_BLOCK]
        p = p[(p >= 0) & (p < n)]
        blk[p], e[p] = b, ex[b]
        seen[p] += 1
    return bool(ntpp % MOE_BLOCK == 0 and np.all(seen == 1) and np.array_equal(e, ids_flat)), blk, e, par


def statics_ok(L: dict[str, Any], kind: str, M: int) -> bool:
    """The launch is the served path's (marlin_moe.py:136-230): moe block 8, the launch kind's top_k, router weight and sizes,
    fp32 reduce without atomic add, full K, the e2m1 weight type over e8m0 scales and bf16 activations, nothing optional."""
    K, N = shape(kind)
    a = L["args"]
    want = {"moe_block_size": MOE_BLOCK, "top_k": launch_top_k(kind), "mul_topk_weights": kind == "w2", "size_m": prob_m(kind, M),
            "size_n": N, "size_k": K, "is_k_full": True, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False,
            "b_q_type": "float4_e2m1f", "thread_k": -1, "thread_n": -1, "blocks_per_sm": -1}
    return (all(a.get(k) == v for k, v in want.items()) and L["absent"] == sorted(ABSENT)
            and L["dtypes"] == {"input": "torch.bfloat16", "b_scales": "torch.float8_e8m0fnu", "topk_weights": "torch.float32",
                                "b_qweight": "torch.int32", "output": "torch.bfloat16"})


def _launch_arrays(L: dict[str, Any], blk: np.ndarray, e: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    arrays = {"x": _u16(L["input"]), "C": _u16(L["output"]), "blk": blk, "e": e}
    if kind == "w2":
        arrays["g"] = _u32(L["topk_weights"]).reshape(-1)
    return arrays


def _inputs(seed: int, M: int, pool_kind: str) -> tuple:
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _activation(gen, M, HIDDEN)
    return (x, *_routing(gen, M, _pool(gen, pool_kind)))


def profiled_runs(work: Path, W: dict[str, Any], runs: list[tuple]) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Each (label, x, ids, topk_weights) of `runs` through `fused_marlin_moe` under one Interceptor and ONE profiler session (a
    session per launch stops recording device activity after about 20: fa2_target_capture_gpu.kernels_by_launch); returns (the
    launches, the Marlin kernels in launch order, errors by label)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    errors: dict[str, str] = {}
    with profile(activities=[ProfilerActivity.CUDA]) as prof, Interceptor() as icp:
        for label, x, ids, tw in runs:
            icp.label = label
            try:
                fused_marlin_moe(x, W["w13"], W["w2"], None, None, W["s13"], W["s2"], tw, ids,
                                 quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=NUM_EXPERTS)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                errors[label] = _error(e)
    path = work / "trace.json"
    prof.export_chrome_trace(str(path))
    kernels = marlin_kernels(json.loads(path.read_text()))
    path.unlink()
    return icp.launches, kernels, errors


def _kernel(k: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if k is None else {n: k[n] for n in ("template", "grid", "block", "registers", "shared", "name")}


def case_launches(work: Path, name: str, M: int, pool_kind: str, ids, launches: list[dict], W: dict[str, Any], sms: int,
                  types: list[int] | None) -> tuple[list[dict], list[dict], list[dict]]:
    """One case's records from its two runs' launches (labels `<name>/run1`, `<name>/run2`), each with its kernel; returns (launch
    records, launch specs for the jobs, the captured launches for the forced phase)."""
    import torch

    by = {(L["label"].split("/")[1], kind_of(L)): L for L in launches if L["label"].startswith(f"{name}/")}
    ids_flat = ids.cpu().numpy().astype(np.int64).reshape(-1)
    records, specs, captured = [], [], []
    for kind in ("w13", "w2"):
        L, twin = by.get(("run1", kind)), by.get(("run2", kind))
        if L is None or twin is None:
            raise RuntimeError(f"no {kind} launch in both runs: {sorted(by)}")
        ok, blk, e, par = assignment(L["sorted_token_ids"].cpu().numpy(), L["expert_ids"].cpu().numpy(),
                                     int(L["num_tokens_past_padded"].reshape(-1)[0]), ids_flat)
        K, N = shape(kind)
        config, kernel_ok = kernel_config(L.get("kernel"), sms, types)
        rec = {"phase": "auto", "case": name, "launch": kind, "M": M, "pool": pool_kind, "pairs": int(ids_flat.size), "K": K, "N": N,
               "prob_m": prob_m(kind, M), "par": par, "experts_used": int(np.unique(ids_flat).size), "args": L["args"],
               "absent": L["absent"], "dtypes": L["dtypes"], "workspace": L["workspace"],
               "deterministic": bool(torch.equal(L["output"], twin["output"])), "assignment_ok": ok, "statics_ok": statics_ok(L, kind, M),
               "kernel": _kernel(L.get("kernel")), "config": config, "kernel_ok": kernel_ok,
               "config_deterministic": config == kernel_config(twin.get("kernel"), sms, types)[0]}
        records.append(rec)
        specs.append({"phase": "auto", "case": name, "launch": kind, "config": None, "M": M, "K": K, "N": N, "E": NUM_EXPERTS,
                      "top_k": launch_top_k(kind), "pairs": int(ids_flat.size), "par": par, "weights": W["paths"][kind],
                      "paths": _save(work / f"{name}_{kind}", **_launch_arrays(L, blk, e, kind))})
        captured.append({"case": name, "launch": kind, "M": M, "L": L, "blk": blk, "e": e, "par": par})
    return records, specs, captured


def sweep_record(launches: list[dict], sms: int, types: list[int] | None) -> dict[str, Any]:
    """The served config of each launch kind at each M of SERVED_M, from the sweep's launches (labels `sweep/<M>`); `ok` when every
    (kind, M) has a launch at the served statics whose kernel reads; `model` the limits under which ops.cu's rule gives them."""
    served: dict[str, dict[str, list[int] | None]] = {"w13": {}, "w2": {}}
    statics = True
    for L in launches:
        if L["label"].startswith("sweep/"):
            M, kind = int(L["label"].split("/")[1]), kind_of(L)
            config, ok = kernel_config(L.get("kernel"), sms, types)
            served[kind][str(M)] = config if ok else None
            statics = statics and statics_ok(L, kind, M)
    ok = statics and all(served[k].get(str(M)) for k in served for M in SERVED_M)
    return {"served": served, "statics_ok": statics, "model": model(served, sms), "ok": bool(ok)}


def served_config(sweep: dict[str, Any], kind: str, M: int) -> list[int] | None:
    return sweep.get("served", {}).get(kind, {}).get(str(M))


def last_cuda_error() -> int | None:
    """`cudaGetLastError` of the CUDA runtime this process loaded, which also clears it, so that a forced launch the device refused
    does not surface at the next torch launch; None without a dynamically loaded runtime (a static one keeps its own error)."""
    with open("/proc/self/maps") as f:
        lib = next((ln.split()[-1] for ln in f if "libcudart" in ln and ln.split()[-1].startswith("/")), None)
    return None if lib is None else int(ctypes.CDLL(lib).cudaGetLastError())


def forced_launches(work: Path, W: dict[str, Any], cap: dict[str, Any], sms: int,
                    seed: int) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """One captured launch again at every candidate config, forced, on the probe input into a SENTINEL-filled output; returns
    (records, specs, the probe's record: its row shifts and the accepted configs grouped by identical words)."""
    import hashlib

    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

    kind, L = cap["launch"], cap["L"]
    K, N = shape(kind)
    tk_ = launch_top_k(kind)
    w = _load(W["paths"][kind])
    xs, shifts = probe(np.random.default_rng(seed), [cap["e"][r * tk_:(r + 1) * tk_] for r in range(cap["e"].size // tk_)], w["wq"], w["ws"])
    if list(xs.shape) != list(L["input"].shape):
        raise RuntimeError(f"probe {list(xs.shape)} for input {list(L['input'].shape)}")
    x = torch.from_numpy(xs.view(np.int16)).to(L["input"].device).view(torch.bfloat16)
    launches, specs, classes = [], [], {}
    for tk, tn, b in CANDIDATES:
        out = torch.full_like(L["output"].view(torch.int16), SENTINEL).view(torch.bfloat16)
        rec = {"phase": "forced", "case": cap["case"], "launch": kind, "M": cap["M"], "config": [tk, tn, b], "pairs": int(out.shape[0]),
               "K": K, "N": N, "refused": None}
        last_cuda_error()
        try:
            ops.moe_wna16_marlin_gemm(**{**L["call"], "input": x, "output": out, "workspace": marlin_make_workspace_new(out.device, 4),
                                         "thread_k": tk, "thread_n": tn, "blocks_per_sm": b})
            torch.cuda.synchronize()
            rec["cuda_error"] = last_cuda_error()
            written = bool((out.view(torch.int16) != SENTINEL).all())
            if rec["cuda_error"] or not written:
                rec["refused"] = f"cuda error {rec['cuda_error']}, every word written: {written}"
        except RuntimeError as e:
            rec["refused"] = _error(e)[:300]
            rec["cuda_error"] = last_cuda_error()
        rec["accepted"] = rec["refused"] is None
        launches.append(rec)
        if rec["accepted"]:
            words = _u16(out)
            rec["nonfinite"] = int(((words & 0x7F80) == 0x7F80).sum())
            classes.setdefault(hashlib.sha256(words.tobytes()).hexdigest(), []).append([tk, tn, b])
            specs.append({"phase": "forced", "case": cap["case"], "launch": kind, "config": [tk, tn, b], "statics": [tk, tn, sms * b],
                          "M": cap["M"], "K": K, "N": N, "E": NUM_EXPERTS, "top_k": tk_, "pairs": rec["pairs"],
                          "par": cap["par"], "weights": W["paths"][kind],
                          "paths": _save(work / f"forced_{cap['case']}_{kind}_{tk}_{tn}_{b}",
                                         **_launch_arrays({**L, "input": x, "output": out}, cap["blk"], cap["e"], kind))})
    accepted = sum(r["accepted"] for r in launches)
    probe_rec = {"case": cap["case"], "launch": kind, "shifts": shifts, "accepted": accepted, "classes": list(classes.values()),
                 "ok": accepted < 2 or len(classes) > 1}
    return launches, specs, probe_rec


# ---------------------------------------------------------------------------------------------------------------- comparisons
def _operands(job: dict[str, Any], p: np.ndarray) -> tuple:
    a, w = _load(job["paths"]), _load(job["weights"])
    x = np.asarray(a["x"])[p // job["top_k"]]
    g = np.asarray(a["g"])[p] if "g" in a else None
    return a, w, x, np.asarray(a["blk"])[p], np.asarray(a["e"])[p], g


def word_jobs(spec: dict[str, Any], statics: list[int]) -> list[dict[str, Any]]:
    return [{**spec, "job": "words", "statics": list(statics), "pair_range": [p0, p1]}
            for p0, p1 in _chunks(spec["pairs"], spec["N"] * spec["K"])]


def _chunks(n: int, cost: int) -> list[tuple[int, int]]:
    step = max(1, JOB_COORD_K // max(1, cost))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _twin(job: dict[str, Any], p: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, w, x, blk, e, g = _operands(job, p)
    TK, TN, GRID = job["statics"]
    want = MR.launch(x, blk, job["par"], e, w["wq"], w["ws"], cols, TK=TK, TN=TN, GRID=GRID, g=g)
    return want, np.asarray(a["C"])[p][:, cols]


def compare_words(job: dict[str, Any]) -> dict[str, Any]:
    p0, p1 = job["pair_range"]
    want, got = _twin(job, np.arange(p0, p1), np.arange(job["N"]))
    bad = np.argwhere(want != got)
    rec = {"words": int(want.size), "mismatch": len(bad), "first_mismatch": None}
    if len(bad):
        i, n = (int(v) for v in bad[0])
        rec["first_mismatch"] = {"pair": p0 + i, "column": n, "want": int(want[i, n]), "kernel": int(got[i, n])}
    return rec


def reference_job(spec: dict[str, Any], statics: list[int]) -> dict[str, Any]:
    return {**spec, "job": "reference", "statics": list(statics)}


def compare_reference(job: dict[str, Any]) -> dict[str, Any]:
    """The plain-integer reference against the twin at the launch's statics: the schedule on its first four blocks, two pairs' sampled
    coordinates, and for the w2 launch one coordinate through the expert gather over every expert's rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.kernels.derived_rows import f2fp_bf16
    from verity_vllm.program.registry import marlin_moe as MM

    TK, TN, GRID = job["statics"]
    K, N, E, P, par = job["K"], job["N"], job["E"], job["pairs"], job["par"]
    KT, NT = K // TK, N // TN
    routed = job["launch"] == "w2"
    _, w, x, blk, e, g = _operands(job, np.arange(P))
    rec = {"reference_schedules": 0, "reference_coords": 0, "reference_expert_coords": 0, "twin_mismatch": 0}
    sched = bind(MM.MarlinMoeSchedule, KT=KT, NT=NT, GRID=GRID)
    cuts = {}
    for b in sorted(set(blk.tolist()))[:4]:
        cuts[b] = MR.schedule(KT, NT, GRID, b, par)
        got = np.array(evaluate(sched, [b], [par]), dtype=bool).reshape(NT, KT - 1)
        rec["reference_schedules"] += 1
        rec["twin_mismatch"] += not np.array_equal(got, cuts[b])
    coord = bind(MM.MarlinMxfp4CoordinateW if routed else MM.MarlinMxfp4Coordinate, K=K, TK=TK, DOT=MM.DOT)
    gb = None if g is None else f2fp_bf16(np.asarray(g, dtype=np.uint32))
    for p in sorted({0, P - 1}):
        cut_p = cuts.get(int(blk[p]))
        if cut_p is None:
            cut_p = MR.schedule(KT, NT, GRID, int(blk[p]), par)
        for n in sorted({0, N - 1, (p * 7919) % N}):
            q, s, cut = w["wq"][e[p], n], w["ws"][e[p], n], cut_p[n // TN]
            lead = [x[p].tolist(), cut.astype(int).tolist()] + ([int(gb[p])] if routed else [])
            y = evaluate(coord, *lead, q.tolist(), s.tolist())
            twin = MR.coordinates(x[p][None], cut[None], q[None], s[None], TK=TK, g=None if gb is None else gb[p:p + 1])
            rec["reference_coords"] += 1
            rec["twin_mismatch"] += y != (int(twin[0]),)
    if routed:
        n = (P * 7919) % N
        cut = cuts.get(int(blk[0]), MR.schedule(KT, NT, GRID, int(blk[0]), par))[n // TN]
        fn = bind(MM.MarlinMoeExpertCoordinateW, E=E, K=K, TK=TK, DOT=MM.DOT)
        y = evaluate(fn, x[0].tolist(), cut.astype(int).tolist(), int(e[0]), int(gb[0]), np.asarray(w["wq"][:, n]).ravel().tolist(),
                     np.asarray(w["ws"][:, n]).ravel().tolist())
        twin = MR.coordinates(x[0][None], cut[None], w["wq"][e[0], n][None], w["ws"][e[0], n][None], TK=TK, g=gb[:1])
        rec["reference_expert_coords"] += 1
        rec["twin_mismatch"] += y != (int(twin[0]),)
    return rec


def compare(job: dict[str, Any]) -> dict[str, Any]:
    head = {k: job.get(k) for k in ("job", "phase", "case", "launch", "config", "statics", "pair_range")}
    try:
        return {**head, **{"words": compare_words, "reference": compare_reference}[job["job"]](job)}
    except Exception as e:  # noqa: BLE001
        return {**head, "error": _error(e)}


# ------------------------------------------------------------------------------------------------------------- the served config
def _auto(M: int, kind: str, sms: int, allow: dict[tuple[int, int], int]) -> tuple[int, int, int] | None:
    N = shape(kind)[1]
    try:
        return MR.auto_config(prob_m(kind, M), N, launch_top_k(kind), sms, allow)
    except ValueError:
        return None


def model(served: dict[str, dict[str, list[int] | None]], sms: int) -> dict[str, Any]:
    """Per launch kind, the per-config limits `allow` (each of the three configs' in 0..4, 0 invalid) under which ops.cu's rule
    (`MR.auto_config`) gives the swept config at every swept M: information on why the kernel runs what it runs (the limits read the
    shared-memory cache, which depends on K and N, and the registers of a driver-compiled kernel)."""
    out: dict[str, Any] = {}
    for kind, by_m in sorted(served.items()):
        obs = [(int(M), tuple(c)) for M, c in sorted(by_m.items(), key=lambda kv: int(kv[0])) if c]
        consistent = []
        for a in itertools.product(range(5), repeat=len(MR.SMALL_BATCH_CONFIGS)):
            allow = {(tk, tn): v for (tk, tn, _), v in zip(MR.SMALL_BATCH_CONFIGS, a) if v}
            if obs and all(_auto(M, kind, sms, allow) == c for M, c in obs):
                consistent.append(allow)
        out[kind] = {"observed": len(obs), "consistent": len(consistent),
                     "allow": [{f"{tk}/{tn}": v for (tk, tn), v in allow.items()} for allow in consistent[:8]]}
    return out


# ------------------------------------------------------------------------------------------------------------------------- bind
def bind_check(served: set[tuple[str, int, int, int]]) -> dict[str, Any]:
    """`MarlinMoeExpertGemm(W)_v1` bind at every served (kind, TK, TN, GRID): the weight slabs [E][N][K] and [E][N][K / 32], and the
    Boolean version each names."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity_vllm.program.boolean import boolean_version
    from verity_vllm.program.registry import marlin_moe as MM

    out: dict[str, Any] = {}
    for kind, TK, TN, GRID in sorted(served):
        K, N = shape(kind)
        fn = bind(MM.MarlinMoeExpertGemm if kind == "w13" else MM.MarlinMoeExpertGemmW, E=NUM_EXPERTS, K=K, N=N, TK=TK, TN=TN, GRID=GRID,
                  DOT=MM.DOT)
        params, ret = fn.signature()
        p = {k: list(type_shape(t)) for k, t in params}
        ok = p["wq"] == [NUM_EXPERTS, N, K] and p["ws"] == [NUM_EXPERTS, N, K // GROUP] and p["x"] == [K] and list(type_shape(ret)) == [N]
        out[descriptor_id(fn)] = {"ok": ok, "boolean": descriptor_id(boolean_version(fn))}
    return {"bindings": out, "ok": bool(out) and all(v["ok"] for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
TOTALS = ("words", "mismatch", "reference_schedules", "reference_coords", "reference_expert_coords", "twin_mismatch")


def _launch_ok(x: dict[str, Any]) -> bool:
    if x["phase"] == "forced":
        return True
    return bool(x["deterministic"] and x["statics_ok"] and x["assignment_ok"] and x["kernel_ok"] and x["config_deterministic"]
                and x.get("identified") is True)


def _words_ok(t: dict[str, Any]) -> bool:
    return (t["words"] > 0 and t["mismatch"] == 0 and t["reference_schedules"] > 0 and t["reference_coords"] > 0
            and t["reference_expert_coords"] > 0 and t["twin_mismatch"] == 0)


def _forced_ok(doc: dict[str, Any]) -> bool:
    """Each kind has an accepted forced launch, and every forced launch's probe tells two accepted configs apart."""
    forced = [x for x in doc["launches"] if x["phase"] == "forced"]
    if not doc.get("forced_phase"):
        return True
    probes = doc.get("probe", {})
    return (all(any(x["accepted"] for x in forced if x["launch"] == k) for k in ("w13", "w2"))
            and {f"{x['case']}/{x['launch']}" for x in forced} == set(probes) and all(p["ok"] for p in probes.values()))


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's own verdict: its launches' checks, the configs it ran (`auto` launches at their kernel's config, `forced` ones that
    were accepted), its probes and its words, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(x["case"], {"launches_ok": True, "totals": {k: 0 for k in TOTALS}, "configs": [], "refused": 0})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
        if x["phase"] == "auto":
            c["configs"].append(f"{x['launch']}:{'/'.join(map(str, x.get('config') or ['?']))}")
        else:
            c["refused"] += not x["accepted"]
    for p in doc.get("probe", {}).values():
        c = out.get(p["case"])
        if c is not None:
            c["launches_ok"] = c["launches_ok"] and p["ok"]
    for e in doc["errors"]:
        name = e.split("/")[1] if e.startswith("case/") else None
        if name and name not in out:
            out[name] = {"launches_ok": False, "totals": {k: 0 for k in TOTALS}, "configs": [], "refused": 0}
    for r in doc.get("comparisons", []):
        c = out.get(r["case"])
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
    for name, c in out.items():
        t = c["totals"]
        errors = [e for e in doc["errors"] if e == f"case/{name}" or e.startswith((f"case/{name}/", f"compare/{name}/"))]
        auto = any(x["phase"] == "auto" for x in doc["launches"] if x["case"] == name)
        c["ok"] = (c["launches_ok"] and auto and t["words"] > 0 and t["mismatch"] == 0 and t["twin_mismatch"] == 0
                   and t["reference_coords"] > 0 and not errors)
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    launches = doc["launches"]
    return (doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["bind"].get("ok") is True
            and doc["sweep"].get("ok") is True
            and {x["launch"] for x in launches if x["phase"] == "auto"} == {"w13", "w2"} and all(_launch_ok(x) for x in launches)
            and _forced_ok(doc) and _words_ok(doc["totals"]))


def _key(r: dict[str, Any]) -> tuple:
    return (r["phase"], r["case"], r["launch"], tuple(r["config"] or ()))


def _pool_map(ex, jobs: list[dict[str, Any]], doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for res in ex.map(compare, jobs):
        if "error" in res:
            doc["errors"][f"compare/{res['case']}/{res['launch']}/{res['job']}/{res['config']}"] = res["error"]
        else:
            out.append(res)
    return out


def _say(msg: str) -> None:
    print(f"[dsv4-marlin] {msg}", flush=True)


def gpu_phases(doc: dict[str, Any], work: Path, sms: int, quick: bool, forced: bool) -> tuple[list[dict], list[dict]]:
    """selection, binary, the weights, every case's auto launches and the forced ones; returns (auto specs, forced specs)."""
    for phase, fn in (("selection", selection), ("binary", binary)):
        try:
            doc[phase] = fn()
        except Exception as e:  # noqa: BLE001
            doc[phase] = {"ok": False}
            doc["errors"][phase] = _error(e)
        _say(f"{phase} {json.dumps(doc[phase])[:1200]}")
    try:
        W = weights(work, 10_000)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["weights"] = _error(e)
        _say(f"weights ERROR {doc['errors']['weights'][:600]}")
        return [], []
    doc["weights"] = W["info"]
    _say(f"weights {json.dumps(W['info'])}")
    try:
        types = served_types()
    except Exception as e:  # noqa: BLE001
        types = None
        doc["errors"]["types"] = _error(e)
    picked = [(i, name, M, pool) for i, (name, M, pool) in enumerate(CASES) if not quick or name in QUICK]
    runs, routed = [], {}
    for i, name, M, pool in picked:
        x, ids, tw = _inputs(11_000 + i, M, pool)
        routed[name] = ids
        runs += [(f"{name}/run1", x, ids, tw), (f"{name}/run2", x, ids, tw)]
    runs += [(f"sweep/{M}", *_inputs(12_000 + M, M, "uniform")) for M in SERVED_M]
    try:
        launches, kernels, run_errors = profiled_runs(work, W, runs)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["runs"] = _error(e)
        _say(f"runs ERROR {doc['errors']['runs'][:600]}")
        return [], []
    doc["errors"].update({label if label.startswith("sweep/") else f"case/{label}": err for label, err in run_errors.items()})
    doc["kernels"] = {"launches": len(launches), "kernels": len(kernels), "registers": sorted({str(k["registers"]) for k in kernels})}
    if len(kernels) == len(launches):
        for L, k in zip(launches, kernels):
            L["kernel"] = k
    else:
        doc["errors"]["kernels"] = f"{len(kernels)} Marlin kernels in the profile for {len(launches)} launches"
    _say(f"kernels {json.dumps(doc['kernels'])}")
    doc["sweep"] = sweep_record(launches, sms, types)
    _say(f"sweep ok={doc['sweep']['ok']} served={json.dumps(doc['sweep']['served'])} "
         f"model={json.dumps({k: v['consistent'] for k, v in doc['sweep']['model'].items()})}")
    specs, captured = [], []
    for _, name, M, pool in picked:
        try:
            records, sp, cap = case_launches(work, name, M, pool, routed[name], launches, W, sms, types)
        except Exception as e:  # noqa: BLE001
            doc["errors"][f"case/{name}"] = _error(e)
            _say(f"auto {name} ERROR {doc['errors'][f'case/{name}'][:600]}")
            continue
        doc["launches"] += records
        specs += sp
        captured += [c for c in cap if forced and name in (QUICK_FORCED if quick else FORCED)]
        for x in records:
            _say(f"auto {name:9s} {x['launch']:3s} M={M:2d} pairs={x['pairs']:3d} par={x['par']:3d} experts={x['experts_used']:3d} "
                 f"det={x['deterministic']} statics={x['statics_ok']} assignment={x['assignment_ok']} kernel={x['config']} "
                 f"kernel_ok={x['kernel_ok']}")
    forced_specs = []
    doc["probe"] = {}
    for j, cap in enumerate(captured):
        key = f"case/{cap['case']}/forced/{cap['launch']}"
        try:
            records, sp, pr = forced_launches(work, W, cap, sms, 13_000 + j)
        except Exception as e:  # noqa: BLE001
            doc["errors"][key] = _error(e)
            _say(f"forced {cap['case']} {cap['launch']} ERROR {doc['errors'][key][:600]}")
            continue
        doc["launches"] += records
        doc["probe"][f"{cap['case']}/{cap['launch']}"] = pr
        forced_specs += sp
        _say(f"forced {cap['case']:9s} {cap['launch']:3s} accepted={[x['config'] for x in records if x['accepted']]} "
             f"refused={[x['config'] for x in records if not x['accepted']]} probe classes={pr['classes']} shifts={pr['shifts']}")
    return specs, forced_specs


def compare_all(ex, doc: dict[str, Any], specs: list[dict], forced_specs: list[dict], sms: int) -> None:
    """Every auto launch's words at its kernel's config with its reference job, and every accepted forced launch's at its own;
    `doc["comparisons"]` per (phase, case, launch, config)."""
    auto = {(x["case"], x["launch"]): x for x in doc["launches"] if x["phase"] == "auto"}
    jobs = []
    for s in specs:
        x = auto[(s["case"], s["launch"])]
        x["served"] = served_config(doc["sweep"], s["launch"], s["M"])
        x["identified"] = bool(x["kernel_ok"] and x["config"] == x["served"])
        at = x["config"] if x["kernel_ok"] else None
        if at is not None:
            statics = [at[0], at[1], sms * at[2]]
            jobs += word_jobs({**s, "config": list(at)}, statics) + [reference_job({**s, "config": list(at)}, statics)]
    jobs += [j for s in forced_specs for j in word_jobs(s, s["statics"])]
    _say(f"comparing: {len(jobs)} jobs")
    results: dict[tuple, dict[str, Any]] = {}
    for res in _pool_map(ex, jobs, doc):
        agg = results.setdefault(_key(res), {"phase": res["phase"], "case": res["case"], "launch": res["launch"], "config": res["config"],
                                             **{k: 0 for k in TOTALS}, "first_mismatch": None})
        for k in TOTALS:
            agg[k] += res.get(k, 0)
        agg["first_mismatch"] = agg["first_mismatch"] or res.get("first_mismatch")
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        _say(f"words {r['phase']:6s} {r['case']:9s} {r['launch']:3s} {'/'.join(map(str, r['config'] or [])):10s} words={r['words']} "
             f"!={r['mismatch']} ref={r['reference_schedules']}+{r['reference_coords']}+{r['reference_expert_coords']} "
             f"twin!={r['twin_mismatch']} {json.dumps(r['first_mismatch']) if r['first_mismatch'] else ''}")


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool,
        forced: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment()
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "forced_phase": forced,
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": [],
                           "selection": {}, "sweep": {}, "probe": {}, "comparisons": []}
    _say(f"gate {json.dumps(doc['gate'])}")
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-MARLIN GATE-FAIL", flush=True)
        raise SystemExit(3)
    sms = int(doc["gate"]["sms"])
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        specs, forced_specs = gpu_phases(doc, work, sms, quick, forced)
    _say(f"comparing on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        compare_all(ex, doc, specs, forced_specs, sms)
    served = {(k, c[0], c[1], sms * c[2]) for k in ("w13", "w2") for M in SERVED_M for c in [served_config(doc["sweep"], k, M)] if c}
    try:
        doc["bind"] = bind_check(served)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    _say(f"bind {json.dumps(doc['bind'])[:1500]}")
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for name, c in doc["cases"].items():
        t = c["totals"]
        _say(f"case {name:9s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} !={t['mismatch']} "
             f"twin!={t['twin_mismatch']} forced_refused={c['refused']} {' '.join(c['configs'])}")
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    served1 = {k: served_config(doc["sweep"], k, 1) for k in ("w13", "w2")}
    print(f"DSV4-MARLIN {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} served@M1={json.dumps(served1)} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's Marlin MXFP4 expert GEMMs on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (logical weights, launch words); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"cases {QUICK}, forced {QUICK_FORCED}: a first signal in minutes")
    ap.add_argument("--skip-forced", action="store_true", help="no forced launches (their check is then not part of the verdict)")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, not a.skip_forced)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
