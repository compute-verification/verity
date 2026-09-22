"""Interception (CONTRACT.md §4 rung 2, ``dispatch-mode``) for Model Runner V2 in eager replay.

Mechanism
---------
* A ``TorchDispatchMode`` is active around ``GPUModelRunner.execute_model`` and
  ``GPUModelRunner.sample`` (wrapped on the *instance* of the in-process model runner; vLLM
  source is not modified). It sees every aten op and every vLLM custom op
  (``_C.*``, ``vllm.*``) the forward executes, records the outputs of compute ops, and
  propagates identities through view/alias/copy ops.
* Two batch-invariant kernels are launched by vLLM straight from Python (Triton), invisible
  to any dispatch mode: ``matmul_persistent`` (every GEMM under ``VLLM_BATCH_INVARIANT=1``
  on SM8x) and the no-residual ``_rms_norm_kernel``. They are made dispatcher-visible by
  wrapping them as ``torch.ops.verity.bi_mm`` / ``torch.ops.verity.bi_rms_norm`` and
  rebinding the two module attributes vLLM calls
  (``batch_invariant.mm_batch_invariant`` and ``layernorm.rms_norm_batch_invariant``).
  Nothing else is patched; the list is in ``PATCHES``.
* ``nn.Module`` forward pre/post hooks maintain the module-path stack that gives every op
  its stable name (``layers.3.self_attn.qkv_proj.gemm``); they capture nothing themselves,
  and the post hook audits that every module output tensor was produced by a recorded op
  (a dispatcher-invisible kernel would show up here as an unbound module output).
* Snapshots: each recorded output is copied to pinned host memory on the forward stream at
  record time (``copy_`` ``non_blocking``), so a later in-place kernel or allocator reuse
  cannot change what was captured; the step is synchronised before hashing/writing.
* Batch -> (request, position): rows ``[qsl[i], qsl[i+1])`` of a per-token tensor belong to
  ``req_ids[i]`` at positions ``num_computed_tokens[i] + k``; logits rows map through
  ``logits_indices``. Padding rows are dropped.
* KV cache identity: K written at (t, layer) is ``rotary_emb.rope/port=1`` at t, V is the v
  column slice of ``qkv_proj.gemm/port=0`` at t. The slot each position is written to is
  taken from the forward context's slot mapping and recorded in an owner map; the block
  table of every request in the step is walked for positions ``0..t`` and each slot's owner
  must be ``(same request, t'')``. Any other owner (prefix-cache hit from another request,
  or a slot never written in this process) is an unresolved dependency and is recorded.
"""

from __future__ import annotations

import json
import re
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from verity_vllm import hashing
from verity_vllm_adapter.driver import external_id
from verity_vllm.capture import Boundary, CachedPrefix, RunWriter, StepRecord
from verity_vllm.identity import value_id

PATCHES = [
    "vllm.model_executor.determinism.batch_invariant.mm_batch_invariant -> torch.ops.verity.bi_mm (wraps matmul_persistent)",
    "vllm.model_executor.layers.layernorm.rms_norm_batch_invariant -> torch.ops.verity.bi_rms_norm for residual=None (wraps _rms_norm_kernel); residual path untouched (_C.fused_add_rms_norm)",
    "GPUModelRunner instance: execute_model/sample wrapped (dispatch mode + RU bookkeeping)",
    "vllm.model_executor.layers.fused_moe.experts.triton_moe.invoke_fused_moe_triton_kernel -> reporting wrapper (original launch unchanged; arguments and the kernel's output buffer are read after the launch)",
    "torch.ops.vllm.moe_forward / moe_forward_shared: body re-entered under a nested TorchDispatchMode (eager path; the custom op is opaque only for torch.compile)",
]

# ---------------------------------------------------------------------------------------
# op classification
# ---------------------------------------------------------------------------------------

ALIAS_OPS = {
    "aten.view", "aten._unsafe_view", "aten.reshape", "aten.t", "aten.transpose", "aten.permute",
    "aten.split", "aten.split_with_sizes", "aten.slice", "aten.select", "aten.unsqueeze",
    "aten.squeeze", "aten.expand", "aten.alias", "aten.detach", "aten.narrow", "aten.as_strided",
    "aten.unbind", "aten.unfold", "aten.view_as_real", "aten.flatten", "aten.lift_fresh",
    "aten.resolve_conj", "aten.resolve_neg", "aten.detach_",
}
COPY_OPS = {"aten.clone", "aten.contiguous", "aten.copy_", "aten._to_copy", "aten.to", "aten.copy"}
ALLOC_OPS = {
    "aten.empty", "aten.empty_like", "aten.empty_strided", "aten.zeros", "aten.zeros_like",
    "aten.ones", "aten.ones_like", "aten.full", "aten.full_like", "aten.arange", "aten.new_empty",
    "aten.new_zeros", "aten.new_ones", "aten.new_full", "aten.scalar_tensor", "aten.zero_",
    "aten.fill_", "aten.fill", "aten.empty_permuted", "aten.randn", "aten.rand",
}
SIDE_EFFECT_OPS = {"vllm.unified_kv_cache_update"}
IGNORE_OPS = {
    "aten._local_scalar_dense", "aten.item", "aten.is_nonzero", "aten.equal", "aten.sym_size",
    "aten.sym_numel", "aten.sym_stride", "aten.sym_storage_offset", "aten.record_stream",
    "aten.set_", "aten.resize_", "aten._has_compatible_shallow_copy_type", "aten.is_same_size",
    "aten._pin_memory", "aten.pin_memory", "aten.is_pinned",
}

# op family names (Table 2 op_id families / description op_type)
OP_FAMILY = {
    "verity.bi_mm": ("gemm", "gemm"),
    "aten.mm": ("gemm", "gemm"),
    "aten.addmm": ("gemm", "gemm"),
    "aten.matmul": ("gemm", "gemm"),
    "aten.linear": ("gemm", "gemm"),
    "aten.bmm": ("gemm", "gemm"),
    "aten.embedding": ("embedding", "embedding"),
    "verity.bi_rms_norm": ("rmsnorm", "rmsnorm"),
    "_C.rms_norm": ("rmsnorm", "rmsnorm"),
    "_C.fused_add_rms_norm": ("rmsnorm", "rmsnorm"),
    "_C.rotary_embedding": ("rope", "rope"),
    "_C.silu_and_mul": ("act_mul", "silu_mul"),
    "vllm.unified_attention_with_output": ("attn", "attention"),
    "vllm.unified_kv_cache_update": ("kv_write", "other"),
    "aten.add": ("add", "residual_add"),
    "aten.index": ("gather", "other"),
    "aten.index_select": ("gather", "other"),
    # MoE block (W9): the router's fused softmax+top-k, the routed-experts reduction, the shared
    # expert's sigmoid gate; the per-slot expert output comes from the Triton kernel wrapper
    "_moe_C.topk_softmax": ("router", "router"),
    "_moe_C.moe_sum": ("combine", "combine"),
    "aten.sigmoid": ("sigmoid", "other"),
    "aten.mul": ("mul", "other"),
    # tensor parallelism (lane TP): the collectives are dispatcher-visible custom ops on CUDA
    # (vllm.platforms.cuda.use_custom_op_collectives); their outputs are per-rank materialised values.
    # all_reduce: the sum of the two ranks' row-parallel partials (o_proj, down_proj, the vocab-
    # parallel embedding); all_gather: the concatenation of the two ranks' lm_head logit slices.
    # (description op_type: `allreduce` is in the schema's closed list; all_gather uses the schema's
    # `other` escape, its relation id says what it is)
    "vllm.all_reduce": ("all_reduce", "allreduce"),
    "vllm.all_gather": ("all_gather", "other"),
    # fused vocab-parallel embedding (TP > 1, unquantized): range mask + id shift + gather + zero-fill in one kernel
    "_C.vocab_parallel_embedding": ("embedding", "embedding"),
    # [QZ, additive] FP8 linear (lane QZ, port b1-fp8: `--quantization fp8` on sm_90, CutlassFP8ScaledMMLinearKernel):
    # the dynamic per-token activation quantisation (`out` = e4m3 words of the row, `scale` = the row's f32 scale;
    # both are materialised values of the token's computation and are recorded as ports 0 and 1 of one node) and the
    # CUTLASS scaled GEMM with the fused epilogue (`bf16(acc * s_x * s_w + bias)`; the bias is *inside* this node,
    # there is no separate aten.add for qkv_proj under this profile).  description op_type: `other` (schema escape)
    # for the quant, `gemm` for the scaled GEMM.
    "_C.dynamic_per_token_scaled_fp8_quant": ("fp8_quant", "other"),
    "_C.cutlass_scaled_mm": ("gemm", "gemm"),
}
# ops whose "outputs" are mutated arguments: op -> list of (arg index, port)
MUTATING_OUTPUTS = {
    "_C.rotary_embedding": [(1, 0), (2, 1)],  # (positions, query, key, ...) -> q, k rotated in place
    "_C.fused_add_rms_norm": [(0, 0), (1, 1)],  # (input, residual, weight, eps) -> normed, new residual
    "_C.rms_norm": [(0, 0)],  # (out, input, weight, eps)
    "_C.silu_and_mul": [(0, 0)],  # (out, x)
    "vllm.unified_attention_with_output": [(3, 0)],  # (q, k, v, output, layer_name, ...)
    # (topk_weights, topk_ids, token_expert_indices, gating_output, renormalize) -> ids port 0,
    # weights port 1 (port 1 is recorded under its own node, see PORT_SPLIT)
    "_moe_C.topk_softmax": [(1, 0), (0, 1)],
    "_moe_C.moe_sum": [(1, 0)],  # (input [M, topk, K], output [M, K])
    "_C.vocab_parallel_embedding": [(0, 0)],  # (out, input_ids, weight, ...) -> this rank's partial embedding rows
    # [QZ, additive] (result!, input, scale!, scale_ub?) -> port 0 = x_q [M, K] e4m3, port 1 = s_x [M, 1] f32
    "_C.dynamic_per_token_scaled_fp8_quant": [(0, 0), (2, 1)],
    # [QZ, additive] (out!, a, b, a_scales, b_scales, bias?) -> port 0 = bf16 [M, N]
    "_C.cutlass_scaled_mm": [(0, 0)],
}
# argument indices that are pure output buffers of the op (never inputs)
OUTPUT_ONLY_ARGS = {
    "_moe_C.topk_softmax": {0, 1, 2},
    "_moe_C.moe_sum": {1},
    "_C.dynamic_per_token_scaled_fp8_quant": {0, 2},  # [QZ, additive]
}
# batch-level row masks passed to a kernel: not a value of any token's computation. topk_softmax's
# is_padding (VLLM_MOE_SKIP_PADDING) marks CUDA-graph padding rows; every committed row is a real
# token, so the mask is False there and the row's result does not depend on it.
BATCH_MASK_ARGS = {"_moe_C.topk_softmax": {6: "is_padding: batch padding-row mask, False on every committed (real-token) row"}}
# a port recorded under its own node (own relation): op -> {port: (node suffix, family)}
PORT_SPLIT = {"_moe_C.topk_softmax": {1: ("topk_weights", "router")}}
SPLIT_RELATION = {("_moe_C.topk_softmax", 1): "moe.topk.weights.same_runtime.v1"}
# opaque custom ops whose body is re-entered with a nested dispatch mode so the ops inside are
# recorded (vLLM wraps the whole fused-MoE forward in one custom op for torch.compile's benefit;
# in eager mode the body runs the same kernels the mode would otherwise see)
MOE_OPAQUE_OPS = {"vllm.moe_forward", "vllm.moe_forward_shared"}
# batch-level bookkeeping inside the MoE block (index buffers for the Triton kernels, no values)
MOE_BOOKKEEPING_OPS = {
    "_moe_C.moe_align_block_size", "_moe_C.sgl_moe_align_block_size", "_moe_C.batched_moe_align_block_size",
    "aten.cumsum", "aten.bincount",
}
ARG_NAMES = {
    "verity.bi_mm": ["x", "w"],  # w arrives as weight.t(); the checker transposes the opened weight
    "aten.mm": ["x", "w"],
    "aten.addmm": ["bias", "x", "w"],
    "aten.matmul": ["x", "w"],
    "aten.linear": ["x", "w", "bias"],
    "aten.embedding": ["table", "token"],
    "verity.bi_rms_norm": ["x", "w", "eps"],
    "_C.rms_norm": ["out", "x", "w", "eps"],
    "_C.fused_add_rms_norm": ["x", "residual", "w", "eps"],
    "_C.rotary_embedding": ["positions", "q", "k", "head_size", "cos_sin_cache", "is_neox"],
    "_C.silu_and_mul": ["out", "x"],
    "vllm.unified_attention_with_output": ["q", "k", "v", "out", "layer_name"],
    "aten.add": ["a", "b"],
    "aten.index": ["x", "idx"],
    "aten.index_select": ["x", "dim", "idx"],
    "_moe_C.topk_softmax": ["weights_out", "ids_out", "token_expert_indices", "logits", "renormalize", "e_score_correction_bias", "is_padding"],
    "_moe_C.moe_sum": ["slots", "out"],
    "aten.sigmoid": ["x"],
    "aten.mul": ["a", "b"],
    "vllm.all_reduce": ["x", "group_name"],
    "vllm.all_gather": ["x", "dim", "world_size", "group_name"],
    "_C.vocab_parallel_embedding": ["out", "token", "table", "org_vocab_start_index", "org_vocab_end_index", "num_org_vocab_padding",
                                    "added_vocab_start_index", "added_vocab_end_index"],
    # [QZ, additive] FP8 linear: x = the bf16 activation row (a node output), scale_ub = None on this profile
    "_C.dynamic_per_token_scaled_fp8_quant": ["out", "x", "scale_out", "scale_ub"],
    # [QZ, additive] x_q / x_scale = the quant node's ports 0 / 1; w = the online-quantised e4m3 weight parameter
    # (`layer.weight`, stored [K, N] as the transpose of the quantised [N, K] tensor), w_scale = `layer.weight_scale`
    # (one f32 word per fused layer), bias = the bf16 bias parameter or None
    "_C.cutlass_scaled_mm": ["out", "x_q", "w", "x_scale", "w_scale", "bias"],
}
# how the same runtime op will be re-executed by the checker (Table 2 runtime_impl)
RUNTIME_IMPL = {
    "verity.bi_mm": "triton:vllm.model_executor.determinism.batch_invariant.matmul_persistent (via linear_batch_invariant)",
    "verity.bi_rms_norm": "triton:vllm.model_executor.determinism.batch_invariant._rms_norm_kernel",
    "_C.fused_add_rms_norm": "cuda:torch.ops._C.fused_add_rms_norm",
    "_C.rms_norm": "cuda:torch.ops._C.rms_norm",
    "_C.rotary_embedding": "cuda:torch.ops._C.rotary_embedding",
    "_C.silu_and_mul": "cuda:torch.ops._C.silu_and_mul",
    "aten.embedding": "aten:embedding (CUDA)",
    "aten.mm": "aten:mm (batch-invariant aten IMPL override if installed)",
    "aten.addmm": "aten:addmm",
    "aten.linear": "aten:linear (batch-invariant aten IMPL override)",
    "aten.matmul": "aten:matmul",
    "aten.add": "aten:add",
    "aten.index": "aten:index",
    "aten.index_select": "aten:index_select",
    "_moe_C.topk_softmax": "cuda:torch.ops._moe_C.topk_softmax (fused softmax + top-k per row)",
    "_moe_C.moe_sum": "cuda:torch.ops._moe_C.moe_sum (bf16 sequential sum over the top-k slots)",
    "aten.sigmoid": "aten:sigmoid (CUDA, bf16)",
    "aten.mul": "aten:mul (CUDA, bf16)",
    "triton.fused_moe": "triton:vllm.model_executor.layers.fused_moe.fused_moe.invoke_fused_moe_triton_kernel (w13 GEMM -> _C.silu_and_mul -> w2 GEMM * routed weight), per (token, slot) row of intermediate_cache3",
    "vllm.all_reduce": "vllm.distributed.parallel_state.all_reduce -> CudaCommunicator.all_reduce (custom_all_reduce kernel for small messages, else NCCL; VLLM_BATCH_INVARIANT=1 pins the NCCL algorithm)",
    "vllm.all_gather": "vllm.distributed.parallel_state.all_gather -> CudaCommunicator.all_gather (NCCL all-gather, a copy: no arithmetic)",
    "_C.vocab_parallel_embedding": "cuda:torch.ops._C.vocab_parallel_embedding (rank-owned rows gathered, zero rows for tokens of the other rank's vocab range)",
    # [QZ, additive]
    "_C.dynamic_per_token_scaled_fp8_quant": "cuda:torch.ops._C.dynamic_per_token_scaled_fp8_quant (dynamic_per_token_scaled_fp8_quant_kernel_strided: per-row absmax, s = max(absmax/448, 1/(448*512)), x_q = cvt.rn.satfinite.e4m3x2.f32(clamp(x/s, ±448)))",
    "_C.cutlass_scaled_mm": "cuda:torch.ops._C.cutlass_scaled_mm (CUTLASS 3.x sm90 FP8 batch-invariant dispatch: QGMMA.64x64x32.F32.E4M3.E4M3 k32 steps, f32 accumulator, epilogue bf16(fma(acc * s_x, s_w, bias)))",
}
# name of the Python-launched fused-MoE Triton kernel pair as a runtime op
MOE_KERNEL_OP = "triton.fused_moe"
# hook the Triton wrapper reports to: (A, B, C, topk_weights, sorted_token_ids, expert_ids,
# num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type) after the launch
_MOE_HOOK: list[Any] = []
_MOE_OPAQUE_IMPL: dict[str, Any] = {}


def moe_opaque_impl(name: str) -> Any:
    """The Python function vLLM registered as the CUDA implementation of an opaque MoE custom op
    (``direct_register_custom_op(op_name=..., op_func=...)`` in fused_moe/runner/moe_runner.py)."""
    if not _MOE_OPAQUE_IMPL:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
        _MOE_OPAQUE_IMPL["vllm.moe_forward"] = mr._moe_forward
        _MOE_OPAQUE_IMPL["vllm.moe_forward_shared"] = mr._moe_forward_shared
    return _MOE_OPAQUE_IMPL[name]


def op_name(func: Any) -> str:
    """'aten.mm', '_C.rotary_embedding', 'vllm.unified_attention_with_output', 'verity.bi_mm'."""
    ns = func.namespace
    name = func._schema.name.split("::")[-1]
    return f"{ns}.{name}"


# ---------------------------------------------------------------------------------------
# dispatcher-visible wrappers for the two Python-launched Triton kernels
# ---------------------------------------------------------------------------------------

_ORIGINALS: dict[str, Any] = {}


def install_kernel_wrappers() -> list[str]:
    """Register torch.ops.verity.* and rebind the two vLLM module attributes. Idempotent."""
    if _ORIGINALS:
        return list(_ORIGINALS)
    import vllm.model_executor.determinism.batch_invariant as bi
    import vllm.model_executor.layers.layernorm as ln

    orig_mm = bi.mm_batch_invariant
    orig_rms = ln.rms_norm_batch_invariant

    @torch.library.custom_op("verity::bi_mm", mutates_args=())
    def bi_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return orig_mm(a, b)

    @bi_mm.register_fake
    def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.new_empty((a.shape[0], b.shape[1]))

    @torch.library.custom_op("verity::bi_rms_norm", mutates_args=())
    def bi_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
        return orig_rms(x, weight, eps, None)

    @bi_rms_norm.register_fake
    def _(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
        return torch.empty_like(x)

    def mm_wrapper(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if bias is not None:
            return orig_mm(a, b, bias)
        return torch.ops.verity.bi_mm(a, b)

    def rms_wrapper(
        input: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6, residual: torch.Tensor | None = None
    ):
        if residual is not None:
            return orig_rms(input, weight, eps, residual)
        return torch.ops.verity.bi_rms_norm(input, weight, float(eps))

    bi.mm_batch_invariant = mm_wrapper
    ln.rms_norm_batch_invariant = rms_wrapper
    _ORIGINALS["batch_invariant.mm_batch_invariant"] = orig_mm
    _ORIGINALS["layernorm.rms_norm_batch_invariant"] = orig_rms

    # fused-MoE Triton kernel pair (W9): Python-launched, hence dispatcher-invisible. The wrapper
    # runs the original launch unchanged and then reports its arguments to the capture; the
    # per-(token, slot) expert outputs are read from intermediate_cache3 (the kernel's own output
    # buffer), so the computation itself is untouched.
    try:
        import vllm.model_executor.layers.fused_moe.experts.triton_moe as tm

        orig_invoke = tm.invoke_fused_moe_triton_kernel

        def invoke_wrapper(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                           mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                           use_int4_w4a16, per_channel_quant, block_shape=None, B_bias=None):
            out = orig_invoke(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                              mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                              use_int4_w4a16, per_channel_quant, block_shape, B_bias)
            for hook in _MOE_HOOK:
                hook(A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, bool(mul_routed_weight),
                     int(top_k), dict(config), str(compute_type),
                     {"use_fp8_w8a8": use_fp8_w8a8, "use_int8_w8a8": use_int8_w8a8, "use_int8_w8a16": use_int8_w8a16,
                      "use_int4_w4a16": use_int4_w4a16, "per_channel_quant": per_channel_quant, "block_shape": block_shape,
                      "B_bias": B_bias is not None, "A_scale": A_scale is not None, "B_scale": B_scale is not None})
            return out

        tm.invoke_fused_moe_triton_kernel = invoke_wrapper
        _ORIGINALS["triton_moe.invoke_fused_moe_triton_kernel"] = orig_invoke
    except Exception as e:  # noqa: BLE001  (dense models: module may be absent from the pinned build)
        _ORIGINALS["triton_moe.invoke_fused_moe_triton_kernel"] = f"not wrapped: {e!r}"
    return list(_ORIGINALS)


def originals() -> dict[str, Any]:
    return dict(_ORIGINALS)


# ---------------------------------------------------------------------------------------
# tensor identity registry
# ---------------------------------------------------------------------------------------


@dataclass
class Record:
    """One recorded op output: the bytes at [start, end) of a storage, and its identity."""

    node: str  # op name, e.g. layers.3.self_attn.qkv_proj.gemm
    port: int
    seq: int  # global record index within the step (time order)
    storage: int
    start: int
    end: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    host: torch.Tensor | None  # pinned snapshot (contiguous), None if not snapshotted
    row_kind: str  # "token" (rows = batch token index), "logits" (rows = logits index), "none",
    # "moe_slot" (rows = batch token index; row i holds top_k slot rows, mapped to e= by the ids record)
    op: str
    family: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArgRef:
    source: str  # node | weight | buffer | token | positions | unbound | const
    node: str | None = None
    port: int | None = None
    cols: tuple[int, int] | None = None  # column slice of the producer's row, if a sub-view
    param: str | None = None
    detail: str | None = None

    def key(self) -> tuple:
        return (self.source, self.node, self.port, self.cols, self.param)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source}
        for k in ("node", "port", "param", "detail"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.cols is not None:
            d["cols"] = list(self.cols)
        return d


class Registry:
    def __init__(self) -> None:
        self.by_storage: dict[int, list[Record]] = {}
        self.records: list[Record] = []

    def reset(self) -> None:
        self.by_storage.clear()
        self.records.clear()

    def add(self, rec: Record) -> None:
        self.records.append(rec)
        self.by_storage.setdefault(rec.storage, []).append(rec)

    def lookup(self, t: torch.Tensor) -> tuple[Record | None, tuple[int, int] | None]:
        """Newest record covering t's bytes; with the column slice if t is a row sub-view."""
        try:
            storage = t.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            return None, None
        recs = self.by_storage.get(storage)
        if not recs:
            return None, None
        start = t.data_ptr()
        esz = t.element_size()
        fallback: Record | None = None
        for rec in reversed(recs):
            if rec.dtype != t.dtype or not (rec.start <= start < rec.end):
                continue
            # same bytes, any shape (view/reshape/transpose of the whole record)
            if start == rec.start and t.numel() == math.prod(rec.shape):
                return rec, None
            # row sub-view: leading dim equal to the record's, same row stride, contiguous inner
            # extent that is a column range of the record's rows (qkv split -> q | k | v)
            # RC: a one-row view carries no information in stride(0) (torch's view() recomputes it from
            # the inner extent, e.g. 256 for v=[1,2,128] of a [1,2048] qkv row), so the leading stride is
            # only compared when there is more than one row.  Previously a single-token step left every
            # attention v argument "alias-unresolved" (an arg_violation) on Qwen2's bias-add output.
            if (
                len(rec.shape) >= 2 and t.dim() >= 2 and t.shape[0] == rec.shape[0]
                and (t.shape[0] == 1 or t.stride(0) == rec.stride[0])
            ):
                exp = 1
                ok = True
                for d in range(t.dim() - 1, 0, -1):
                    if t.shape[d] != 1 and t.stride(d) != exp:
                        ok = False
                        break
                    exp *= t.shape[d]
                if ok:
                    inner = math.prod(t.shape[1:])
                    c0 = (start - rec.start) // esz
                    row_len = math.prod(rec.shape[1:])
                    if 0 <= c0 and c0 + inner <= row_len:
                        if c0 == 0 and inner == row_len:
                            return rec, None
                        return rec, (int(c0), int(c0 + inner))
            if fallback is None:
                fallback = rec
        if fallback is not None:
            return fallback, ("unknown", "unknown")  # type: ignore[return-value]
        return None, None


# ---------------------------------------------------------------------------------------
# the dispatch mode
# ---------------------------------------------------------------------------------------


class PinnedArena:
    """Bump allocator over one pinned host buffer; reset per step."""

    def __init__(self, nbytes: int) -> None:
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off = 0
        self.overflow_allocs = 0

    def reset(self) -> None:
        self.off = 0

    def snapshot(self, t: torch.Tensor) -> torch.Tensor:
        """Enqueue a D2H copy of ``t``'s current bytes on the current stream; returns the pinned
        host view. Later kernels on the stream (in-place mutation, allocator reuse) are ordered
        behind the copy; the caller synchronises once per step before reading."""
        src = t if t.is_contiguous() else t.contiguous()
        host = self.take(src)
        host.copy_(src, non_blocking=True)
        return host

    def take(self, t: torch.Tensor) -> torch.Tensor:
        nbytes = t.numel() * t.element_size()
        aligned = (self.off + 255) // 256 * 256
        if aligned + nbytes > self.buf.numel():
            self.overflow_allocs += 1
            return torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        view = self.buf[aligned : aligned + nbytes].view(t.dtype).view(t.shape)
        self.off = aligned + nbytes
        return view


class Interceptor(TorchDispatchMode):
    def __init__(self, capture: "Capture") -> None:
        super().__init__()
        self.cap = capture
        self.enabled = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.enabled and op_name(func) in MOE_OPAQUE_OPS:
            # the mode is popped while this handler runs, so the custom op's body would be
            # invisible: run the op's registered CUDA implementation (the same Python function the
            # dispatcher calls; vllm.utils.torch_utils.direct_register_custom_op) under a fresh
            # mode bound to the same capture. Calling the OpOverload again under the fresh mode
            # would re-enter this handler forever. Outputs are aliases of values recorded inside
            # (moe_sum output, shared-expert gate product).
            impl = moe_opaque_impl(op_name(func))
            inner = Interceptor(self.cap)
            inner.enabled = True
            self.cap.moe_depth += 1
            self.cap.op_histogram[op_name(func)] = self.cap.op_histogram.get(op_name(func), 0) + 1
            try:
                with inner:
                    return impl(*args, **kwargs)
            finally:
                self.cap.moe_depth -= 1
        out = func(*args, **kwargs)
        if not self.enabled:
            return out
        try:
            self.cap.on_op(func, args, kwargs, out)
        except Exception as e:  # noqa: BLE001
            self.cap.errors.append({"op": str(func), "error": repr(e)})
            if len(self.cap.errors) > 50:
                raise
        return out


@dataclass
class NodeTemplate:
    op: str
    layer: str
    family: str
    op_type: str
    runtime_op: str
    runtime_impl: str
    ports: dict[int, dict[str, Any]] = field(default_factory=dict)  # port -> {dtype, row_shape}
    args: list[dict[str, Any]] = field(default_factory=list)
    consts: dict[str, Any] = field(default_factory=dict)
    instances: int = 0
    arg_violations: list[dict[str, Any]] = field(default_factory=list)
    row_kind: str = "token"
    phases: set[str] = field(default_factory=set)
    module_path: str = ""
    relation: str | None = None  # explicit relation id (compiled kernels, MoE, cast); None -> RELATION_FOR_RUNTIME_OP
    experts: int | None = None  # slotted node (MoE): outputs carry e=<slot>, population = slots
    # extra description fields merged into the node (compiled: kernel identity, launch spec; MoE: slot origins)
    extra: dict[str, Any] = field(default_factory=dict)


class Capture:
    """Per-process capture state; one instance per replay run."""

    def __init__(
        self,
        *,
        runner: Any,
        model: torch.nn.Module,
        profile_id: str,
        run_dir: Path,
        writer: RunWriter | None,
        hash_only: bool,
        prompt_lens: dict[str, int],
        arena_bytes: int = 8 << 30,
        log: Any = print,
        tokens: dict[str, list[int]] | None = None,
        cross_request_kv: bool = False,
        shard: int | None = None,
    ) -> None:
        self.runner = runner
        self.model = model
        self.profile_id = profile_id
        # tensor-parallel rank of this process (lane TP): every value and weight this process
        # materialises is identified with ``/s=<rank>`` (capture-format canonical id field ``s``);
        # None for the unsharded runs, whose ids are unchanged
        self.shard = shard
        self.run_dir = Path(run_dir)
        self.writer = writer
        self.hash_only = hash_only
        self.prompt_lens = prompt_lens
        self.log = log
        # state rows (W10): full token sequence per request (prompt + recorded outputs) for the
        # cross-request KV identity rule, and whether that rule is in force (R3 cached replay)
        self.tokens = tokens or {}
        self.cross_request_kv = cross_request_kv
        self.cached_reads: dict[str, dict[int, str]] = {}  # request -> {position: source request}
        self.kv_cross_request_resolved = 0
        self.kv_unresolved_under_own_request_rule = 0
        self._cpl_cache: dict[tuple[str, str], int] = {}
        self.req_emitted_positions: dict[str, set[int]] = {}
        self.recompute_steps: list[dict[str, Any]] = []
        self.req_sampled: dict[str, int] = {}  # sampled positions emitted so far per request
        self.cur_sampled_positions: set[tuple[str, int]] = set()  # (request, t) sampled in the current step
        self.registry = Registry()
        self.arena = PinnedArena(arena_bytes)
        self.mode = Interceptor(self)
        self.errors: list[dict[str, Any]] = []
        self.module_stack: list[str] = []
        self.op_counter: dict[tuple[str, str], int] = {}
        self.templates: dict[str, NodeTemplate] = {}
        self.params: dict[int, tuple[str, torch.Tensor]] = {}
        self.param_ranges: list[tuple[int, int, str]] = []
        self.step_index = 0
        self.req_steps: dict[str, int] = {}
        self.req_records: dict[str, list[StepRecord]] = {}
        self.req_outputs: dict[str, list[int]] = {}
        self.owner: dict[int, tuple[str, int]] = {}  # kv slot -> (request, position)
        self.kv_audit: list[dict[str, Any]] = []
        self.unbound: list[dict[str, Any]] = []
        self.module_audit: list[dict[str, Any]] = []
        self.op_histogram: dict[str, int] = {}
        self.leaf_count = 0
        self.ru_count = 0
        self.bytes_captured = 0
        self.timing = {"snapshot_s": 0.0, "emit_s": 0.0, "forward_s": 0.0, "sync_s": 0.0}
        self.phase = "forward"
        self.hash_index: dict[tuple[str, int], dict[str, Any]] = {}
        self.non_token_outputs: dict[str, int] = {}
        self.input_alias: dict[int, str] = {}
        self.side_effect_ops: dict[str, int] = {}
        self.no_record = False  # rung (a): batch bookkeeping only (positions, sampler agreement), nothing captured
        # MoE (W9): nesting depth inside vllm.moe_forward*, the current layer's router record,
        # the w13 launch of the current block (its A row = the block input), witness counters
        self.moe_depth = 0
        self.cur_topk: dict[str, Record] = {}  # module -> ids record (port 0 of topk)
        self.cur_moe_w13: dict[str, dict[str, Any]] = {}  # module -> first-kernel facts
        self.moe_expert_params: set[str] = set()  # fused expert params to store per expert
        self.moe_routes: dict[str, list[int]] = {}  # layer -> count of (token, slot) per expert
        self.moe_active_slots = 0
        self.moe_inactive_slots = 0
        self.moe_configs: dict[str, dict[str, Any]] = {}  # "M=<n>" -> Triton config observed
        self.moe_steps_num_tokens: dict[str, int] = {}  # "<M>" -> committed positions whose step had M token rows
        self.moe_skipped_ops: dict[str, int] = {}
        self._install_params()
        self._install_hooks()
        self._wrap_runner()
        _MOE_HOOK.clear()
        _MOE_HOOK.append(self.on_moe_kernel)

    # ---------------- installation ----------------
    def _install_params(self) -> None:
        for name, p in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if not p.is_cuda:
                continue
            ptr = p.data_ptr()
            name = normalize_module(name)
            if ptr not in self.params:
                self.params[ptr] = (name, p)
                self.param_ranges.append((ptr, ptr + p.numel() * p.element_size(), name))
        self.param_ranges.sort()

    def _install_hooks(self) -> None:
        def pre(mod, args, _name=None):
            name = mod._verity_name
            # a module instance shared across layers (vLLM caches RotaryEmbedding via get_rope) is
            # registered under the first layer's path; name it after the layer that is calling it
            if self.module_stack and name.startswith("layers."):
                parent = self.module_stack[-1]
                m_par = _LAYER_PREFIX.match(parent)
                m_own = _LAYER_PREFIX.match(name)
                if m_par and m_own and m_par.group(0) != m_own.group(0):
                    name = m_par.group(0) + name[len(m_own.group(0)):]
            self.module_stack.append(name)

        def post(mod, args, output):
            name = self.module_stack.pop() if self.module_stack else None
            if not self.mode.enabled:
                return
            # audit: every tensor output of a module must be a recorded value (or alias of one)
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for o in outs:
                if isinstance(o, torch.Tensor) and o.is_cuda and o.numel() > 0:
                    rec, _ = self.registry.lookup(o)
                    if rec is None and self._param_of(o) is None:
                        self.module_audit.append({"module": name, "shape": list(o.shape), "dtype": str(o.dtype), "step": self.step_index})

        for name, mod in self.model.named_modules():
            if name == "":
                continue
            mod._verity_name = normalize_module(name)
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)

    def _wrap_runner(self) -> None:
        runner = self.runner
        cap = self
        orig_execute = runner.execute_model
        orig_sample = runner.sample

        def execute_model(scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", False) or (len(a) >= 2 and a[1])
            if dummy or scheduler_output.total_num_scheduled_tokens == 0:
                return orig_execute(scheduler_output, *a, **kw)
            cap.begin_step(scheduler_output)
            t0 = time.perf_counter()
            if cap.no_record:
                out = orig_execute(scheduler_output, *a, **kw)
            else:
                with cap.mode:
                    cap.mode.enabled = True
                    try:
                        out = orig_execute(scheduler_output, *a, **kw)
                    finally:
                        cap.mode.enabled = False
            cap.timing["forward_s"] += time.perf_counter() - t0
            cap.after_forward()
            return out

        def sample(hidden_states, input_batch, grammar_output):
            cap.phase = "sample"
            if cap.no_record:
                res = orig_sample(hidden_states, input_batch, grammar_output)
            else:
                with cap.mode:
                    cap.mode.enabled = True
                    try:
                        res = orig_sample(hidden_states, input_batch, grammar_output)
                    finally:
                        cap.mode.enabled = False
            cap.phase = "forward"
            cap.after_sample(res, input_batch)
            return res

        runner.execute_model = execute_model
        runner.sample = sample

    # ---------------- step lifecycle ----------------
    def begin_step(self, scheduler_output) -> None:
        self.registry.reset()
        self.arena.reset()
        self.op_counter.clear()
        self.module_stack.clear()
        self.input_alias = {}
        self.cur_scheduler_output = scheduler_output
        self.cur_batch = None
        self.cur_logits_map = None
        self.cur_sampled = None
        self.cur_logprobs = None
        self.phase = "forward"

    def after_forward(self) -> None:
        st = self.runner.execute_model_state
        ib = st.input_batch
        self.cur_batch = ib
        n = ib.num_reqs
        qsl = np.asarray(ib.query_start_loc_np[: n + 1])
        nct = np.asarray(ib.num_computed_tokens_np[:n])
        self.cur_qsl = qsl
        self.cur_nct = nct
        self.cur_req_ids = [external_id(r) for r in ib.req_ids]
        self.cur_num_tokens = int(ib.num_tokens)
        # positions per token row (absolute)
        rows_req = np.repeat(np.arange(n), np.diff(qsl))
        rows_pos = nct[rows_req] + (np.arange(qsl[-1]) - qsl[rows_req])
        self.cur_rows_req = rows_req
        self.cur_rows_pos = rows_pos
        # logits rows: request i at its last scheduled position
        self.cur_logits_map = [(self.cur_req_ids[i], int(nct[i] + (qsl[i + 1] - qsl[i]) - 1)) for i in range(n)]
        # KV slot ownership from the forward context's slot mapping (one KV group expected)
        self._record_kv_owners(st)

    def _record_kv_owners(self, st) -> None:
        sm = st.slot_mappings_by_layer
        if not sm:
            return
        first = next(iter(sm.values()))
        slots = first[: self.cur_num_tokens].cpu().numpy()
        distinct = {id(v) for v in sm.values()}
        self.kv_groups = len(distinct)
        for i in range(self.cur_num_tokens):
            self.owner[int(slots[i])] = (self.cur_req_ids[self.cur_rows_req[i]], int(self.cur_rows_pos[i]))
        # audit the reads: block table walk for every request over positions 0..t_max
        try:
            bt = self.runner.block_tables.gather_block_tables(self.cur_batch.idx_mapping, num_reqs_padded=self.cur_batch.num_reqs_after_padding)
            bt0 = bt[0][: len(self.cur_req_ids)].cpu().numpy()
            bs = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
        except Exception as e:  # noqa: BLE001
            self.kv_audit.append({"step": self.step_index, "error": repr(e)})
            return
        for i, rid in enumerate(self.cur_req_ids):
            t_max = int(self.cur_nct[i] + (self.cur_qsl[i + 1] - self.cur_qsl[i]) - 1)
            bad = []
            nbad = 0
            cross = 0
            own_rule_bad = 0
            for t in range(t_max + 1):
                slot = int(bt0[i][t // bs]) * bs + t % bs
                own = self.owner.get(slot)
                if own == (rid, t):
                    continue
                own_rule_bad += 1
                # R3 identity rule: a slot written by ANOTHER request at the same position is a
                # valid read iff the two token sequences agree on 0..t (the K/V at t is a function
                # of tokens 0..t under the causal model); the read binds to the source's canonical
                # id.  Recorded per (request, position); never silently accepted otherwise.
                if self.cross_request_kv and own is not None and own[1] == t and own[0] != rid and t < self._common_prefix_len(own[0], rid):
                    cross += 1
                    self.cached_reads.setdefault(rid, {}).setdefault(t, own[0])
                    continue
                nbad += 1
                if len(bad) <= 8:
                    bad.append({"t": t, "slot": slot, "owner": own})
            self.kv_cross_request_resolved += cross
            self.kv_unresolved_under_own_request_rule += own_rule_bad
            self.kv_audit.append(
                {"step": self.step_index, "request_id": rid, "positions_read": t_max + 1, "unresolved": nbad, "examples": bad, "block_size": bs,
                 "cross_request_resolved": cross}
            )

    def _common_prefix_len(self, a: str, b: str) -> int:
        key = (a, b)
        n = self._cpl_cache.get(key)
        if n is None:
            ta, tb = self.tokens.get(a, []), self.tokens.get(b, [])
            n = 0
            for x, y in zip(ta, tb):
                if x != y:
                    break
                n += 1
            self._cpl_cache[key] = n
        return n

    def after_sample(self, res, input_batch) -> None:
        sampler_output = res[0] if isinstance(res, tuple) else res
        try:
            sampled = sampler_output.sampled_token_ids
            self.cur_sampled = sampled.detach().to("cpu").numpy().reshape(len(self.cur_req_ids), -1)[:, 0]
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_sample", "error": repr(e)})
            self.cur_sampled = None
        if not self.no_record:
            self.emit_step()
        self.step_index += 1

    # ---------------- op recording ----------------
    def _param_of(self, t: torch.Tensor) -> str | None:
        ptr = t.data_ptr()
        hit = self.params.get(ptr)
        if hit is not None:
            return hit[0]
        # binary search ranges
        lo, hi = 0, len(self.param_ranges)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.param_ranges[mid][0] <= ptr:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            s, e, name = self.param_ranges[lo - 1]
            if s <= ptr < e:
                return name
        return None

    def _module_name(self) -> str:
        return self.module_stack[-1] if self.module_stack else ("sample" if self.phase == "sample" else "top")

    def _classify_arg(self, t: torch.Tensor) -> ArgRef:
        if t.numel() == 0:
            return ArgRef("const", detail=f"empty_tensor:{str(t.dtype).replace('torch.', '')}")
        rec, cols = self.registry.lookup(t)
        if rec is not None:
            if cols == ("unknown", "unknown"):
                return ArgRef("node", rec.node, rec.port, None, detail="alias-unresolved")
            return ArgRef("node", rec.node, rec.port, cols)
        p = self._param_of(t)
        if p is not None:
            return ArgRef("weight", param=p)
        which = self._input_kind(t)
        if which == "input_ids":
            return ArgRef("token", detail="input_ids")
        if which == "positions":
            return ArgRef("positions")
        if self.cur_batch is not None and t.data_ptr() == self.cur_batch.logits_indices.data_ptr():
            return ArgRef("const", detail="logits_indices")
        return ArgRef("unbound", detail=f"shape={list(t.shape)} dtype={t.dtype}")

    def _input_kind(self, t: torch.Tensor) -> str | None:
        ib = self.runner.input_buffers
        ptr = t.data_ptr()
        if ptr == ib.input_ids.data_ptr():
            return "input_ids"
        if ptr == ib.positions.data_ptr():
            return "positions"
        return self.input_alias.get(ptr)

    def on_op(self, func, args, kwargs, out) -> None:
        name = op_name(func)
        self.op_histogram[name] = self.op_histogram.get(name, 0) + 1
        if name in ALLOC_OPS or name in IGNORE_OPS:
            return
        if name in ALIAS_OPS:
            return  # identity resolved lazily by address in Registry.lookup
        if name in COPY_OPS:
            self._on_copy(name, args, out)
            return
        if name in SIDE_EFFECT_OPS:
            # KV-cache write: no value output (K/V identities are the rope/qkv leaves; the
            # slot each position lands in is taken from the forward context, see _record_kv_owners)
            self.side_effect_ops[name] = self.side_effect_ops.get(name, 0) + 1
            return
        if self.phase == "sample":
            if name in ("aten.index", "aten.index_select") and not self.module_stack:
                self._on_gather_alias(name, args, out)
                return
            if not self.module_stack:
                # the sampler's own kernels (temperature, argmax, logprobs) are outside the
                # committed graph: W3's sampler.select relation recomputes from the logits
                self.op_histogram["sample-phase-skipped:" + name] = self.op_histogram.get("sample-phase-skipped:" + name, 0) + 1
                return
        module = self._module_name()
        if self.moe_depth > 0:
            # inside the fused-MoE block: index bookkeeping and the routed experts' own
            # activation ([M*top_k, N] rows of intermediate_cache1, an internal of the per-slot
            # expert unit that the kernel wrapper records as a whole) are not values of the graph
            if name in MOE_BOOKKEEPING_OPS or (name == "_C.silu_and_mul" and module.endswith(".experts")):
                self.moe_skipped_ops[name] = self.moe_skipped_ops.get(name, 0) + 1
                return
        family, op_type = OP_FAMILY.get(name, ("other", "other"))
        if name in MUTATING_OUTPUTS:
            outs = [(port, args[i]) for i, port in MUTATING_OUTPUTS[name]]
        else:
            if isinstance(out, torch.Tensor):
                outs = [(0, out)]
            elif isinstance(out, (tuple, list)):
                outs = [(i, o) for i, o in enumerate(out) if isinstance(o, torch.Tensor)]
            else:
                return
        if not outs or not all(o.is_cuda for _, o in outs):
            return
        if name == "vllm.unified_attention_with_output":
            # the op carries its own module path (``model.layers.3.self_attn.attn``); use it. In the
            # compiled replay no submodule hook fires (the pieces run as Inductor code), so the module
            # stack would otherwise say ``model``.
            ln = next((a for a in list(args) + list((kwargs or {}).values()) if isinstance(a, str) and "attn" in a), None)
            if ln:
                module = normalize_module(ln)
            elif module in ("top", "model"):
                # Inductor's extern call passes the layer name in a form we did not see: fall back to
                # the layer of the query's producer (the rope / qkv node of the same layer)
                q = args[0] if args and isinstance(args[0], torch.Tensor) else None
                rec = self.registry.lookup(q)[0] if q is not None else None
                lay = layer_of(rec.node) if rec is not None else None
                if lay not in (None, "embed", "norm", "lm_head", "top"):
                    module = f"layers.{lay}.self_attn.attn"
        node = node_name(module, family, self.op_counter, name)
        # argument identities (before adding outputs, so in-place ops see their inputs)
        names = ARG_NAMES.get(name, [])
        arg_refs: list[dict[str, Any]] = []
        consts: dict[str, Any] = {}
        output_only = OUTPUT_ONLY_ARGS.get(name, set())
        batch_masks = BATCH_MASK_ARGS.get(name, {})
        for i, a in enumerate(args):
            an = names[i] if i < len(names) else f"arg{i}"
            if isinstance(a, torch.Tensor):
                if name in MUTATING_OUTPUTS and any(i == ai for ai, _ in MUTATING_OUTPUTS[name]) and an == "out":
                    continue  # pure output buffer
                if i in output_only:
                    continue
                if i in batch_masks:
                    consts[an] = batch_masks[i]
                    continue
                ref = self._classify_arg(a)
                if ref.source == "unbound":
                    self.unbound.append({"node": node, "arg": an, "detail": ref.detail, "step": self.step_index})
                d = ref.to_json()
                d["name"] = an
                d["shape"] = list(a.shape)
                d["dtype"] = str(a.dtype).replace("torch.", "")
                arg_refs.append(d)
            elif isinstance(a, (int, float, bool, str)):
                consts[an] = a
        layer = layer_of(module)
        tpl = self.templates.get(node)
        if tpl is None:
            tpl = NodeTemplate(
                op=node, layer=layer, family=family, op_type=op_type, runtime_op=name,
                runtime_impl=RUNTIME_IMPL.get(name, "unresolved"), args=arg_refs, consts=consts, module_path=module,
            )
            if name == "vllm.unified_attention_with_output":
                tpl.runtime_impl = self._attn_impl(module)
                tpl.consts.update(self._attn_consts(module))
            if name == "verity.bi_mm" and self.phase == "sample":
                tpl.op_type = "lm_head"
            self.templates[node] = tpl
        else:
            same = [self._arg_key(a) for a in arg_refs] == [self._arg_key(a) for a in tpl.args]
            if not same and len(tpl.arg_violations) < 20:
                tpl.arg_violations.append({"step": self.step_index, "got": arg_refs})
        tpl.instances += 1
        # a port recorded under its own node (its own relation), e.g. the router's weights
        split = PORT_SPLIT.get(name, {})
        split_tpls: dict[int, NodeTemplate] = {}
        for port, (suffix, sfamily) in split.items():
            snode = f"{module}.{suffix}" if module not in ("top", "sample") else suffix
            stpl = self.templates.get(snode)
            if stpl is None:
                stpl = NodeTemplate(op=snode, layer=layer, family=sfamily, op_type=op_type, runtime_op=name,
                                    runtime_impl=RUNTIME_IMPL.get(name, "unresolved"), args=list(arg_refs), consts=dict(consts),
                                    module_path=module, relation=SPLIT_RELATION.get((name, port)))
                self.templates[snode] = stpl
            stpl.instances += 1
            split_tpls[port] = stpl
        if name == "_moe_C.topk_softmax":
            tpl.relation = "moe.topk.ids.v1"
        # record outputs with pinned snapshots (row kind is decided at emit time, when the
        # batch's padded token count is known)
        t0 = time.perf_counter()
        for port, o in outs:
            if o.dim() == 0 or o.numel() == 0:
                continue
            row_kind = "logits" if self.phase == "sample" else "token"
            host = self.arena.snapshot(o)
            rnode, rport, rtpl = node, port, tpl
            if port in split_tpls:
                rnode, rport, rtpl = split_tpls[port].op, 0, split_tpls[port]
            rec = Record(
                node=rnode, port=rport, seq=len(self.registry.records), storage=o.untyped_storage().data_ptr(),
                start=o.data_ptr(), end=_extent(o), shape=tuple(o.shape), stride=tuple(o.stride()), dtype=o.dtype,
                host=host, row_kind=row_kind, op=name, family=family,
            )
            self.registry.add(rec)
            rtpl.ports.setdefault(rport, {"dtype": str(o.dtype).replace("torch.", ""), "row_shape": list(o.shape[1:]), "row_kind": row_kind})
            if name == "_moe_C.topk_softmax" and port == 0:
                self.cur_topk[module] = rec
        self.timing["snapshot_s"] += time.perf_counter() - t0

    # ---------------- MoE: the fused expert kernel pair ----------------
    def on_moe_kernel(self, A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                      mul_routed_weight: bool, top_k: int, config: dict[str, Any], compute_type: str, flags: dict[str, Any]) -> None:
        """Called by the Triton wrapper after each fused-MoE GEMM launch. The first launch (w13,
        top_k = k) fixes the block input row identity; the second (w2, mul_routed_weight) leaves the
        per-(token, slot) expert outputs in C = intermediate_cache3 [M, top_k, K], recorded as a
        slotted node whose e= index comes from the router's ids record at emit time."""
        if not self.mode.enabled and self.moe_depth == 0:
            return
        try:
            module = self._module_name()
            wname = self._param_of(B)
            if wname is not None:
                self.moe_expert_params.add(wname)
            M = int(A.shape[0]) if not mul_routed_weight else int(C.shape[0])
            self.moe_configs.setdefault(f"M={M}", {**config, "compute_type": compute_type, **flags})
            if not mul_routed_weight:
                self.cur_moe_w13[module] = {
                    "x": self._classify_arg(A), "x_shape": list(A.shape), "x_dtype": str(A.dtype).replace("torch.", ""),
                    "w13": wname, "w13_shape": list(B.shape), "top_k": int(top_k), "M": M, "config": dict(config),
                }
                return
            facts = self.cur_moe_w13.get(module)
            ids_rec = self.cur_topk.get(module)
            if facts is None or ids_rec is None:
                self.errors.append({"where": "on_moe_kernel", "error": f"w2 launch without w13/topk facts in {module}"})
                return
            E = int(B.shape[0])
            K = int(C.shape[-1])
            node = f"{module}.expert_out"
            layer = layer_of(module)
            # the Triton launch config is a declared kernel-path fact of the node (recorded from the
            # launch: BLOCK_SIZE_M/N/K, GROUP_SIZE_M, SPLIT_K). vLLM selects it from the batch's token
            # count M in general, but under VLLM_BATCH_INVARIANT=1 (this profile) get_moe_configs()
            # returns None and get_default_config() a fixed config, so it is M-independent; a run in
            # which the same node sees two configs is recorded as such and its relation unresolved.
            launch_config = {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v) for k, v in dict(config).items()}
            w13_config = {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v) for k, v in dict(facts["config"]).items()}
            tpl = self.templates.get(node)
            if tpl is None:
                x = facts["x"]
                xd = x.to_json()
                xd.update({"name": "x", "shape": facts["x_shape"], "dtype": facts["x_dtype"]})
                if x.source == "unbound":
                    self.unbound.append({"node": node, "arg": "x", "detail": x.detail, "step": self.step_index})
                args = [
                    xd,
                    {"name": "ids", "source": "node", "node": ids_rec.node, "port": 0, "shape": list(ids_rec.shape), "dtype": "int32"},
                    {"name": "weights", "source": "node", "node": f"{module}.topk_weights", "port": 0, "shape": list(ids_rec.shape), "dtype": "float32"},
                    {"name": "w13", "source": "weight", "param": facts["w13"], "expert": "same", "lazy": True, "shape": facts["w13_shape"][1:], "dtype": str(B.dtype).replace("torch.", "")},
                    {"name": "w2", "source": "weight", "param": wname, "expert": "same", "lazy": True, "shape": list(B.shape[1:]), "dtype": str(B.dtype).replace("torch.", "")},
                ]
                tpl = NodeTemplate(
                    op=node, layer=layer, family="expert", op_type="expert", runtime_op=MOE_KERNEL_OP,
                    runtime_impl=RUNTIME_IMPL[MOE_KERNEL_OP], args=args, module_path=module, relation="moe.expert_slot.same_runtime.v1",
                    experts=E,
                    consts={"top_k": int(facts["top_k"]), "num_experts": E, "w1_shape": facts["w13_shape"], "w2_shape": list(B.shape),
                            "activation": "silu", "compute_type": compute_type, "mul_routed_weight": True,
                            "apply_router_weight_on_input": False, **{k: v for k, v in flags.items() if k != "B_bias"},
                            "triton_config": launch_config,
                            "config_source": "declared: the launch's config (both GEMMs of fused_experts use one config per call; "
                                             "vllm try_get_optimal_moe_config(w1_shape, w2_shape, top_k, None, M) -> under "
                                             "VLLM_BATCH_INVARIANT=1 get_moe_configs() is None and get_default_config() is fixed, M-independent)"},
                    extra={"slot_origins": {"active": "captured", "inactive": "derived"}, "inactive_fill": "zeros"},
                )
                self.templates[node] = tpl
            if tpl.consts.get("triton_config") != launch_config or w13_config != launch_config:
                # two configs for one node (a stock, non-batch-invariant profile would do this): the
                # node's kernel path is not one declared computation; record and unresolve
                seen = tpl.consts.setdefault("triton_config_conflicts", [])
                for cfg in (launch_config, w13_config):
                    if cfg not in seen and cfg != tpl.consts.get("triton_config"):
                        seen.append(cfg)
                tpl.relation = "unresolved"
                tpl.consts["config_source"] = "CONFLICT: the launch config varied across steps of this run (see triton_config_conflicts)"
            tpl.instances += 1
            t0 = time.perf_counter()
            host = self.arena.snapshot(C)
            rec = Record(
                node=node, port=0, seq=len(self.registry.records), storage=C.untyped_storage().data_ptr(),
                start=C.data_ptr(), end=_extent(C), shape=tuple(C.shape), stride=tuple(C.stride()), dtype=C.dtype,
                host=host, row_kind="moe_slot", op=MOE_KERNEL_OP, family="expert", extra={"ids": ids_rec, "experts": E},
            )
            self.registry.add(rec)
            tpl.ports.setdefault(0, {"dtype": str(C.dtype).replace("torch.", ""), "row_shape": [K], "row_kind": "moe_slot"})
            self.timing["snapshot_s"] += time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "on_moe_kernel", "error": repr(e)})

    def _on_gather_alias(self, name: str, args, out) -> None:
        """hidden_states[logits_indices] in the sampler: rows of the final norm output at the
        sampled positions. An alias with a row remap, not a fresh value (no leaf)."""
        src = args[0]
        rec, cols = self.registry.lookup(src)
        if rec is None or not isinstance(out, torch.Tensor):
            # gathers of the input_ids / positions buffers (sampler bookkeeping) are not values
            if not isinstance(src, torch.Tensor) or self._input_kind(src) is None:
                self.unbound.append({"node": "lm_head.gather", "arg": "x", "detail": f"source not recorded shape={list(getattr(src, 'shape', []))}", "step": self.step_index})
            return
        alias = Record(
            node=rec.node, port=rec.port, seq=len(self.registry.records), storage=out.untyped_storage().data_ptr(),
            start=out.data_ptr(), end=_extent(out), shape=tuple(out.shape), stride=tuple(out.stride()), dtype=out.dtype,
            host=None, row_kind="logits", op="alias-gather", family=rec.family,
        )
        self.registry.add(alias)

    def _attn_consts(self, module: str) -> dict[str, Any]:
        """What the attention relation needs to re-run the backend: heads, scale, block size, FA version."""
        out: dict[str, Any] = {}
        try:
            for name, mod in self.model.named_modules():
                if normalize_module(name) == module:
                    impl = mod.impl
                    out = {
                        "num_heads": int(impl.num_heads), "num_kv_heads": int(impl.num_kv_heads), "head_size": int(impl.head_size),
                        "scale": float(impl.scale), "sliding_window": getattr(impl, "sliding_window", None),
                        "logits_soft_cap": getattr(impl, "logits_soft_cap", None), "kv_cache_dtype": str(getattr(impl, "kv_cache_dtype", "auto")),
                        "fa_version": getattr(impl, "vllm_flash_attn_version", None), "num_splits": 1,
                        "block_size": int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size),
                        "backend": getattr(getattr(mod, "attn_backend", None), "__name__", None),
                    }
                    break
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)
        return out

    def _attn_impl(self, module: str) -> str:
        try:
            for name, mod in self.model.named_modules():
                if normalize_module(name) == module:
                    impl = getattr(mod, "impl", None)
                    be = getattr(mod, "attn_backend", None)
                    return f"{impl.__class__.__module__}.{impl.__class__.__name__} backend={getattr(be, '__name__', be)}"
        except Exception:  # noqa: BLE001
            pass
        return "unresolved"

    @staticmethod
    def _arg_key(a: dict[str, Any]) -> tuple:
        return (a.get("name"), a.get("source"), a.get("node"), a.get("port"), tuple(a.get("cols") or ()), a.get("param"))

    def _on_copy(self, name: str, args, out) -> None:
        # copies keep the identity of their source when dtype is unchanged; a dtype change is a cast node
        src = args[1] if name == "aten.copy_" else args[0]
        dst = args[0] if name == "aten.copy_" else out
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor) or not dst.is_cuda:
            return
        rec, cols = self.registry.lookup(src)
        if rec is None:
            # input_ids.long() / positions copies keep their boundary identity
            which = self._input_kind(src)
            if which is not None:
                self.input_alias[dst.data_ptr()] = which
            return
        if dst.dtype != src.dtype:
            self.on_op_cast(src, dst, rec, cols)
            return
        # alias record: same identity, new address
        alias = Record(
            node=rec.node, port=rec.port, seq=len(self.registry.records), storage=dst.untyped_storage().data_ptr(),
            start=dst.data_ptr(), end=dst.data_ptr() + dst.numel() * dst.element_size(), shape=tuple(dst.shape),
            stride=tuple(dst.stride()), dtype=dst.dtype, host=None, row_kind=rec.row_kind, op="alias", family=rec.family,
        )
        if cols is not None and cols != ("unknown", "unknown"):
            alias.node = f"{rec.node}[{cols[0]}:{cols[1]}]"
        self.registry.add(alias)

    def on_op_cast(self, src, dst, rec, cols) -> None:
        module = self._module_name()
        node = node_name(module, "cast", self.op_counter, "aten._to_copy")
        tpl = self.templates.get(node)
        if tpl is None:
            tpl = NodeTemplate(op=node, layer=layer_of(module), family="cast", op_type="other", runtime_op="aten._to_copy", runtime_impl="aten:_to_copy",
                               args=[{"name": "x", "source": "node", "node": rec.node, "port": rec.port, **({"cols": list(cols)} if cols and cols != ("unknown", "unknown") else {})}],
                               consts={"dtype": str(dst.dtype)}, module_path=module,
                               relation=None if (src.dtype in (torch.bfloat16, torch.float16) and dst.dtype == torch.float32) else "unresolved")
            self.templates[node] = tpl
        tpl.instances += 1
        row_kind = self._row_kind(dst)
        host = self.arena.snapshot(dst)
        r = Record(node=node, port=0, seq=len(self.registry.records), storage=dst.untyped_storage().data_ptr(), start=dst.data_ptr(),
                   end=dst.data_ptr() + dst.numel() * dst.element_size(), shape=tuple(dst.shape), stride=tuple(dst.stride()), dtype=dst.dtype,
                   host=host, row_kind=row_kind, op="aten._to_copy", family="cast")
        self.registry.add(r)
        tpl.ports.setdefault(0, {"dtype": str(dst.dtype).replace("torch.", ""), "row_shape": list(dst.shape[1:]) if row_kind != "none" else list(dst.shape), "row_kind": row_kind})

    def _row_kind(self, o: torch.Tensor) -> str:  # kept for the cast path
        return "logits" if self.phase == "sample" else "token"

    # ---------------- emission ----------------
    def emit_step(self) -> None:
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        self.timing["sync_s"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        self.cur_sampled_positions = set()
        n_tokens = self.cur_num_tokens
        n_padded = int(self.cur_batch.num_tokens_after_padding)
        nreq = len(self.cur_req_ids)
        # decide row kinds now that the batch is known: per-token tensors have the (padded) token
        # count as leading dim; logits-phase tensors have the request count
        recs = [r for r in self.registry.records if r.host is not None]
        per_req: dict[str, list[tuple[Record, int, int]]] = {rid: [] for rid in self.cur_req_ids}
        for r in recs:
            if r.row_kind in ("token", "moe_slot"):
                if r.shape[0] not in (n_tokens, n_padded):
                    r.row_kind = "none"
                    self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
                    continue
                for i in range(n_tokens):
                    rid = self.cur_req_ids[self.cur_rows_req[i]]
                    per_req[rid].append((r, i, int(self.cur_rows_pos[i])))
            elif r.row_kind == "logits":
                if r.shape[0] != nreq:
                    self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
                    continue
                for i in range(nreq):
                    rid, t = self.cur_logits_map[i]
                    # a partial prefill chunk's logits row is computed but its token is
                    # discarded by the scheduler: not a sampled position, no leaf
                    if self._is_sampled_position(rid, t):
                        per_req[rid].append((r, i, t))
        for i, rid in enumerate(self.cur_req_ids):
            step = self.req_steps.get(rid, 0)
            positions = [int(self.cur_nct[i] + k) for k in range(int(self.cur_qsl[i + 1] - self.cur_qsl[i]))]
            plen = self.prompt_lens.get(rid)
            # RU phase: prefill / decode as before; "mixed" when a post-preemption recompute chunk
            # spans prompt and generated positions (each leaf's ph is position-determined, R4)
            if plen is None or positions[-1] < plen:
                phase = "prefill"
            elif positions[0] >= plen:
                phase = "decode"
            else:
                phase = "mixed"
            emitted = self.req_emitted_positions.setdefault(rid, set())
            again = [t for t in positions if t in emitted]
            recompute = bool(again)
            if recompute:
                self.recompute_steps.append({"request_id": rid, "step": step, "engine_step": self.step_index, "recomputed": len(again),
                                             "new": len(positions) - len(again), "positions": [positions[0], positions[-1]]})
            emitted.update(positions)
            self.req_records.setdefault(rid, []).append(StepRecord(step, phase, tuple(positions), recompute))
            self.req_steps[rid] = step + 1
            entries = per_req[rid]
            self._write_ru(rid, step, phase, positions, entries, i, recompute)
        self.timing["emit_s"] += time.perf_counter() - t1

    def _phase_at(self, rid: str, t: int) -> str:
        plen = self.prompt_lens.get(rid)
        return "prefill" if plen is None or t < plen else "decode"

    def _is_sampled_position(self, rid: str, t: int) -> bool:
        """The scheduler keeps a step's sampled token only when the step reaches the request's
        current end (prompt + tokens sampled so far); a partial prefill chunk's logits row is
        discarded. For a non-preempted trajectory this is exactly ``t >= prompt_len - 1`` (decode
        advances one position per step); after a preemption (R4) a recompute chunk can END inside
        the generated region without being the request's current end, so the count is tracked."""
        plen = self.prompt_lens.get(rid)
        if plen is None:
            return True
        return t == plen + self.req_sampled.get(rid, 0) - 1

    def _write_ru(self, rid: str, step: int, phase: str, positions: list[int], entries, req_index: int, recompute: bool = False) -> None:
        leaves: dict[str, dict[str, Any]] = {}
        families: list[dict[str, Any]] = []
        ru = None
        if self.writer is not None:
            ru = self.writer.ru(rid, step, phase=phase, positions=positions, on_duplicate="flag")
        has_moe = False
        for r, row, t in entries:
            layer = self.templates[r.node].layer
            dtype = str(r.dtype).replace("torch.", "")
            # capture-format v1.1 (W10): the id's phase is determined by the position (prefill iff
            # t < prompt_len), not by the step's scheduling phase (a recompute chunk is "mixed")
            id_phase = self._phase_at(rid, t)
            if r.row_kind == "moe_slot":
                # slotted node: the top_k routed slots are captured under e=<expert>, the other
                # slots of the fixed population are derived zero leaves (compact family record)
                has_moe = True
                E = int(r.extra["experts"])
                ids = [int(v) for v in r.extra["ids"].host[row].reshape(-1).tolist()]
                fam = str(value_id(self.profile_id, rid, t=t, phase=id_phase, layer=layer, op=r.node, port=r.port, shard=self.shard))
                shape = [int(r.shape[-1])]
                hist = self.moe_routes.setdefault(layer, [0] * E)
                for k, e in enumerate(ids):
                    cid = f"{fam}/e={e}"
                    data = r.host[row, k].reshape(-1).view(torch.uint8).numpy().tobytes()
                    if ru is not None:
                        ru.add(cid, dtype, shape, data)
                    else:
                        leaves[cid] = {"dtype": dtype, "shape": shape, "nbytes": len(data), "leaf": hashing.leaf_hash(cid, dtype, shape, data).hex()}
                    self.leaf_count += 1
                    self.bytes_captured += len(data)
                    hist[e] += 1
                family = {"family": fam, "slot_field": "e", "slots": E, "active": sorted(ids), "dtype": dtype, "shape": shape, "fill": "zeros"}
                if ru is not None:
                    ru.add_derived_family(fam, slots=E, active=ids, dtype=dtype, shape=shape)
                else:
                    families.append(family)
                self.leaf_count += E - len(ids)
                self.moe_active_slots += len(ids)
                self.moe_inactive_slots += E - len(ids)
                continue
            cid = str(value_id(self.profile_id, rid, t=t, phase=id_phase, layer=layer, op=r.node, port=r.port, shard=self.shard))
            hostrow = r.host[row].reshape(-1)
            data = hostrow.view(torch.uint8).numpy().tobytes()
            shape = list(r.shape[1:])
            if ru is not None:
                ru.add(cid, dtype, shape, data)
            else:
                leaves[cid] = {"dtype": dtype, "shape": shape, "nbytes": len(data), "leaf": hashing.leaf_hash(cid, dtype, shape, data).hex()}
            self.leaf_count += 1
            self.bytes_captured += len(data)
        if has_moe:
            # the batch's token count M of this step is a scheduling fact, not a value of any token:
            # it is recorded per step (steptrace, capture_summary.moe.triton_configs_by_num_tokens)
            # and the fused-MoE launch config it selected is a declared const of the expert node.
            self.moe_steps_num_tokens.setdefault(str(int(self.cur_num_tokens)), 0)
            self.moe_steps_num_tokens[str(int(self.cur_num_tokens))] += len(positions)
        # sampler.select at the request's last position of this step
        final = self._is_sampled_position(rid, positions[-1])
        if final:
            self.req_sampled[rid] = self.req_sampled.get(rid, 0) + 1
            self.cur_sampled_positions.add((rid, positions[-1]))
        if self.cur_sampled is not None and final:
            t = positions[-1]
            tok = int(self.cur_sampled[req_index])
            cid = str(value_id(self.profile_id, rid, t=t, phase=self._phase_at(rid, t), layer="sampler", op="sampler.select", port=0, shard=self.shard))
            data = np.array(tok, dtype=np.int32).tobytes()
            if ru is not None:
                ru.add(cid, "int32", [], data)
            else:
                leaves[cid] = {"dtype": "int32", "shape": [], "nbytes": 4, "leaf": hashing.leaf_hash(cid, "int32", [], data).hex()}
            self.leaf_count += 1
            self.req_outputs.setdefault(rid, []).append(tok)
            if "sampler.select" not in self.templates:
                # under tensor parallelism the sampler reads the all-gathered logits (every rank holds the
                # full row); unsharded, the lm_head GEMM's row itself
                logits_node = "lm_head.all_gather" if "lm_head.all_gather" in self.templates else "lm_head.gemm"
                self.templates["sampler.select"] = NodeTemplate(
                    op="sampler.select", layer="sampler", family="sampler", op_type="sampler", runtime_op="vllm.v1.worker.gpu.sample",
                    runtime_impl="vllm.v1.worker.gpu.sample.sampler (greedy argmax; trace replay overwrites the emitted token)",
                    args=[{"name": "logits", "source": "node", "node": logits_node, "port": 0},
                          {"name": "sampling_params", "source": "boundary", "field": "sampling_params"},
                          {"name": "seed", "source": "boundary", "field": "seed"}], row_kind="logits",
                )
            self.templates["sampler.select"].instances += 1
            self.templates["sampler.select"].ports.setdefault(0, {"dtype": "int32", "row_shape": [], "row_kind": "logits"})
        for tpl in self.templates.values():
            tpl.phases.update({self._phase_at(rid, positions[0]), self._phase_at(rid, positions[-1])})
        if ru is not None:
            ru.finish()
        else:
            d = self.run_dir / "ru" / rid / f"{step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            doc = {"request_id": rid, "step": step, "phase": phase, "positions": positions, "leaves": leaves}
            if families:
                doc["families"] = families
            if recompute:
                doc["recompute"] = True
            (d / "leaves.json").write_text(json.dumps(doc))
        self.ru_count += 1

    # ---------------- finish ----------------
    def boundaries(self, requests: dict[str, dict[str, Any]]) -> list[Boundary]:
        out = []
        for rid, info in requests.items():
            steps = self.req_records.get(rid, [])
            b = Boundary.build(
                rid, info["prompt_token_ids"], info["output_token_ids"], info["sampling_params"],
                seed=info.get("seed"), finish_reason=info.get("finish_reason"), steps=steps, max_tokens=info["sampling_params"].get("max_tokens"),
                chosen_logprobs=info.get("chosen_logprobs"),  # WB2: bound as float32 output_logprob leaves when present
                cached_prefix=self.cached_prefix_of(rid, len(info["prompt_token_ids"]), len(info["output_token_ids"])),
            )
            out.append(b)
        return out

    def cached_prefix_of(self, rid: str, prompt_len: int, n_out: int) -> list[CachedPrefix]:
        """R3: the positions this request never computed itself, each attributed to the request whose
        K/V rows it read there (owner map at read time), grouped by source. Empty when every
        position was computed by the request (the B0/B1 rows)."""
        covered = {t for s in self.req_records.get(rid, []) for t in s.positions}
        processed = prompt_len + max(n_out - 1, 0)
        missing = [t for t in range(processed) if t not in covered]
        if not missing:
            return []
        reads = self.cached_reads.get(rid, {})
        by_src: dict[str, list[int]] = {}
        for t in missing:
            src = reads.get(t)
            if src is None:
                # never computed and never read from anyone else's rows: leave it uncovered so the
                # boundary validation (and the replay) fails loudly instead of inventing a source
                continue
            by_src.setdefault(src, []).append(t)
        return [CachedPrefix(src, self.prompt_lens.get(src, 0), tuple(ts)) for src, ts in sorted(by_src.items())]

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.step_index,
            "replay_units": self.ru_count,
            "leaves": self.leaf_count,
            "bytes_captured": self.bytes_captured,
            "templates": len(self.templates),
            "unbound_inputs": len(self.unbound),
            "unbound_examples": self.unbound[:10],
            "module_outputs_not_recorded": len(self.module_audit),
            "non_token_outputs": self.non_token_outputs,
            "module_audit_examples": self.module_audit[:10],
            "kv_reads_unresolved": sum(a.get("unresolved", 0) for a in self.kv_audit),
            "kv_reads_checked": sum(a.get("positions_read", 0) for a in self.kv_audit),
            "kv_groups": getattr(self, "kv_groups", None),
            # state rows (W10): cross-request KV reads bound under the R3 identity rule, what the
            # own-request-only rule would have left unresolved, cached positions per request, and
            # post-preemption recompute steps (R4)
            "kv_cross_request_rule": self.cross_request_kv,
            "kv_reads_cross_request_resolved": self.kv_cross_request_resolved,
            "kv_reads_unresolved_under_own_request_rule": self.kv_unresolved_under_own_request_rule,
            "cached_positions_per_request": {rid: len(v) for rid, v in self.cached_reads.items()},
            "cached_sources_per_request": {rid: sorted({s for s in v.values()}) for rid, v in self.cached_reads.items()},
            "recompute_steps": self.recompute_steps,
            "requests_with_recompute": len({r["request_id"] for r in self.recompute_steps}),
            "errors": self.errors[:20],
            "op_histogram": self.op_histogram,
            "timing": self.timing,
            "arena_overflow_allocs": self.arena.overflow_allocs,
            "patches": getattr(self, "patches_doc", PATCHES),
            # §15.8 witness of activation for the MoE block: routes actually exercised per layer
            "moe": {
                "active_slots": self.moe_active_slots,
                "inactive_slots": self.moe_inactive_slots,
                "routes_per_layer": {layer: hist for layer, hist in sorted(self.moe_routes.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)},
                "experts_exercised_per_layer": {layer: sum(1 for c in hist if c) for layer, hist in self.moe_routes.items()},
                "triton_configs_by_num_tokens": self.moe_configs,
                "positions_by_num_tokens": self.moe_steps_num_tokens,
                "skipped_internal_ops": self.moe_skipped_ops,
                "expert_params": sorted(self.moe_expert_params),
            } if self.moe_routes else None,
        }

    def description(self, model_repo: str, symbols: dict[str, int], extra_nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """description.json (contract §6) in W2's schema. Args are identity rules; the adapter's
        column slices, constants and runtime op names ride in node-level extras
        (``arg_cols``, ``consts``, ``runtime_op``) that W2's parser keeps in ``Node.extra``."""
        from verity_vllm_adapter.relations_gpu import RELATION_FOR_RUNTIME_OP

        nodes = []
        for tpl in sorted(self.templates.values(), key=lambda t: node_sort_key(t.op)):
            p0 = tpl.ports.get(0) or next(iter(tpl.ports.values()))
            args = []
            arg_cols: dict[str, list[int]] = {}
            unbound: list[str] = []
            for a in tpl.args:
                src = a["source"]
                name = a["name"]
                if name == "layer_name":
                    continue
                if src == "node":
                    d: dict[str, Any] = {"name": name, "source": "node", "op": a["node"], "port": a.get("port", 0)}
                    if tpl.family == "attn" and name in ("k", "v"):
                        d["t"] = "all_le"
                    if a.get("cols"):
                        arg_cols[name] = list(a["cols"])
                    # a non-slotted consumer of a slotted producer reads the whole population
                    # (the combination sums the routed slots; which ones is the checked router's
                    # result, never a stored mask)
                    target = self.templates.get(a["node"])
                    if target is not None and target.experts and not tpl.experts:
                        d["expert"] = "all"
                elif src == "weight":
                    d = {"name": name, "source": "weight", "param": a["param"]}
                    for k in ("expert", "lazy"):
                        if k in a:
                            d[k] = a[k]
                elif src == "token":
                    d = {"name": name, "source": "token", "which": "input"}
                elif src == "boundary":
                    d = {"name": name, "source": "boundary", "field": a["field"]}
                elif src == "positions":
                    d = {"name": name, "source": "const", "value": "positions"}
                else:
                    d = {"name": name, "source": "const", "value": a.get("detail")}
                    if src == "unbound":
                        unbound.append(name)
                args.append(d)
            relation = tpl.relation or RELATION_FOR_RUNTIME_OP.get(tpl.runtime_op, "unresolved")
            if tpl.op == "sampler.select":
                relation = "sampler.select.v1"
            if tpl.op_type == "lm_head":
                relation = "lm_head.same_runtime.v1"
            if tpl.family == "combine":
                # the reduction over the routed slots needs the router's ids to know which of the
                # population's slots contribute, in route order
                router = tpl.module_path + ".topk"
                if router in self.templates and not any(x.get("name") == "ids" for x in args):
                    args.append({"name": "ids", "source": "node", "op": router, "port": 0})
            if unbound:
                relation = "unresolved"
            node = {
                "op": tpl.op,
                "layer": tpl.layer,
                "op_type": tpl.op_type,
                "dtype": p0["dtype"],
                "shape": p0["row_shape"],
                "ports": max(tpl.ports) + 1 if tpl.ports else 1,
                "outputs": [{"port": k, "dtype": v["dtype"], "shape": v["row_shape"]} for k, v in sorted(tpl.ports.items()) if k != 0],
                "phases": sorted(tpl.phases) or ["prefill", "decode"],
                "positions": "sampled" if tpl.row_kind == "logits" or p0.get("row_kind") == "logits" else "all",
                "args": args,
                "relation": relation,
                "origin": "captured",
                "decompositions": [{"id": "row", "interface": self._interface(tpl)}],
                "runtime_impl": tpl.runtime_impl,
                # adapter extras (kept by W2's parser in Node.extra)
                "family": tpl.family,
                "runtime_op": tpl.runtime_op,
                "consts": tpl.consts,
                "arg_cols": arg_cols,
                "unbound_args": unbound,
                "module_path": tpl.module_path,
                "instances_observed": tpl.instances,
                "arg_violations": tpl.arg_violations[:5],
            }
            if tpl.experts:
                node["experts"] = int(tpl.experts)
            node.update(tpl.extra)
            nodes.append(node)
        for n in extra_nodes or []:
            nodes.append(n)
        doc = {
            "schema": "verity-vllm/description/v1",
            "profile_id": self.profile_id,
            "model": model_repo,
            "symbols": symbols,
            "interception": getattr(self, "interception", "dispatch-mode"),
            "patches": getattr(self, "patches_doc", PATCHES),
            "nodes": nodes,
        }
        if self.shard is not None:
            doc["shard"] = int(self.shard)  # one rank's view; tp_merge joins the ranks' descriptions
        return doc

    def _interface(self, tpl: NodeTemplate) -> dict[str, Any]:
        p0 = tpl.ports.get(0) or next(iter(tpl.ports.values()))
        out_elems = int(math.prod(p0["row_shape"])) if p0["row_shape"] else 1
        out_bits = out_elems * _dtype_bits(p0["dtype"])
        in_bytes = 0
        for a in tpl.args:
            if a.get("source") == "weight":
                in_bytes += 0  # weights are opened per tensor; counted separately in Table 2 notes
            else:
                shp = a.get("shape") or []
                in_bytes += (int(math.prod(shp[1:])) if len(shp) > 1 else 1) * _dtype_bits(a.get("dtype", "bfloat16")) // 8
        return {"input_bytes": in_bytes, "output_bits": out_bits, "range": "row"}


def _extent(t: torch.Tensor) -> int:
    """One past the last byte a (positively strided) view can touch."""
    if t.numel() == 0:
        return t.data_ptr()
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return t.data_ptr() + (last + 1) * t.element_size()


def _dtype_bits(dtype: str) -> int:
    return {"bfloat16": 16, "float16": 16, "float32": 32, "int64": 64, "int32": 32, "int8": 8, "uint8": 8, "bool": 8, "float8_e4m3fn": 8}.get(dtype, 16)


_LAYER_PREFIX = re.compile(r"^layers\.\d+\.")


DRAFT_PREFIX = "draft."  # lane SD (speculative.py): the draft model's modules/params live under this prefix


def normalize_module(name: str) -> str:
    # lane SD (additive): vLLM loads the draft model with prefix "draft_model" (DraftModelProposer._get_model),
    # so its attention layers announce themselves as "draft_model.model.layers.N.self_attn.attn"
    draft = False
    if name.startswith("draft_model."):
        name = name[len("draft_model.") :]
        draft = True
    elif name.startswith(DRAFT_PREFIX):
        name = name[len(DRAFT_PREFIX) :]
        draft = True
    if name.startswith("model."):
        name = name[len("model.") :]
    if name in ("logits_processor",):
        name = "lm_head"
    return DRAFT_PREFIX + name if draft else name


def layer_of(module: str) -> str:
    if module.startswith(DRAFT_PREFIX):  # lane SD (additive): the draft model's layers are indexed like the target's
        module = module[len(DRAFT_PREFIX) :]
    parts = module.split(".")
    if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
        return parts[1]
    if module.startswith("embed_tokens"):
        return "embed"
    if module.startswith("lm_head"):
        return "lm_head"
    if module == "norm" or module.startswith("norm."):
        return "norm"
    if module == "sample":
        return "sampler"
    return "norm" if "norm" in module else "embed"


def node_name(module: str, family: str, counter: dict[tuple[str, str], int], runtime_op: str) -> str:
    suffix = {"gemm": "gemm", "embedding": "embedding", "rmsnorm": "rmsnorm", "rope": "rope", "act_mul": "act_mul", "attn": "attn", "add": "add", "gather": "gather", "cast": "cast", "kv_write": "kv_write",
              "router": "topk", "combine": "combine", "sigmoid": "sigmoid", "mul": "mul",
              "all_reduce": "all_reduce", "all_gather": "all_gather",
              "fp8_quant": "fp8_quant"}.get(family, runtime_op.replace(".", "_"))  # [QZ, additive] fp8_quant
    base = f"{module}.{suffix}" if module not in ("top", "sample") else suffix
    if module == "sample" and family == "gemm":
        base = "lm_head.gemm"
    k = counter.get((module, suffix), 0)
    counter[(module, suffix)] = k + 1
    return base if k == 0 else f"{base}{k}"


def node_sort_key(op: str) -> tuple:
    if op.startswith(DRAFT_PREFIX):  # lane SD (additive): the draft model's nodes sort after the target's
        return (5,) + node_sort_key(op[len(DRAFT_PREFIX) :])[1:] + (op,)
    parts = op.split(".")
    if parts[0] == "layers" and parts[1].isdigit():
        return (1, int(parts[1]), op)
    if parts[0] == "embed_tokens":
        return (0, 0, op)
    if parts[0] == "norm":
        return (2, 0, op)
    if parts[0] == "lm_head":
        return (3, 0, op)
    return (4, 0, op)
