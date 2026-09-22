"""``same_runtime`` relation checkers for the B0/B1 op families (contract §7, class same_runtime).

Each checker receives a :class:`verity_vllm.description.NodeInstance` and the *authenticated*
opened inputs (W2's runner verified every opening against the committed roots before calling
us) and returns the exact bytes it predicts for the committed output row. It re-executes the
**registered runtime operator** on the GPU:

==============================  ==========================================================
relation                        runtime op re-executed (identical library call)
==============================  ==========================================================
embedding.same_runtime.v1       torch.nn.functional.embedding (aten::embedding, CUDA)
gemm.same_runtime.v1            vllm ...batch_invariant.matmul_persistent(x, w.t()) (Triton)
lm_head.same_runtime.v1         same as gemm at the vocab shape
add.same_runtime.v1             aten::add (bias add of the qkv projection, Qwen2)
rmsnorm.same_runtime.v1         vllm ...batch_invariant.rms_norm_batch_invariant (Triton) or
                                torch.ops._C.fused_add_rms_norm (CUDA), by the node's runtime_op
rope.same_runtime.v1            torch.ops._C.rotary_embedding (CUDA)
silu_mul.same_runtime.v1        torch.ops._C.silu_and_mul (CUDA)
attention.same_runtime.v1       vllm.vllm_flash_attn.flash_attn_varlen_func on a paged KV cache
                                rebuilt from the opened K/V rows with reshape_and_cache_flash
                                (the two calls FlashAttentionImpl.forward / unified_kv_cache_update
                                make), num_splits=1 (batch-invariant profile)
==============================  ==========================================================

The checkers never see the live replay process; every operand is an opened leaf, a weight leaf,
or a description constant (``inst.node.extra["consts"]`` / ``["arg_cols"]``), and the operator
is applied to the **row alone** (M = 1): agreement with the row captured inside a batch of
arbitrary composition is precisely the batch-invariance promise the profile relies on.

``includes_full_op`` is True for every relation here (the checker re-runs the whole operator
for the row; for attention the whole causal prefix 0..t).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from verity_vllm import dtypes
from verity_vllm import relations as rel
from verity_vllm.capture import Value
from verity_vllm.description import NodeInstance
from verity_vllm.relations import SAME_RUNTIME, Inputs, RelationError, need

DEVICE = "cuda"

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "bool": torch.bool,
}

_ORIG: dict[str, Any] = {}


def _runtime() -> dict[str, Any]:
    """The vLLM runtime functions, resolved once. Uses the *original* kernels (the capture's
    dispatcher-visible wrappers are never installed in the verifier process)."""
    if _ORIG:
        return _ORIG
    import vllm._custom_ops as ops
    import vllm.model_executor.determinism.batch_invariant as bi
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    _ORIG.update(
        {
            "matmul_persistent": bi.matmul_persistent,
            "rms_norm_batch_invariant": bi.rms_norm_batch_invariant,
            "fused_add_rms_norm": torch.ops._C.fused_add_rms_norm,
            "rotary_embedding": torch.ops._C.rotary_embedding,
            "silu_and_mul": torch.ops._C.silu_and_mul,
            "reshape_and_cache_flash": ops.reshape_and_cache_flash,
            "flash_attn_varlen_func": flash_attn_varlen_func,
        }
    )
    return _ORIG


def to_tensor(value: Value, device: str = DEVICE) -> torch.Tensor:
    dt = _TORCH_DTYPES.get(value.dtype)
    if dt is None:
        raise RelationError(f"{value.canonical_id}: unsupported dtype {value.dtype}")
    t = torch.frombuffer(bytearray(value.data), dtype=dt)
    return t.reshape(tuple(value.shape)).to(device)


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def consts(inst: NodeInstance) -> dict[str, Any]:
    return dict(inst.node.extra.get("consts", {}))


def arg_cols(inst: NodeInstance, name: str) -> tuple[int, int] | None:
    cols = inst.node.extra.get("arg_cols", {}).get(name)
    return (int(cols[0]), int(cols[1])) if cols else None


def row(inputs: Inputs, inst: NodeInstance, name: str) -> torch.Tensor:
    """One opened input row (with the producer's column slice applied), as [1, ...]."""
    v = need(inputs, name)
    t = to_tensor(v)
    cols = arg_cols(inst, name)
    if cols is not None:
        t = t.reshape(-1)[cols[0] : cols[1]]
    return t.reshape(1, *t.shape) if t.dim() == 1 else t.reshape(1, -1)


def rows(inputs: Inputs, inst: NodeInstance, name: str) -> torch.Tensor:
    """An ``all_le`` input: rows t'=0..t stacked as [t+1, ...] (column slice applied)."""
    vs = inputs.get(name)
    if not isinstance(vs, list) or not vs or not all(isinstance(v, Value) for v in vs):
        raise RelationError(f"{name}: expected a list of opened rows")
    cols = arg_cols(inst, name)
    ts = []
    for v in vs:
        t = to_tensor(v).reshape(-1)
        if cols is not None:
            t = t[cols[0] : cols[1]]
        ts.append(t)
    return torch.stack(ts, 0)


def weight(inputs: Inputs, name: str) -> torch.Tensor:
    return to_tensor(need(inputs, name))


def runtime_op(inst: NodeInstance) -> str:
    return str(inst.node.extra.get("runtime_op", ""))


# --- checkers ------------------------------------------------------------------------------------


@rel.register("embedding.same_runtime.v1", klass=SAME_RUNTIME, op_type="embedding", includes_full_op=True, owner="W1",
              doc="aten::embedding on the opened table and the boundary token at t (CUDA).")
def embedding_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    token = rel.scalar_int(need(inputs, "token"))
    table = weight(inputs, "table")
    ids = torch.tensor([token], dtype=torch.int64, device=table.device)
    out = torch.nn.functional.embedding(ids, table)
    return tensor_bytes(out[0])


def _gemm(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _runtime()
    x = row(inputs, inst, "x")
    w = weight(inputs, "w")
    op = runtime_op(inst)
    if op in ("verity.bi_mm", "inductor.matmul_kernel_persistent"):
        # the compiled pieces inline the very same user-defined Triton kernel (vLLM's
        # matmul_kernel_persistent, same constexpr config) via triton_kernel_wrapper_mutation; the
        # standalone launch through vLLM's Python entry point is the same kernel source (§16.1)
        out = rt["matmul_persistent"](x, w.t())
    elif op in ("aten.mm", "aten.matmul"):
        out = torch.mm(x, w.t())
    elif op == "aten.linear":
        out = torch.nn.functional.linear(x, w)
    else:
        raise RelationError(f"gemm relation: unknown runtime_op {op!r}")
    return tensor_bytes(out[0])


rel.register("gemm.same_runtime.v1", klass=SAME_RUNTIME, op_type="gemm", includes_full_op=True, owner="W1",
             doc="vLLM batch-invariant Triton GEMM (matmul_persistent) on the opened row and weight, M=1.")(_gemm)
rel.register("lm_head.same_runtime.v1", klass=SAME_RUNTIME, op_type="lm_head", includes_full_op=True, owner="W1",
             doc="same GEMM at the vocab shape on the opened final-norm row and lm_head weight.")(_gemm)


@rel.register("add.same_runtime.v1", klass=SAME_RUNTIME, op_type="residual_add", includes_full_op=True, owner="W1",
              doc="aten::add (CUDA, bf16) of the opened row and the bias weight.")
def add_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    a = row(inputs, inst, "a")
    b = inputs.get("b")
    bt = to_tensor(b) if isinstance(b, Value) else None
    if bt is None:
        raise RelationError("add: missing b")
    out = a + bt.reshape(1, -1) if bt.dim() == 1 else a + bt
    return tensor_bytes(out[0])


@rel.register("rmsnorm.same_runtime.v1", klass=SAME_RUNTIME, op_type="rmsnorm", includes_full_op=True, owner="W1",
              doc="verity.bi_rms_norm -> vLLM Triton rms_norm_batch_invariant; _C.fused_add_rms_norm -> the CUDA op "
                  "(port 0 normed output, port 1 the new residual).")
def rmsnorm_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _runtime()
    c = consts(inst)
    eps = float(c.get("eps", 1e-6))
    x = row(inputs, inst, "x").clone()
    w = weight(inputs, "w") if isinstance(inputs.get("w"), Value) else None
    op = runtime_op(inst)
    if op == "verity.bi_rms_norm":
        out = rt["rms_norm_batch_invariant"](x, w, eps, None)
        return tensor_bytes(out[0])
    if op == "_C.fused_add_rms_norm":
        residual = row(inputs, inst, "residual").clone()
        rt["fused_add_rms_norm"](x, residual, w, eps)
        return tensor_bytes((x if inst.port == 0 else residual)[0])
    if op == "_C.rms_norm":
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, w, eps)
        return tensor_bytes(out[0])
    raise RelationError(f"rmsnorm relation: unknown runtime_op {op!r}")


@rel.register("rope.same_runtime.v1", klass=SAME_RUNTIME, op_type="rope", includes_full_op=True, owner="W1",
              doc="torch.ops._C.rotary_embedding at position t on the opened q/k column slices of the qkv row "
                  "and the cos_sin_cache buffer (port 0 q, port 1 k).")
def rope_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _runtime()
    c = consts(inst)
    q = row(inputs, inst, "q").clone()
    k = row(inputs, inst, "k").clone()
    cache = weight(inputs, "cos_sin_cache")
    positions = torch.tensor([inst.t], dtype=torch.int64, device=q.device)
    rt["rotary_embedding"](positions, q, k, int(c["head_size"]), cache, bool(c["is_neox"]))
    return tensor_bytes((q if inst.port == 0 else k)[0])


@rel.register("silu_mul.same_runtime.v1", klass=SAME_RUNTIME, op_type="silu_mul", includes_full_op=True, owner="W1",
              doc="torch.ops._C.silu_and_mul on the opened gate_up row.",
              # verity_vllm.verify --torch installs relations_torch (W2's torch.nn.functional version, arg
              # `gate_up`) before the plugins; adapter captures declare the arg `x` and ran the vLLM kernel,
              # so this registration replaces it when both are loaded (bundle verify --torch require --plugin ...).
              replace=True)
def silu_mul_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _runtime()
    x = row(inputs, inst, "x")
    out = torch.empty((1, x.shape[1] // 2), dtype=x.dtype, device=x.device)
    rt["silu_and_mul"](out, x)
    return tensor_bytes(out[0])


@rel.register("attention.same_runtime.v1", klass=SAME_RUNTIME, op_type="attention", includes_full_op=True, owner="W1",
              doc="vLLM FlashAttention (flash_attn_varlen_func, paged KV rebuilt with reshape_and_cache_flash from the "
                  "opened post-RoPE K rows and V slices 0..t, num_splits=1) for the one query row at t.")
def attention_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _runtime()
    c = consts(inst)
    H, KVH, D = int(c["num_heads"]), int(c["num_kv_heads"]), int(c["head_size"])
    bs = int(c["block_size"])
    q = row(inputs, inst, "q").reshape(1, H, D)
    k = rows(inputs, inst, "k").reshape(-1, KVH, D)
    v = rows(inputs, inst, "v").reshape(-1, KVH, D)
    n = k.shape[0]
    if n != inst.t + 1 or v.shape[0] != n:
        raise RelationError(f"attention: {n} K rows / {v.shape[0]} V rows for t={inst.t}")
    nblocks = (n + bs - 1) // bs
    # the runtime's KV cache layout for FlashAttention: [2, num_blocks, block_size, KVH, D]
    kv = torch.zeros((2, nblocks, bs, KVH, D), dtype=q.dtype, device=q.device)
    key_cache, value_cache = kv[0], kv[1]
    slot_mapping = torch.arange(n, dtype=torch.int64, device=q.device)
    k_scale = torch.ones((), dtype=torch.float32, device=q.device)
    v_scale = torch.ones((), dtype=torch.float32, device=q.device)
    rt["reshape_and_cache_flash"](k, v, key_cache, value_cache, slot_mapping, str(c.get("kv_cache_dtype", "auto")), k_scale, v_scale)
    block_table = torch.arange(nblocks, dtype=torch.int32, device=q.device).reshape(1, -1)
    out = torch.empty((1, H, D), dtype=q.dtype, device=q.device)
    window = c.get("sliding_window")
    rt["flash_attn_varlen_func"](
        q=q, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=q.device), max_seqlen_q=1,
        seqused_k=torch.tensor([n], dtype=torch.int32, device=q.device), max_seqlen_k=n,
        softmax_scale=float(c["scale"]), causal=True, alibi_slopes=None,
        window_size=list(window) if window else None, block_table=block_table,
        softcap=float(c.get("logits_soft_cap") or 0.0), scheduler_metadata=None,
        fa_version=int(c.get("fa_version", 2)), q_descale=None, k_descale=None, v_descale=None,
        num_splits=int(c.get("num_splits", 1)),
    )
    return tensor_bytes(out[0])


# --- MoE block (W9) --------------------------------------------------------------------------------


def _moe_runtime() -> dict[str, Any]:
    if "topk_softmax" not in _ORIG:
        import triton.language as tl
        import vllm.model_executor.layers.fused_moe.fused_moe as fm
        from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config

        # the *original* launch function (the capture's reporting wrapper is never installed here)
        invoke = getattr(fm, "invoke_fused_moe_triton_kernel", invoke_fused_moe_triton_kernel)
        _ORIG.update({
            "topk_softmax": torch.ops._moe_C.topk_softmax,
            "moe_sum": torch.ops._moe_C.moe_sum,
            "invoke_fused_moe_triton_kernel": invoke,
            "try_get_optimal_moe_config": try_get_optimal_moe_config,
            "tl": tl,
        })
    return _ORIG


def _lazy(inputs: Inputs, name: str) -> Value:
    v = inputs.get(name)
    if isinstance(v, rel.LazyInput):
        v = v.get()
    if not isinstance(v, Value):
        raise RelationError(f"missing input {name!r}")
    return v


@rel.register("moe.topk.weights.same_runtime.v1", klass=SAME_RUNTIME, op_type="router", includes_full_op=True, owner="W9",
              doc="torch.ops._moe_C.topk_softmax on the one opened float32 router-logits row (M=1): the fused "
                  "kernel's softmax probabilities of the top_k experts (renormalize per the node const), float32 [top_k].")
def moe_topk_weights_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _moe_runtime()
    c = consts(inst)
    logits = row(inputs, inst, "logits").to(torch.float32)
    k = int(inst.node.port_shape(inst.port)[-1])
    weights = torch.empty((1, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((1, k), dtype=torch.int32, device=logits.device)
    tei = torch.empty((1, k), dtype=torch.int32, device=logits.device)
    if "e_score_correction_bias" in inputs:
        raise RelationError("moe.topk.weights.same_runtime.v1: e_score_correction_bias is not supported")
    # (weights_out, ids_out, token_expert_indices, logits, renormalize, e_score_correction_bias=None,
    #  is_padding=None): the batch padding mask is a const of the node (False on every committed row)
    rt["topk_softmax"](weights, ids, tei, logits, bool(c.get("renormalize", False)), None, None)
    return tensor_bytes(weights[0])


@rel.register("moe.expert_slot.same_runtime.v1", klass=SAME_RUNTIME, op_type="expert", includes_full_op=True, owner="W9",
              doc="Expert slot e at token t. Activity is decided from the CHECKED router ids (an opened, authenticated "
                  "input), never from a stored mask: e not in ids -> the prescribed zero contribution (derived leaf, "
                  "no weight is opened); e in ids -> vLLM's fused-MoE Triton kernel pair re-launched for the one "
                  "(token, expert) row: w13 GEMM -> _C.silu_and_mul -> w2 GEMM * routed weight, with the node's declared "
                  "Triton launch config (const triton_config, recorded from the run; revision-1 captures instead open the "
                  "batch.num_tokens leaf and re-select the config vLLM picks for that token count), on the per-expert "
                  "weight leaves opened lazily.")
def moe_expert_slot_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    rt = _moe_runtime()
    c = consts(inst)
    ids = need(inputs, "ids").array().reshape(-1).astype(np.int64)
    e = inst.expert
    if e is None:
        raise RelationError("expert slot relation needs an e= instance")
    K = int(inst.node.port_shape(inst.port)[-1]) if isinstance(inst.node.port_shape(inst.port)[-1], int) else int(inputs["__symbols__"][inst.node.port_shape(inst.port)[-1]])
    if inst.dtype != "bfloat16":
        raise RelationError(f"expert slot: unsupported dtype {inst.dtype}")
    hits = np.flatnonzero(ids == e)
    if len(hits) == 0:
        return bytes(K * 2)  # inactive: zero contribution, canonical +0.0 bf16 bytes
    k = int(hits[0])
    weights = need(inputs, "weights").array().reshape(-1).astype(np.float32)
    routed_w = float(weights[k])
    x = row(inputs, inst, "x").contiguous()  # [1, H] bf16
    w13 = to_tensor(_lazy(inputs, "w13"))  # [2N, H]
    w2 = to_tensor(_lazy(inputs, "w2"))  # [H, N]
    w1_shape = tuple(int(v) for v in c["w1_shape"])
    w2_shape = tuple(int(v) for v in c["w2_shape"])
    top_k = int(c["top_k"])
    if c.get("triton_config_conflicts"):
        raise RelationError("expert slot: the run recorded more than one Triton config for this node (see consts.triton_config_conflicts)")
    if isinstance(c.get("triton_config"), dict):
        config = {str(k_): v for k_, v in c["triton_config"].items()}
    else:
        M = rel.scalar_int(need(inputs, "num_tokens"))
        config = rt["try_get_optimal_moe_config"](w1_shape, w2_shape, top_k, None, M)
    tl = rt["tl"]
    ct = {"bf16": tl.bfloat16, "fp16": tl.float16, "fp32": tl.float32}[str(c.get("compute_type", "bf16"))]
    dev = x.device
    bm = int(config["BLOCK_SIZE_M"])
    N2 = w1_shape[1]
    N = w2_shape[2]
    # one (token, expert) pair, top_k = 1 for the re-run: row 0 of A is the token, expert 0 of the
    # single-expert weight tensors is e; padding rows carry the sentinel numel (= 1) like
    # moe_align_block_size writes them
    sorted_token_ids = torch.full((bm,), 1, dtype=torch.int32, device=dev)
    sorted_token_ids[0] = 0
    expert_ids = torch.zeros((1,), dtype=torch.int32, device=dev)
    num_tokens_post_padded = torch.tensor([bm], dtype=torch.int32, device=dev)
    flags = dict(use_fp8_w8a8=bool(c.get("use_fp8_w8a8", False)), use_int8_w8a8=bool(c.get("use_int8_w8a8", False)),
                 use_int8_w8a16=bool(c.get("use_int8_w8a16", False)), use_int4_w4a16=bool(c.get("use_int4_w4a16", False)),
                 per_channel_quant=bool(c.get("per_channel_quant", False)), block_shape=c.get("block_shape"), B_bias=None)
    cache1 = torch.empty((1, 1, N2), dtype=x.dtype, device=dev)
    rt["invoke_fused_moe_triton_kernel"](x, w13.unsqueeze(0), cache1, None, None, None, sorted_token_ids, expert_ids,
                                         num_tokens_post_padded, False, 1, config, compute_type=ct, **flags)
    cache2 = torch.empty((1, N), dtype=x.dtype, device=dev)
    _runtime()["silu_and_mul"](cache2, cache1.view(-1, N2))
    cache3 = torch.empty((1, 1, K), dtype=x.dtype, device=dev)
    tw = torch.tensor([[routed_w]], dtype=torch.float32, device=dev)
    rt["invoke_fused_moe_triton_kernel"](cache2, w2.unsqueeze(0), cache3, None, None, tw, sorted_token_ids, expert_ids,
                                         num_tokens_post_padded, True, 1, config, compute_type=ct, **flags)
    return tensor_bytes(cache3[0, 0])


@rel.register("sigmoid.same_runtime.v1", klass=SAME_RUNTIME, op_type="other", includes_full_op=True, owner="W9",
              doc="aten::sigmoid (CUDA) on the opened row (the shared expert's gate).")
def sigmoid_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    x = row(inputs, inst, "x")
    return tensor_bytes(torch.sigmoid(x)[0])


@rel.register("mul.same_runtime.v1", klass=SAME_RUNTIME, op_type="other", includes_full_op=True, owner="W9",
              doc="aten::mul (CUDA) of the two opened rows with broadcasting (shared-expert gate x shared-expert output).")
def mul_same_runtime(inst: NodeInstance, inputs: Inputs) -> bytes:
    a = row(inputs, inst, "a")
    b = row(inputs, inst, "b")
    return tensor_bytes((a * b)[0])


# --- tensor parallelism (lane TP, profile port b1-tp2) --------------------------------------------
# The collectives are ordinary functions of the ranks' captured partials: the description names the
# ranks' operands explicitly (`shard: r`), so the same relation checks every rank's replica.


@rel.register("allreduce.bf16.v1", klass=rel.INDEPENDENT, op_type="allreduce", includes_full_op=False, owner="TP",
              doc="Sum of the ranks' bf16 partials p0, p1, ... in rank order: each widened to f32, one IEEE add "
                  "per pair, RNE back to bf16 (for two ranks a single add; NCCL ring/tree order for more ranks is "
                  "not modelled and the relation is unverified there).")
def allreduce_bf16(inst: NodeInstance, inputs: Inputs) -> bytes:
    parts = []
    k = 0
    while inputs.get(f"p{k}") is not None:
        parts.append(need(inputs, f"p{k}"))
        k += 1
    if len(parts) < 2:
        raise RelationError("allreduce.bf16.v1 needs at least two partials p0, p1, ...")
    if any(p.dtype != "bfloat16" for p in parts):
        raise RelationError("allreduce.bf16.v1 needs bfloat16 partials")
    if any(p.shape != parts[0].shape for p in parts):
        raise RelationError(f"partial shape mismatch {[p.shape for p in parts]}")
    total = parts[0].float32()
    for p in parts[1:]:
        total = total + p.float32()
    return dtypes.f32_to_bf16_bytes(total)


@rel.register("allgather.copy.v1", klass=rel.INDEPENDENT, op_type="other", includes_full_op=False, owner="TP",
              doc="Concatenation of the ranks' captured slices p0, p1, ... along the last axis (vocab-parallel "
                  "lm_head logits): a gate-free copy, the bytes must match exactly.")
def allgather_copy(inst: NodeInstance, inputs: Inputs) -> bytes:
    parts = []
    k = 0
    while inputs.get(f"p{k}") is not None:
        parts.append(need(inputs, f"p{k}"))
        k += 1
    if len(parts) < 2:
        raise RelationError("allgather.copy.v1 needs at least two slices p0, p1, ...")
    if any(p.dtype != parts[0].dtype for p in parts):
        raise RelationError("allgather.copy.v1 slices must share a dtype")
    return b"".join(p.data for p in parts)


@rel.register("embedding.vocab_parallel.v1", klass=rel.INDEPENDENT, op_type="embedding", includes_full_op=False, owner="TP",
              doc="This rank's partial embedding: row (token - org_vocab_start_index) of the rank's table shard when "
                  "the token falls in [org_vocab_start_index, org_vocab_end_index), else zeros (the all-reduce that "
                  "follows selects the owning rank's row).")
def embedding_vocab_parallel(inst: NodeInstance, inputs: Inputs) -> bytes:
    token = rel.scalar_int(need(inputs, "token"))
    table = need(inputs, "table")
    extra = inst.node.extra or {}
    # the vocab range is the one per-rank constant of the model: the joined description keeps each
    # rank's copy under `consts_by_shard`
    consts = (extra.get("consts_by_shard") or {}).get(str(inst.shard)) or extra.get("consts") or {}
    if len(table.shape) != 2 or "org_vocab_start_index" not in consts or "org_vocab_end_index" not in consts:
        raise RelationError(f"{table.canonical_id}: need a 2-D table shard and the rank's vocab range")
    start, end = int(consts["org_vocab_start_index"]), int(consts["org_vocab_end_index"])
    rows, width = table.shape
    size = dtypes.itemsize(table.dtype)
    assert size is not None
    row_bytes = width * size
    if not start <= token < end:
        return bytes(row_bytes)
    local = token - start
    if not 0 <= local < rows:
        raise RelationError(f"token {token} maps to row {local} outside the shard's {rows} rows")
    return table.data[local * row_bytes: (local + 1) * row_bytes]


RELATION_FOR_RUNTIME_OP = {
    "vllm.all_reduce": "allreduce.bf16.v1",
    "vllm.all_gather": "allgather.copy.v1",
    "_C.vocab_parallel_embedding": "embedding.vocab_parallel.v1",
    "aten._to_copy": "cast.widen.v1",
    "_moe_C.moe_sum": "moe.combine.v1",
    "aten.sigmoid": "sigmoid.same_runtime.v1",
    "aten.mul": "mul.same_runtime.v1",
    "aten.embedding": "embedding.same_runtime.v1",
    "verity.bi_mm": "gemm.same_runtime.v1",
    "aten.mm": "gemm.same_runtime.v1",
    "aten.matmul": "gemm.same_runtime.v1",
    "aten.linear": "gemm.same_runtime.v1",
    "aten.add": "add.same_runtime.v1",
    "verity.bi_rms_norm": "rmsnorm.same_runtime.v1",
    "_C.fused_add_rms_norm": "rmsnorm.same_runtime.v1",
    "_C.rms_norm": "rmsnorm.same_runtime.v1",
    "_C.rotary_embedding": "rope.same_runtime.v1",
    "_C.silu_and_mul": "silu_mul.same_runtime.v1",
    "vllm.unified_attention_with_output": "attention.same_runtime.v1",
}


def install() -> list[str]:
    """Importing this module registers everything; kept for symmetry with W3's registry."""
    try:
        import verity_vllm_sampler.registry as sampler_registry

        sampler_registry.install()
    except Exception as e:  # noqa: BLE001
        print(f"[relations_gpu] sampler registry not installed: {e!r}")
    try:
        import verity_vllm_adapter.relations_inductor  # noqa: F401  (registers inductor.fused.same_runtime.v1)
    except Exception as e:  # noqa: BLE001
        print(f"[relations_gpu] inductor relation not installed: {e!r}")
    return sorted(rel.ids(implemented=True))
