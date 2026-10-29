"""Synthetic raw-observation events for one decode step of one SmolLM2 layer (plus the sample phase), resolved
against the pinned profile.  The op names / schemas / argument orders are those of vLLM `d9105ea80` as far as they are
known from the design and the previous adapter; `test_resolve_m1.py` checks the same patterns against the real M1 log."""

from __future__ import annotations

import pytest

from verity_vllm.observe.events import (DispatchEnter, DispatchExit, Root, RunHeader, SampleEnd, StepEnd, StepStart,
                                   TritonLaunch, ViewDesc)
from verity_vllm.observe.fold.patterns import (ComputeGroup, GeometryError, TransparentGroup, as_slot_rows, as_weight_view, members,
                                     schema_param_names, split_heads)
from verity_vllm.observe.fold.profiles import vllm_d9105ea80_sm89_eager as PROF
from verity_vllm.observe.fold.patterns.sampling import TokenSelectGumbelTwoStage
from verity_vllm.observe.fold.resolver import (Compute, Gather, GatherMany, PerRow, ResolutionError, Shared, StepContext,
                                     Transparent, Unsupported, resolve_step)
from verity_vllm.observe.fold.tree import step_events

C = PROF.C
H, NH, KVH, D, I, V, QKV, GU = C["H"], C["NH"], C["KVH"], C["D"], C["I"], C["V"], C["QKV"], C["GU"]
EPS = C["EPS"]
BS = 16
NUM_BLOCKS = 64
CUDA = "cuda:0"

# allocation ids
A_EMBED, A_LN1, A_QKV_W, A_O_W, A_LN2, A_GU_W, A_DOWN_W, A_NORM, A_COS_SIN, A_KV = range(100, 110)
A_INPUT_IDS, A_POSITIONS, A_SLOTS, A_BLOCK_TABLE, A_LOGITS_IDX, A_CU_Q, A_SEQUSED = range(200, 207)
A_QKV_B = 110


def bf(alloc, shape, strides=None, offset=0):
    strides = strides or _contig(shape)
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), "bfloat16", CUDA)


def i64(alloc, shape, dtype="int64", device=CUDA):
    return ViewDesc(alloc, 0, tuple(shape), _contig(shape), dtype, device)


def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def gemm_consts(num_sms=128, **over):
    c = dict(PROF.GEMM_CONSTEXPRS)
    c["NUM_SMS"] = num_sms
    c.update(over)
    return c


class Builder:
    """Emits DispatchEnter/Exit and TritonLaunch records with fresh ids."""

    def __init__(self, step: int):
        self.step = step
        self.events: list = []
        self.next_id = 1000 * (step + 1)

    def _id(self):
        self.next_id += 1
        return self.next_id

    def call(self, op, schema, args, kwargs=None, outs=None, parent=None, phase="forward", module=None):
        i = self._id()
        self.events.append(DispatchEnter(i, parent, self.step, phase, op, schema, list(args), dict(kwargs or {}), 7, module))
        return i

    def close(self, i, outs=None):
        self.events.append(DispatchExit(i, outs))

    def leaf(self, op, schema, args, kwargs=None, outs=None, parent=None, phase="forward", module=None):
        i = self.call(op, schema, args, kwargs, outs, parent, phase, module)
        self.close(i, outs)
        return i

    def triton(self, fn, args, constexprs, grid, num_warps, num_stages, parent=None, phase="forward"):
        i = self._id()
        self.events.append(TritonLaunch(i, parent, self.step, phase, fn, "sha", dict(args), dict(constexprs), list(grid),
                                        num_warps, num_stages, 7))
        return i


def weights_roots() -> list[Root]:
    return [
        Root(bf(A_EMBED, (V, H)), "weight", "model.embed_tokens.weight"),
        Root(bf(A_LN1, (H,)), "weight", "model.layers.0.input_layernorm.weight"),
        Root(bf(A_QKV_W, (QKV, H)), "weight", "model.layers.0.self_attn.qkv_proj.weight"),
        Root(bf(A_QKV_B, (QKV,)), "weight", "model.layers.0.self_attn.qkv_proj.bias"),
        Root(bf(A_O_W, (H, NH * D)), "weight", "model.layers.0.self_attn.o_proj.weight"),
        Root(bf(A_LN2, (H,)), "weight", "model.layers.0.post_attention_layernorm.weight"),
        Root(bf(A_GU_W, (GU, H)), "weight", "model.layers.0.mlp.gate_up_proj.weight"),
        Root(bf(A_DOWN_W, (H, I)), "weight", "model.layers.0.mlp.down_proj.weight"),
        Root(bf(A_NORM, (H,)), "weight", "model.norm.weight"),
        Root(bf(A_COS_SIN, (C["MAX_POS"], D)), "buffer", "model.layers.0.self_attn.rotary_emb.cos_sin_cache"),
    ]


GREEDY = {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "logprobs": None, "seed": None}


def header(sampling: dict | None = GREEDY) -> RunHeader:
    workload = {"requests": [{"request_id": "r0", "prompt_token_ids": list(range(32)), "max_tokens": 16}]}
    if sampling is not None:
        workload["sampling"] = sampling
    return RunHeader("verity-capture/log/v1", "2026-09-16T00:00:00Z", "88e1b098d719", {"vllm": "d9105ea80"},
                     {"repo": "HuggingFaceTB/SmolLM2-135M", "config": {"hidden_size": H, "num_hidden_layers": 30}},
                     workload)


def kv_cache_views(layout: str = "kappa"):
    """Per-layer FlashAttention KV cache as key/value `[num_blocks, block_size, KVH, D]` views.  `kappa` is the layout
    of the M1 log: one buffer `[num_blocks, KVH, block_size, 2*D]` viewed as key strides `(bs*KVH*2D, KVH*2D, 2D, 1)`
    with the value view at `+D`; `stacked` is `[2, num_blocks, block_size, KVH, D]` with `unbind(0)` views."""
    if layout == "kappa":
        full = bf(A_KV, (NUM_BLOCKS, KVH, BS, 2 * D))
        strides = (BS * KVH * 2 * D, KVH * 2 * D, 2 * D, 1)
        return full, bf(A_KV, (NUM_BLOCKS, BS, KVH, D), strides, offset=0), bf(A_KV, (NUM_BLOCKS, BS, KVH, D), strides, offset=D)
    full = bf(A_KV, (2, NUM_BLOCKS, BS, KVH, D))
    per = NUM_BLOCKS * BS * KVH * D
    key_cache = bf(A_KV, (NUM_BLOCKS, BS, KVH, D), (BS * KVH * D, KVH * D, D, 1), offset=0)
    value_cache = bf(A_KV, (NUM_BLOCKS, BS, KVH, D), (BS * KVH * D, KVH * D, D, 1), offset=per)
    return full, key_cache, value_cache


def gemm_launch(b: Builder, a_view, w_root_view, c_alloc, K, N, M, phase="forward", parent=None, wrapper=False, **over):
    """`aten.t(weight)` alias then the batch-invariant Triton launch — root-level (Python `matmul_persistent`), or, with
    `wrapper=True`, as the child of an `aten.mm` dispatcher call (the batch-invariant `aten::mm` CUDA override)."""
    wt = ViewDesc(w_root_view.alloc, w_root_view.offset, (K, N), (1, K), "bfloat16", CUDA)
    b.leaf("aten.t.default", "aten::t(Tensor(a) self) -> Tensor(a)", [w_root_view], outs=wt, phase=phase, parent=parent)
    c = bf(c_alloc, (M, N))
    args = {"a_ptr": a_view, "b_ptr": wt, "c_ptr": c, "bias_ptr": None, "M": M, "N": N, "K": K,
            "stride_am": K, "stride_ak": 1, "stride_bk": 1, "stride_bn": K, "stride_cm": N, "stride_cn": 1}
    consts = gemm_consts(**over)
    args.update(consts)
    grid = [min(consts["NUM_SMS"], ((M + 127) // 128) * ((N + 127) // 128))]
    mm = None
    if wrapper:
        mm = b.call("aten.mm.default", "aten::mm(Tensor self, Tensor mat2) -> Tensor", [a_view, wt], phase=phase, parent=parent)
    b.triton("vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", args, consts, grid, 8, 3,
             parent=mm if wrapper else parent, phase=phase)
    if wrapper:
        b.close(mm, c)
    return c


def decode_step(step=1, position=32, token=42, prev_token=7, with_unknown=False, with_dangling=False,
                gemm_over=None, mm_wrapper=False, f32_logits=False, sampler="argmax", kv_layout="kappa",
                metadata_kernel=False, num_sms=128, qkv_bias=None, first_prefill=False) -> tuple[list, StepStart]:
    """One decode step of layer 0 + final norm + sampling, as one request `r0` at absolute position `position`.
    `sampler`: `argmax` (plain `aten.argmax` over the logits) or `gumbel` (the V2 sampler's two-stage
    `_gumbel_sample_kernel` + argmax + gather); `metadata_kernel` adds the model runner's integer-only input-preparation
    Triton launch (phase `other`); `qkv_bias` makes qkv_proj Qwen2's biased linear (`linear_batch_invariant`'s `output + bias`
    after the launch; q, k, v read the sum): "add", or "reread" (a view of the pre-bias output taken after the add as well);
    `first_prefill` adds the attention metadata's host-side `is_prefill.int().argmax(-1)` over a mixed step's two requests."""
    M = 1
    T = position + 1
    block_table = [0, 1, 2]
    slot = block_table[position // BS] * BS + position % BS
    input_ids = ViewDesc(A_INPUT_IDS, 0, (M,), (1,), "int32", CUDA)
    positions = i64(A_POSITIONS, (M,))
    start = StepStart(step, [("r0", position)], [prev_token], [position], {"r0": T}, {"r0": position}, {"r0": block_table},
                      BS, [slot], [("r0", position)], [0], 1, M, M, input_ids, positions, "decode")
    b = Builder(step)
    if metadata_kernel:
        # vllm/v1/worker/gpu/block_table.py: slot mappings from block tables and positions (integer pointers only)
        b.triton("vllm.v1.worker.gpu.block_table._compute_slot_mappings_kernel",
                 {"num_tokens": M, "max_num_reqs": 8, "cu_num_tokens": i64(A_CU_Q, (2,), "int32"), "pos_ptr": positions,
                  "block_tables_ptr": ViewDesc(A_BLOCK_TABLE, 0, (8, 256), (256, 1), "int32", CUDA), "slot_mappings_ptr": i64(A_SLOTS, (M,)),
                  "PAD_ID": -1, "BLOCK_SIZE": BS, "NUM_BLOCKS": 1024},
                 {"BLOCK_SIZE": BS, "NUM_BLOCKS": 1024}, [1, 1], 4, 3, phase="other")
    if first_prefill:
        # vllm/v1/attention/backends/utils.py split_decodes_and_prefills: `is_prefill.int().argmax(dim=-1).item()` on the host
        is_prefill, as_int = i64(320, (2,), "bool", "cpu"), i64(321, (2,), "int32", "cpu")
        b.leaf("aten.to.dtype", "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
               [is_prefill, "torch.int32"], outs=as_int, phase="other")
        b.leaf("aten.argmax.default", "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", [as_int, -1],
               outs=i64(322, (), "int64", "cpu"), phase="other")
    # embedding: input_ids.long() then aten.embedding
    ids64 = i64(300, (M,))
    b.leaf("aten._to_copy.default", "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor",
           [input_ids], {"dtype": {"$repr": "torch.int64"}}, outs=ids64)
    h = bf(301, (M, H))
    b.leaf("aten.embedding.default", "aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
           [bf(A_EMBED, (V, H)), ids64, -1, False, False], outs=h, module="model.embed_tokens")
    # layer 0 input norm: Triton _rms_norm_kernel
    n1 = bf(302, (M, H))
    rms_args = {"input_ptr": h, "weight_ptr": bf(A_LN1, (H,)), "output_ptr": n1, "input_row_stride": H, "output_row_stride": H,
                "n_cols": H, "eps": EPS, "BLOCK_SIZE": 1024, "HAS_WEIGHT": True}
    b.triton("vllm.model_executor.determinism.batch_invariant._rms_norm_kernel", rms_args, {"BLOCK_SIZE": 1024, "HAS_WEIGHT": True}, [M], 4, 3)
    # qkv GEMM
    if qkv_bias:   # `matmul_persistent`'s `c = torch.empty((M, N))`, as the capture records it
        b.leaf("aten.empty.memory_format", "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
               [[M, QKV]], {"dtype": {"$repr": "torch.bfloat16"}}, outs=bf(303, (M, QKV)), module="model.layers.0.self_attn.qkv_proj")
    qkv = gemm_launch(b, n1, bf(A_QKV_W, (QKV, H)), 303, H, QKV, M, wrapper=mm_wrapper, **{"num_sms": num_sms, **(gemm_over or {})})
    qa = 303
    if qkv_bias:
        qa = 315
        b.leaf("aten.add.Tensor", "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", [qkv, bf(A_QKV_B, (QKV,))],
               outs=bf(qa, (M, QKV)), module="model.layers.0.self_attn.qkv_proj")
        if qkv_bias == "reread":
            b.leaf("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [qkv, [-1]], outs=bf(303, (M * QKV,)))
        qkv = bf(qa, (M, QKV))
    q = ViewDesc(qa, 0, (M, NH * D), (QKV, 1), "bfloat16", CUDA)
    k = ViewDesc(qa, NH * D, (M, KVH * D), (QKV, 1), "bfloat16", CUDA)
    v = ViewDesc(qa, (NH + KVH) * D, (M, KVH * D), (QKV, 1), "bfloat16", CUDA)
    b.leaf("aten.split_with_sizes.default", "aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]",
           [qkv, [NH * D, KVH * D, KVH * D], -1], outs=[q, k, v])
    # rope in place on q, k
    b.leaf("_C.rotary_embedding.default", "_C::rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()",
           [positions, q, k, D, bf(A_COS_SIN, (C["MAX_POS"], D)), True], outs=None, module="model.layers.0.self_attn.rotary_emb")
    q3 = ViewDesc(qa, 0, (M, NH, D), (QKV, D, 1), "bfloat16", CUDA)
    k3 = ViewDesc(qa, NH * D, (M, KVH, D), (QKV, D, 1), "bfloat16", CUDA)
    v3 = ViewDesc(qa, (NH + KVH) * D, (M, KVH, D), (QKV, D, 1), "bfloat16", CUDA)
    for src, dst, shp in ((q, q3, [-1, NH, D]), (k, k3, [-1, KVH, D]), (v, v3, [-1, KVH, D])):
        b.leaf("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [src, shp], outs=dst)
    attn_out = bf(304, (M, H))
    b.leaf("aten.zeros.default", "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
           [[M, H]], {"dtype": {"$repr": "torch.bfloat16"}, "device": {"$repr": "cuda:0"}}, outs=attn_out)
    out3 = ViewDesc(304, 0, (M, NH, D), (H, D, 1), "bfloat16", CUDA)
    b.leaf("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [attn_out, [-1, NH, D]], outs=out3)
    full, key_cache, value_cache = kv_cache_views(kv_layout)
    slots = i64(A_SLOTS, (M,))
    k_scale = ViewDesc(400, 0, (), (), "float32", CUDA)
    v_scale = ViewDesc(401, 0, (), (), "float32", CUDA)
    # KV cache update scope
    kvu = b.call("vllm.unified_kv_cache_update.default", "vllm::unified_kv_cache_update(Tensor key, Tensor value, str layer_name) -> ()",
                 [k3, v3, "model.layers.0.self_attn.attn"], module="model.layers.0.self_attn.attn")
    b.leaf("aten.unbind.int", "aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]", [full, 0], outs=[key_cache, value_cache], parent=kvu)
    b.leaf("_C_cache_ops.reshape_and_cache_flash.default",
           "_C_cache_ops::reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale) -> ()",
           [k3, v3, key_cache, value_cache, slots, "auto", k_scale, v_scale], parent=kvu)
    b.close(kvu, None)
    # attention scope
    att = b.call("vllm.unified_attention_with_output.default",
                 "vllm::unified_attention_with_output(Tensor query, Tensor key, Tensor value, Tensor! output, str layer_name, Tensor? output_scale=None, Tensor? output_block_scale=None) -> ()",
                 [q3, k3, v3, out3, "model.layers.0.self_attn.attn", None, None], module="model.layers.0.self_attn.attn")
    b.leaf("aten.unbind.int", "aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]", [full, 0], outs=[key_cache, value_cache], parent=att)
    b.leaf("aten.slice.Tensor", "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)",
           [q3, 0, 0, M], outs=q3, parent=att)
    b.leaf("aten.slice.Tensor", "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)",
           [out3, 0, 0, M], outs=out3, parent=att)
    lse = ViewDesc(402, 0, (NH, M), (M, 1), "float32", CUDA)
    b.leaf("_vllm_fa2_C.varlen_fwd.default",
           "_vllm_fa2_C::varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, Generator? gen, int num_splits) -> Tensor[]",
           [q3, key_cache, value_cache, out3, ViewDesc(A_CU_Q, 0, (2,), (1,), "int32", CUDA), None, ViewDesc(A_SEQUSED, 0, (1,), (1,), "int32", CUDA), None,
            ViewDesc(A_BLOCK_TABLE, 0, (1, 3), (3, 1), "int32", CUDA), None, 1, T, 0.0, D ** -0.5, False, True, -1, -1, 0.0, False, None, 1],
           outs=[out3, lse], parent=att)
    b.close(att, None)
    a2 = bf(304, (M, H))
    b.leaf("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [out3, [-1, H]], outs=a2)
    # o_proj, fused norm, gate_up, silu, down, final fused norm
    o = gemm_launch(b, a2, bf(A_O_W, (H, NH * D)), 305, NH * D, H, M, num_sms=num_sms)
    b.leaf("_C.fused_add_rms_norm.default", "_C::fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()",
           [o, h, bf(A_LN2, (H,)), EPS], module="model.layers.0.post_attention_layernorm")
    gu = gemm_launch(b, o, bf(A_GU_W, (GU, H)), 306, H, GU, M, num_sms=num_sms)
    act = bf(307, (M, I))
    b.leaf("aten.empty.memory_format", "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
           [[M, I]], {"dtype": {"$repr": "torch.bfloat16"}}, outs=act)
    b.leaf("_C.silu_and_mul.default", "_C::silu_and_mul(Tensor! result, Tensor input) -> ()", [act, gu], module="model.layers.0.mlp.act_fn")
    d = gemm_launch(b, act, bf(A_DOWN_W, (H, I)), 308, I, H, M, num_sms=num_sms)
    b.leaf("_C.fused_add_rms_norm.default", "_C::fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()",
           [d, h, bf(A_NORM, (H,)), EPS], module="model.norm")
    if with_unknown:
        b.leaf("vllm.some_unknown_op.default", "vllm::some_unknown_op(Tensor x) -> Tensor", [d], outs=bf(499, (M, H)))
    if with_dangling:
        vw = b.call("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [d, [M, H]], outs=d)
        b.triton("some.module.mystery_kernel", {"x": d}, {}, [1], 4, 1, parent=vw)
        b.close(vw, d)
    # sample phase: hidden[logits_indices], lm_head GEMM, argmax, int copies
    hid = bf(309, (1, H))
    b.leaf("aten.index.Tensor", "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor", [d, [i64(A_LOGITS_IDX, (1,))]], outs=hid, phase="sample")
    logits = gemm_launch(b, hid, bf(A_EMBED, (V, H)), 310, H, V, 1, phase="sample", num_sms=num_sms)
    if f32_logits:
        # the sampler's `logits.to(torch.float32)` before the greedy argmax
        l32 = ViewDesc(314, 0, (1, V), (V, 1), "float32", CUDA)
        b.leaf("aten._to_copy.default", "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor",
               [logits], {"dtype": "torch.float32"}, outs=l32, phase="sample")
        logits = l32
    if sampler == "gumbel":
        # vllm/v1/worker/gpu/sample/gumbel.py: blockwise (max, argmax) kernel, argmax over the block maxima, gather
        NB = -(-V // 1024)
        local_argmax = i64(315, (1, NB))
        local_max = ViewDesc(316, 0, (1, NB), (NB, 1), "float32", CUDA)
        b.leaf("aten.new_empty.default", "aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
               [logits, [1, NB]], {"dtype": {"$repr": "torch.int64"}}, outs=local_argmax, phase="sample")
        b.leaf("aten.new_empty.default", "aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
               [logits, [1, NB]], {"dtype": {"$repr": "torch.float32"}}, outs=local_max, phase="sample")
        consts = {"BLOCK_SIZE": 1024, "IS_DRAFTING": False, "APPLY_TEMPERATURE": False, "USE_FP64": False, "PER_TOKEN_COL": False}
        b.triton("vllm.v1.worker.gpu.sample.gumbel._gumbel_sample_kernel",
                 {"local_argmax_ptr": local_argmax, "local_argmax_stride": NB, "local_max_ptr": local_max, "local_max_stride": NB,
                  "logits_cache_ptr": None, "logits_cache_stride": 0, "logits_ptr": logits, "logits_stride": V,
                  "expanded_idx_mapping_ptr": i64(A_LOGITS_IDX, (1,), "int32"), "seeds_ptr": i64(A_LOGITS_IDX, (8,)),
                  "pos_ptr": positions, "temp_ptr": ViewDesc(317, 0, (8,), (1,), "float32", CUDA), "vocab_size": V, **consts},
                 consts, [1, NB], 4, 3, phase="sample")
        idx = i64(318, (1, 1))
        b.leaf("aten.argmax.default", "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", [local_max, -1, True], outs=idx, phase="sample")
        sampled2 = i64(311, (1, 1))
        b.leaf("aten.gather.default", "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor",
               [local_argmax, -1, idx], outs=sampled2, phase="sample")
        sampled = i64(311, (1,))
        b.leaf("aten.view.default", "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", [sampled2, [-1]], outs=sampled, phase="sample")
    else:
        sampled = i64(311, (1,))
        b.leaf("aten.argmax.default", "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", [logits, -1, False], outs=sampled, phase="sample")
    s32 = i64(312, (1,), "int32")
    b.leaf("aten._to_copy.default", "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor",
           [sampled], {"dtype": {"$repr": "torch.int32"}}, outs=s32, phase="sample")
    cpu = i64(313, (1,), "int32", "cpu")
    b.leaf("aten.copy_.default", "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)", [cpu, s32, True], outs=cpu, phase="sample")
    events = [start, *b.events, SampleEnd(step, logits, [token], s32), StepEnd(step)]
    return events, start


def context(start: StepStart, sample=None, sampling=GREEDY) -> StepContext:
    roots: dict[int, list[Root]] = {}
    for r in weights_roots():
        roots.setdefault(r.view.alloc, []).append(r)
    return StepContext(header(sampling), start, sample, [], roots, C, PROF.PROFILE)


def by_spec(resolutions):
    out: dict[str, list[Compute]] = {}
    for r in resolutions:
        for m in members(r):
            if isinstance(m, Compute):
                out.setdefault(m.spec.id, []).append(m)
    return out


# ---------------------------------------------------------------------------------------------------------


def test_decode_step_resolves_every_root():
    events, start = decode_step()
    se = step_events(events)
    ctx = context(start, se.sample)
    res = resolve_step(se, ctx)
    assert len(res) == len(se.roots)
    assert not [r for r in res if isinstance(r, Unsupported)], [r for r in res if isinstance(r, Unsupported)]
    specs = by_spec(res)
    expected = {
        "Embedding_v1{V=49152,H=576}": 1, "RMSNormTriton_v1{N=576,EPS=1e-05}": 1,
        "Gemm_v1{K=576,N=960}": 1, "Gemm_v1{K=576,N=576}": 1, "Gemm_v1{K=576,N=3072}": 1, "Gemm_v1{K=1536,N=576}": 1,
        "Gemm_v1{K=576,N=49152}": 1, "RoPE_v1{NHEADS=9,D=64}": 1, "RoPE_v1{NHEADS=3,D=64}": 1,
        "Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}": 1, "RMSNormFusedCuda_v2{N=576,EPS=1e-05}": 2,
        "SiluMul_v2{I=1536}": 1, "TokenSelect_v1{V=49152}": 1,
    }
    assert {k: len(v) for k, v in specs.items()} == expected
    assert all(c.rows == 1 for cs in specs.values() for c in cs)
    # every event id of the step is consumed exactly once by a primary resolution
    consumed = [i for r in res for i in r.consumed]
    assert sorted(consumed) == sorted(i for root in se.roots for i in root.ids())


def test_gemm_operands_and_pins():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    g = by_spec(res)["Gemm_v1{K=576,N=960}"][0]
    a, w = g.operands
    assert isinstance(a, PerRow) and a.view.shape == (1, H) and a.view.alloc == 302
    assert isinstance(w, Shared) and w.view == ViewDesc(A_QKV_W, 0, (QKV, H), (H, 1), "bfloat16", CUDA)
    assert g.outputs == (PerRow(ViewDesc(303, 0, (1, QKV), (QKV, 1), "bfloat16", CUDA)),)
    assert g.kernels == ("matmul_kernel_persistent",) and g.pattern == "gemm.matmul_kernel_persistent"
    assert g.attribution == (("r0", 32),)
    assert "qkv_proj" in g.hint
    lm = by_spec(res)["Gemm_v1{K=576,N=49152}"][0]
    assert lm.operands[1].view.alloc == A_EMBED and lm.operands[1].view.shape == (V, H)
    assert lm.attribution == (("r0", 32),)


def test_gemm_constexpr_mismatch_is_unsupported():
    events, start = decode_step(gemm_over={"BLOCK_SIZE_K": 32})
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and "pinned" in uns[0].reason and uns[0].detail["mismatches"]["BLOCK_SIZE_K"] == {"observed": 32, "pinned": 64}


def _context_on_host(start, sample, sms, cc=(8, 9)):
    """A StepContext whose run header carries the capture host's GPU document (`profile.gpu.multi_processor_count`), as
    `verity_vllm.engine.engine_profile.gpu_info` records it; `sms=None` = no GPU document (synthetic log)."""
    ctx = context(start, sample)
    prof = dict(ctx.header.profile)
    if sms is not None:
        prof["gpu"] = {"name": "NVIDIA L40S" if tuple(cc) == (8, 9) else f"cc {cc[0]}.{cc[1]}", "compute_capability": list(cc),
                       "multi_processor_count": sms}
    hdr = RunHeader(ctx.header.schema, ctx.header.created_utc, ctx.header.profile_id, prof, ctx.header.model, ctx.header.workload, ctx.header.notes)
    return StepContext(hdr, ctx.start, ctx.sample, ctx.links, ctx.roots, ctx.config, ctx.profile)


def test_gemm_num_sms_is_pinned_to_the_record_target_and_recorded():
    """[X-01] The NUM_SMS pin is the record target's SM count — the device the capture host recorded in the header — so a launch
    showing 142 on a 142-SM host resolves like the profile's own launches on a 128-SM host; the pin choice is counted in the step
    scratch the fold / resolve_log report from.  The Compute is unchanged (digest semantics)."""
    from verity_vllm.observe.fold.patterns import X01_SCRATCH_KEY
    events, start = decode_step(num_sms=142)   # every persistent-GEMM launch of the step ran with the 142-CTA grid
    se = step_events(events)
    ctx = _context_on_host(start, se.sample, 142)
    res = resolve_step(se, ctx)
    assert not [r for r in res if isinstance(r, Unsupported)]
    g = by_spec(res)["Gemm_v1{K=576,N=960}"][0]
    assert g.kernels == ("matmul_kernel_persistent",) and g.pattern == "gemm.matmul_kernel_persistent"
    rec = ctx.scratch[X01_SCRATCH_KEY]
    assert rec["pinned"] == 142 and rec["profile_default"] == 128 and rec["launches"] >= 1 and "from_gpu_document" in rec["source"]


@pytest.mark.parametrize("host_sms,launch_sms", [(None, 142), (128, 142), (108, 142), (142, 128)])
def test_gemm_num_sms_other_than_the_target_is_unsupported(host_sms, launch_sms):
    """[X-01] Exact, not widened: the pin is the target's SM count (the profile default 128 when the header has no device).  A launch
    for another GPU — 142 on a 128- or 108-SM host or in a device-less log, AND the profile's own 128 on a 142-SM host — is refused."""
    events, start = decode_step(gemm_over={"NUM_SMS": launch_sms})
    se = step_events(events)
    ctx = _context_on_host(start, se.sample, host_sms)
    res = resolve_step(se, ctx)
    uns = [r for r in res if isinstance(r, Unsupported)]
    pinned = host_sms if host_sms is not None else 128
    # `gemm_over` rewrites one launch; the step's other GEMM launches keep the profile's 128 — every launch is judged against the
    # same target pin, so exactly those whose NUM_SMS != pinned are refused, each naming (observed, pinned)
    assert uns and all(set(u.detail["mismatches"]) == {"NUM_SMS"} and u.detail["mismatches"]["NUM_SMS"]["pinned"] == pinned for u in uns)
    assert {u.detail["mismatches"]["NUM_SMS"]["observed"] for u in uns} == ({launch_sms, 128} - {pinned})
    assert len(uns) == (1 if pinned == 128 else 5)


def test_gemm_target_profile_is_the_same_object_the_builder_pins():
    """[X-01 portability] One TargetProfile fact: the eager resolver's pin for a record captured on a 142-SM device equals
    `TargetProfile.from_gpu_document(header.gpu).num_sms`, which is what `TargetProfile.from_device()` yields on that host and what
    Bet A's `gemm_pins_for` pins (`NUM_SMS == active TargetProfile.num_sms`, see tests/program/test_vllm_bindings_pins.py)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.observe.fold.patterns import target_of
    events, start = decode_step(num_sms=142)
    se = step_events(events)
    ctx = _context_on_host(start, se.sample, 142)
    tp, recorded = target_of(ctx)
    assert recorded and tp == TargetProfile(compute_capability=(8, 9), num_sms=142)
    assert TargetProfile.from_gpu_document({"name": "x", "cc": "8.9", "sms": 142}) == tp
    assert TargetProfile.from_gpu_document({"name": "x"}) is None and TargetProfile.from_gpu_document(None) is None
    assert target_of(context(start, se.sample)) == (TargetProfile.default(), False)


def test_the_biased_linear_resolves_to_one_gemm_bias_at_the_add():
    """Qwen2's qkv_proj: the launch writes the pre-bias [M, QKV] buffer, `aten.add(c, bias)` its only reader.  The step resolves to a
    no-op at the launch and ONE `GemmBias_v1{K,N}(n1, W, b)` at the add (both weights named), the frontend's binding of the pair;
    every other GEMM of the step stays `Gemm_v1`."""
    events, start = decode_step(qkv_bias="add")
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    specs = by_spec(res)
    (gb,) = specs[f"GemmBias_v1{{K={H},N={QKV}}}"]
    assert gb.operands == (PerRow(bf(302, (1, H))), Shared(ViewDesc(A_QKV_W, 0, (QKV, H), (H, 1), "bfloat16", CUDA)), Shared(bf(A_QKV_B, (QKV,))))
    assert gb.outputs == (PerRow(bf(315, (1, QKV))),) and gb.pattern == "gemm_bias.fused"
    assert gb.roots == (None, "model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.bias")
    assert not any(s.startswith("BiasAdd_v1") for s in specs) and f"Gemm_v1{{K={H},N={QKV}}}" not in specs
    launch = next(r for r in res if getattr(r, "pattern", "") == "gemm_bias.launch")
    assert isinstance(launch, Transparent) and launch.effect == "noop"
    plain, _ = decode_step()
    unbiased = by_spec(resolve_step(step_events(plain), context(start)))
    assert {s: len(v) for s, v in specs.items() if s.startswith("Gemm_v1")} == \
        {s: len(v) for s, v in unbiased.items() if s.startswith("Gemm_v1") and s != f"Gemm_v1{{K={H},N={QKV}}}"}


@pytest.mark.parametrize("cc,sms", [((12, 0), 188), ((9, 0), 132)])
def test_on_a_gemm_v2_host_the_biased_linear_resolves_as_the_frontend_binds_it(cc, sms):
    """The same pair on a `Gemm_v2` target (cc 12.0, cc 9.0): one `GemmBias_v2{K,N,DOT}` at the add, the record target's one-Call
    Definition (`frontend.targets.gemm_bias_spec`), the frontend's choice."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend import targets as TG
    events, start = decode_step(qkv_bias="add", num_sms=sms)
    se = step_events(events)
    specs = by_spec(resolve_step(se, _context_on_host(start, se.sample, sms, cc)))
    tp = TargetProfile(compute_capability=cc, num_sms=sms)
    gemm, fused = TG.gemm_spec(tp, H, QKV).id, TG.gemm_bias_spec(tp, H, QKV)
    assert fused is not None and fused.id.startswith("GemmBias_v2{") and len(specs[fused.id]) == 1, sorted(specs)
    assert gemm not in specs and not any(s.startswith("BiasAdd_v1") for s in specs), sorted(specs)


def test_a_biased_linear_whose_pre_bias_output_is_read_again_keeps_gemm_and_bias_add():
    events, start = decode_step(qkv_bias="reread")
    se = step_events(events)
    specs = by_spec(resolve_step(se, context(start, se.sample)))
    assert f"Gemm_v1{{K={H},N={QKV}}}" in specs and f"BiasAdd_v1{{N={QKV}}}" in specs and not any(s.startswith("GemmBias") for s in specs)


def test_gemm_bias_path_is_unsupported():
    events, start = decode_step(gemm_over={"HAS_BIAS": True})
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert uns and "pinned" in uns[0].reason  # HAS_BIAS is pinned False


def test_rmsnorm_and_fused_norm():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    specs = by_spec(res)
    t = specs["RMSNormTriton_v1{N=576,EPS=1e-05}"][0]
    assert t.operands == (PerRow(bf(301, (1, H))), Shared(bf(A_LN1, (H,)))) and t.outputs == (PerRow(bf(302, (1, H))),)
    f1, f2 = specs["RMSNormFusedCuda_v2{N=576,EPS=1e-05}"]
    assert f1.operands == (PerRow(bf(305, (1, H))), PerRow(bf(301, (1, H))), Shared(bf(A_LN2, (H,))))
    assert f1.outputs == (PerRow(bf(305, (1, H))), PerRow(bf(301, (1, H))))
    assert f2.operands[2] == Shared(bf(A_NORM, (H,)))
    assert f1.kernels == ("fused_add_rms_norm_kernel",)


def test_registered_values_bind_the_norms_and_attention_that_read_them_as_inputs():
    """The accepted profile under `registered_values`: the norms are `RMSNormTriton_v3` / `RMSNormFusedCuda_v4` and attention is
    `Attention_v10`, each with its value as a last `Supplied` operand of the registered `values`; operands before it are unchanged."""
    import dataclasses

    import numpy as np

    from verity_vllm.observe.fold.resolver import Supplied
    from verity.ml.definitions.registry import b1
    from verity.ml.definitions.registry import prims as P

    events, start = decode_step()
    se = step_events(events)
    base = context(start, se.sample)
    ctx = dataclasses.replace(base, profile=dataclasses.replace(base.profile, registered_values=True))
    specs = by_spec(resolve_step(se, ctx))
    eps = Supplied("registration", "eps", 32, int(P.bits_of(np.float32(C["EPS"]))))
    t = specs["RMSNormTriton_v3{N=576}"][0]
    assert t.operands == (PerRow(bf(301, (1, H))), Shared(bf(A_LN1, (H,))), eps)
    f1, _f2 = specs["RMSNormFusedCuda_v4{N=576}"]
    assert f1.operands == (PerRow(bf(305, (1, H))), PerRow(bf(301, (1, H))), Shared(bf(A_LN2, (H,))), eps)
    (a,) = specs["Attention_v10{T=33,NH=9,KVH=3,D=64,BN=128}"]
    assert a.operands[3] == Supplied("registration", "scale_log2_d64", 32, b1.scale_log2_bits(D)) and len(a.operands) == 4
    assert not [k for k in specs if k.startswith(("RMSNormTriton_v1", "RMSNormFusedCuda_v2", "Attention_v3"))]


def test_registered_values_on_the_rtx_pro_6000_bind_fa2s_check_inf_chain_with_the_scale_an_input():
    """The same step on the RTX PRO 6000 host (cc 12.0, FA2's per-iteration Check_inf): attention is `Attention_v11`, the frontend's
    choice there (`targets.attention_spec(..., scale_input=True)`), with the scale its last `Supplied` operand; the norms as above."""
    import dataclasses

    from verity_vllm.observe.fold.resolver import Supplied
    from verity.ml.definitions.registry import b1

    events, start = decode_step(num_sms=188)
    se = step_events(events)
    base = _context_on_host(start, se.sample, 188, (12, 0))
    specs = by_spec(resolve_step(se, dataclasses.replace(base, profile=dataclasses.replace(base.profile, registered_values=True))))
    (a,) = [r for k, v in specs.items() if k.startswith("Attention_v11{T=33,NH=9,KVH=3,D=64,") for r in v]
    assert a.operands[3] == Supplied("registration", "scale_log2_d64", 32, b1.scale_log2_bits(D)) and len(a.operands) == 4
    assert "RMSNormTriton_v3{N=576}" in specs and "RMSNormFusedCuda_v4{N=576}" in specs
    assert not [k for k in specs if k.startswith(("RMSNormTriton_v1", "RMSNormFusedCuda_v2", "Attention_v3", "Attention_v5"))]


def test_rope_group_q_and_k():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    rope = [r for r in res if isinstance(r, Compute) and r.spec.id.startswith("RoPE")]
    assert len(rope) == 1 and isinstance(rope[0], ComputeGroup)
    q, (k,) = rope[0], rope[0].also
    assert q.spec.id == "RoPE_v1{NHEADS=9,D=64}" and k.spec.id == "RoPE_v1{NHEADS=3,D=64}"
    assert q.operands[0] == PerRow(ViewDesc(303, 0, (1, NH, D), (QKV, D, 1), "bfloat16", CUDA))
    assert k.operands[0] == PerRow(ViewDesc(303, NH * D, (1, KVH, D), (QKV, D, 1), "bfloat16", CUDA))
    assert q.operands[1] == Gather(bf(A_COS_SIN, (C["MAX_POS"], D)), (32,)) == k.operands[1]
    assert q.outputs == (q.operands[0],) and k.outputs == (k.operands[0],)
    assert k.consumed == () and len(q.consumed) == 1
    assert q.pattern == k.pattern == "rope.rotary_embedding"


def test_kv_update_scope_moves_k_and_v():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    kv = [r for r in res if isinstance(r, TransparentGroup)]
    assert len(kv) == 1
    k, (v,) = kv[0], kv[0].also
    _, key_cache, value_cache = kv_cache_views()
    kcs, vcs = as_slot_rows(key_cache), as_slot_rows(value_cache)
    # κ layout: slot rows every KVH*2D elements, heads every 2D, the value view at +D
    assert kcs.shape == (NUM_BLOCKS * BS, KVH, D) and kcs.strides == (KVH * 2 * D, 2 * D, 1)
    assert k.effect == v.effect == "move" and k.rows == v.rows == 1
    assert k.src == PerRow(ViewDesc(303, NH * D, (1, KVH, D), (QKV, D, 1), "bfloat16", CUDA)) and k.dst == Gather(kcs, (32,))
    assert v.src == PerRow(ViewDesc(303, (NH + KVH) * D, (1, KVH, D), (QKV, D, 1), "bfloat16", CUDA)) and v.dst == Gather(vcs, (32,))
    assert vcs.offset == D and kcs.offset == 0
    node = next(r for r in se.roots if r.op == "vllm.unified_kv_cache_update.default")
    assert set(k.consumed) == node.ids() and len(node.ids()) == 3     # outer + unbind + reshape_and_cache_flash
    assert kv[0].before == ()                                          # unbind is an alias: consumed, nothing to execute


def test_attention_scope_decode():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    att = [r for r in res if isinstance(r, Compute) and r.spec.id.startswith("Attention")]
    assert len(att) == 1 and isinstance(att[0], ComputeGroup) and att[0].also == () and att[0].before == ()
    a = att[0]
    assert a.spec.id == "Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}" and a.rows == 1
    q, kc, vc = a.operands
    _, key_cache, value_cache = kv_cache_views()
    assert q == PerRow(ViewDesc(303, 0, (1, NH, D), (QKV, D, 1), "bfloat16", CUDA))
    assert kc == GatherMany(as_slot_rows(key_cache), (tuple(range(33)),))
    assert vc == GatherMany(as_slot_rows(value_cache), (tuple(range(33)),))
    assert a.outputs == (PerRow(ViewDesc(304, 0, (1, NH, D), (H, D, 1), "bfloat16", CUDA)),)
    node = next(r for r in se.roots if r.op == "vllm.unified_attention_with_output.default")
    assert set(a.consumed) == node.ids() and len(node.ids()) == 5
    assert a.kernels == ("flash_fwd_splitkv_kernel",) and a.attribution == (("r0", 32),)


def test_attention_groups_prefill_rows_by_T():
    """Four prompt rows at positions 0..3 (block table [5]): one Attention_v3{T} per row, slots from the block table."""
    M = 4
    q3 = ViewDesc(303, 0, (M, NH, D), (QKV, D, 1), "bfloat16", CUDA)
    k3 = ViewDesc(303, NH * D, (M, KVH, D), (QKV, D, 1), "bfloat16", CUDA)
    v3 = ViewDesc(303, (NH + KVH) * D, (M, KVH, D), (QKV, D, 1), "bfloat16", CUDA)
    out3 = ViewDesc(304, 0, (M, NH, D), (H, D, 1), "bfloat16", CUDA)
    start = StepStart(0, [("r0", p) for p in range(M)], [1, 2, 3, 4], list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [5]}, BS,
                      [5 * BS + p for p in range(M)], [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")
    b = Builder(0)
    full, key_cache, value_cache = kv_cache_views()
    att = b.call("vllm.unified_attention_with_output.default",
                 "vllm::unified_attention_with_output(Tensor query, Tensor key, Tensor value, Tensor! output, str layer_name, Tensor? output_scale=None, Tensor? output_block_scale=None) -> ()",
                 [q3, k3, v3, out3, "model.layers.0.self_attn.attn", None, None])
    b.leaf("_vllm_fa2_C.varlen_fwd.default",
           "_vllm_fa2_C::varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, Generator? gen, int num_splits) -> Tensor[]",
           [q3, key_cache, value_cache, out3, i64(A_CU_Q, (2,), "int32"), None, i64(A_SEQUSED, (1,), "int32"), None,
            ViewDesc(A_BLOCK_TABLE, 0, (1, 1), (1, 1), "int32", CUDA), None, M, M, 0.0, D ** -0.5, False, True, -1, -1, 0.0, False, None, 1],
           outs=[out3, ViewDesc(402, 0, (NH, M), (M, 1), "float32", CUDA)], parent=att)
    b.close(att, None)
    se = step_events([start, *b.events, StepEnd(0)])
    res = resolve_step(se, context(start))
    assert len(res) == 1 and isinstance(res[0], ComputeGroup)
    groups = members(res[0])
    assert [g.spec.id for g in groups] == [f"Attention_v3{{T={t},NH=9,KVH=3,D=64,BN=128}}" for t in range(1, M + 1)]
    for i, g in enumerate(groups):
        assert g.rows == 1 and g.attribution == (("r0", i),)
        assert g.operands[0] == PerRow(ViewDesc(303, i * QKV, (1, NH, D), (QKV, D, 1), "bfloat16", CUDA))
        assert g.operands[1] == GatherMany(as_slot_rows(key_cache), (tuple(5 * BS + t for t in range(i + 1)),))
        assert g.outputs == (PerRow(ViewDesc(304, i * H, (1, NH, D), (H, D, 1), "bfloat16", CUDA)),)
    assert set(res[0].consumed) == se.roots[0].ids() and all(g.consumed == () for g in groups[1:])


def test_attention_flags_must_match_the_registered_body():
    events, start = decode_step()
    # flip causal to False in the FA2 op
    for i, e in enumerate(events):
        if isinstance(e, DispatchEnter) and e.op == "_vllm_fa2_C.varlen_fwd.default":
            args = list(e.args)
            args[15] = False
            events[i] = DispatchEnter(e.id, e.parent, e.step, e.phase, e.op, e.schema, args, e.kwargs, e.stream, e.module)
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == "vllm.unified_attention_with_output.default" and "causal" in uns[0].reason
    assert set(uns[0].consumed) == next(r for r in se.roots if r.op == uns[0].op).ids()


def test_silu_embedding_tokenselect_gather():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    specs = by_spec(res)
    s = specs["SiluMul_v2{I=1536}"][0]
    assert s.operands == (PerRow(bf(306, (1, GU))),) and s.outputs == (PerRow(bf(307, (1, I))),)
    e = specs["Embedding_v1{V=49152,H=576}"][0]
    # the token operand is the step's input_ids view (int32), not the int64 copy vLLM made of it
    assert e.operands == (PerRow(ViewDesc(A_INPUT_IDS, 0, (1,), (1,), "int32", CUDA)), Shared(bf(A_EMBED, (V, H))))
    assert e.outputs == (PerRow(bf(301, (1, H))),) and e.attribution == (("r0", 32),)
    ts = specs["TokenSelect_v1{V=49152}"][0]
    assert ts.operands == (PerRow(bf(310, (1, V))),) and ts.outputs == (PerRow(i64(311, (1,))),) and ts.attribution == (("r0", 32),)
    gathers = [r for r in res if isinstance(r, Transparent) and r.pattern == "sample.logits_gather"]
    assert len(gathers) == 1 and gathers[0].effect == "move"
    assert gathers[0].src == Gather(bf(308, (1, H)), (0,)) and gathers[0].dst == PerRow(bf(309, (1, H)))


@pytest.mark.parametrize("numerics", ["b1-eager-v3", "b1-eager"])
def test_silu_is_the_builders_silu_mul_kind_under_the_record_targets_numerics_profile(numerics):
    """`_C.silu_and_mul` folds to the Definition the record target's numerics vocabulary binds `silu_mul` to, the kind the builder's
    `SiluAndMulRule` asks for: `SiluMul_v2` under b1-eager-v3, `SiluMul_v1` under the older vocabularies."""
    import dataclasses

    from verity_vllm.program.frontend.rules.vocab import vocabulary_for
    from verity_vllm.program.frontend.target_profile import TargetProfile

    events, start = decode_step()
    se = step_events(events)
    ctx = context(start, se.sample)
    doc = {**ctx.header.profile, "target": TargetProfile(numerics_profile=numerics).to_json()}
    ctx = dataclasses.replace(ctx, header=dataclasses.replace(ctx.header, profile=doc))
    (s,) = [c for k, v in by_spec(resolve_step(se, ctx)).items() if k.startswith("SiluMul") for c in v]
    assert s.spec is vocabulary_for("profile", numerics).kind("silu_mul", I=I)
    assert s.spec.base_id == {"b1-eager-v3": "SiluMul_v2", "b1-eager": "SiluMul_v1"}[numerics]


def test_transparent_set_alias_alloc_noop():
    events, start = decode_step()
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    by_op = {root.op: r for root, r in zip(se.roots, res)}
    assert by_op["aten.split_with_sizes.default"].effect == "alias"
    assert by_op["aten.t.default"].effect == "alias"
    assert by_op["aten.zeros.default"].effect == "alloc" and by_op["aten.zeros.default"].dst == Shared(bf(304, (1, H)))
    assert by_op["aten.empty.memory_format"].effect == "alloc"
    # integer copies (input_ids.long(), sampled ids -> int32, D2H) are metadata noops
    copies = [r for root, r in zip(se.roots, res) if root.op in ("aten._to_copy.default", "aten.copy_.default")]
    assert len(copies) == 3 and all(r.effect == "noop" for r in copies)
    assert all(r.pattern == "transparent.aten" for r in copies)


def test_gemm_under_aten_mm_wrapper():
    """The batch-invariant `aten::mm` CUDA override launches the kernel from inside the dispatcher call: the launch is
    the wrapper's child and the wrapper is consumed as scope."""
    events, start = decode_step(mm_wrapper=True)
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    node = next(r for r in se.roots if r.op == "aten.mm.default")
    g = res[se.roots.index(node)]
    assert isinstance(g, Compute) and g.spec.id == "Gemm_v1{K=576,N=960}" and set(g.consumed) == node.ids() and len(node.ids()) == 2
    assert g.operands[1] == Shared(bf(A_QKV_W, (QKV, H))) and g.outputs == (PerRow(bf(303, (1, QKV))),)
    assert not isinstance(g, ComputeGroup) or g.before == ()


def test_f32_logits_upcast_is_repointed_to_bf16_source():
    events, start = decode_step(f32_logits=True)
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    ts = by_spec(res)["TokenSelect_v1{V=49152}"][0]
    assert ts.operands == (PerRow(bf(310, (1, V))),)          # the bf16 logits, not the float32 copy
    up = [r for root, r in zip(se.roots, res) if root.op == "aten._to_copy.default" and root.phase == "sample"]
    assert any(isinstance(r, Transparent) and r.effect == "noop" and "widening" in r.hint for r in up)


def test_gumbel_two_stage_sampler_is_one_token_select():
    """The V2 sampler's greedy path (κ, M1 log): the blockwise kernel and the argmax over block maxima are noops; the
    gather of the block argmax is the TokenSelect over the bf16 logits, with the sampled [R,1] view as output."""
    events, start = decode_step(sampler="gumbel")
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)], [r for r in res if isinstance(r, Unsupported)]
    ts = by_spec(res)["TokenSelect_v1{V=49152}"]
    assert len(ts) == 1
    assert ts[0].operands == (PerRow(bf(310, (1, V))),) and ts[0].outputs == (PerRow(i64(311, (1, 1))),)
    assert ts[0].pattern == "sample.gumbel_two_stage" and ts[0].attribution == (("r0", 32),)
    by_op = {root.op: r for root, r in zip(se.roots, res)}
    assert by_op["vllm.v1.worker.gpu.sample.gumbel._gumbel_sample_kernel"].effect == "noop"
    assert by_op["aten.argmax.default"].effect == "noop"
    assert isinstance(by_op["aten.gather.default"], Compute)
    stages = [r for r in res if getattr(r, "pattern", None) == "sample.gumbel_two_stage"]
    assert len(stages) == 3
    # a non-greedy header makes the sampler Unsupported (the kernel's temperature branch is a workload fact).  The stochastic pattern owns
    # the kernel under it (the profile lists it first) and refuses the greedy chain's bf16 logits, which no float32 widening precedes; the
    # greedy pattern refuses the header by itself
    stochastic = {"greedy": False, "temperature": 0.7}
    se2 = step_events(events)
    res2 = resolve_step(se2, context(start, None, sampling=stochastic))
    uns = [r for r in res2 if isinstance(r, Unsupported)]
    assert uns and uns[0].op.endswith("_gumbel_sample_kernel") and uns[0].reason == "sampler logits are not a row-contiguous float32 [R,V] view"
    assert "TokenSelect_v1{V=49152}" not in by_spec(res2)
    kernel = next(n for n in se2.roots if n.op.endswith("_gumbel_sample_kernel"))
    alone = TokenSelectGumbelTwoStage(vocab=V, block_size=PROF.SAMPLER_BLOCK_SIZE).match(kernel, context(start, None, sampling=stochastic))
    assert isinstance(alone, Unsupported) and "outside greedy sampling" in alone.reason


def test_integer_metadata_triton_kernel_is_noop():
    events, start = decode_step(metadata_kernel=True)
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    r = res[0]
    assert se.roots[0].op.endswith("_compute_slot_mappings_kernel") and se.roots[0].phase == "other"
    assert isinstance(r, Transparent) and r.effect == "noop" and r.pattern == "transparent.triton_metadata"
    assert r.kernels == ("_compute_slot_mappings_kernel",)


def test_a_mixed_steps_first_prefill_argmax_is_integer_metadata():
    """[GLM-5.3 B8] vLLM's `split_decodes_and_prefills` finds a mixed step's first prefill request as an argmax over the host's
    int32 `is_prefill` (its width the step's requests): integer step metadata (noop) under either sampler, never a TokenSelect."""
    events, start = decode_step(first_prefill=True)
    for sampling in (GREEDY, {"greedy": False, "temperature": 0.8, "top_p": 0.95}):
        se = step_events(events)
        res = resolve_step(se, context(start, se.sample, sampling=sampling))
        meta = [r for root, r in zip(se.roots, res) if root.phase == "other"]
        assert [r.effect for r in meta] == ["noop", "noop"] and {r.pattern for r in meta} == {"transparent.aten"}, meta
        if sampling is GREEDY:
            assert not [r for r in res if isinstance(r, Unsupported)]
            assert len(by_spec(res)["TokenSelect_v1{V=49152}"]) == 1


def test_stacked_kv_cache_layout_also_resolves():
    """`[2, num_blocks, block_size, KVH, D]` with contiguous unbind(0) views (the layout DESIGN §6 anticipated)."""
    events, start = decode_step(kv_layout="stacked")
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    assert not [r for r in res if isinstance(r, Unsupported)]
    kv = [r for r in res if isinstance(r, TransparentGroup)][0]
    _, key_cache, value_cache = kv_cache_views("stacked")
    kcs, vcs = as_slot_rows(key_cache), as_slot_rows(value_cache)
    assert kcs.strides == (KVH * D, D, 1) and vcs.offset == NUM_BLOCKS * BS * KVH * D
    assert kv.dst == Gather(kcs, (32,)) and kv.also[0].dst == Gather(vcs, (32,))
    with pytest.raises(GeometryError):
        as_slot_rows(bf(A_KV, (NUM_BLOCKS, BS, KVH, D), (BS * KVH * D + 7, KVH * D, D, 1)))   # non-uniform slot rows


def test_unknown_op_is_unsupported_and_consumes_its_subtree():
    events, start = decode_step(with_unknown=True)
    se = step_events(events)
    res = resolve_step(se, context(start, se.sample))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == "vllm.some_unknown_op.default" and uns[0].reason == "no pattern"
    node = next(r for r in se.roots if r.op == uns[0].op)
    assert uns[0].consumed == (node.id,)


def test_unconsumed_child_raises():
    events, start = decode_step(with_dangling=True)
    se = step_events(events)
    with pytest.raises(ResolutionError, match="unconsumed"):
        resolve_step(se, context(start, se.sample))


def test_geometry_helpers():
    w = ViewDesc(5, 0, (960, 576), (576, 1), "bfloat16", CUDA)
    wt = ViewDesc(5, 0, (576, 960), (1, 576), "bfloat16", CUDA)
    assert as_weight_view(wt, 576, 960) == w and as_weight_view(w, 576, 960) == w
    with pytest.raises(ValueError):
        as_weight_view(ViewDesc(5, 0, (576, 960), (960, 1), "bfloat16", CUDA), 576, 960)
    q = ViewDesc(3, 0, (4, 576), (960, 1), "bfloat16", CUDA)
    assert split_heads(q, 9, 64) == ViewDesc(3, 0, (4, 9, 64), (960, 64, 1), "bfloat16", CUDA)
    with pytest.raises(ValueError):
        split_heads(ViewDesc(3, 0, (4, 576), (1, 4), "bfloat16", CUDA), 9, 64)
    cache = ViewDesc(9, 7, (64, 16, 3, 64), (3072, 192, 64, 1), "bfloat16", CUDA)
    assert as_slot_rows(cache) == ViewDesc(9, 7, (1024, 3, 64), (192, 64, 1), "bfloat16", CUDA)
    with pytest.raises(ValueError):
        as_slot_rows(ViewDesc(9, 0, (64, 16, 3, 64), (3072, 64, 1024, 1), "bfloat16", CUDA))
    assert schema_param_names("_vllm_fa2_C::varlen_fwd(Tensor q, Tensor k, Tensor!? out, int[] sizes, *, float softmax_scale=0.5) -> Tensor[]") == \
        ["q", "k", "out", "sizes", "softmax_scale"]
    assert schema_param_names("") == []


def test_profile_weight_fields_and_root_classes():
    wf = PROF.weight_field
    assert wf("model.embed_tokens.weight") == ("embed",) and wf("lm_head.weight") == ("embed",)
    assert wf("model.layers.12.input_layernorm.weight") == ("layers", 12, "ln1")
    assert wf("model.layers.12.self_attn.qkv_proj.weight") == ("layers", 12, "qkv_w")
    assert wf("model.layers.12.self_attn.o_proj.weight") == ("layers", 12, "o_w")
    assert wf("model.layers.12.post_attention_layernorm.weight") == ("layers", 12, "ln2")
    assert wf("model.layers.12.mlp.gate_up_proj.weight") == ("layers", 12, "gate_up_w")
    assert wf("model.layers.12.mlp.down_proj.weight") == ("layers", 12, "down_w")
    assert wf("model.norm.weight") == ("norm",)
    assert wf("model.layers.0.self_attn.rotary_emb.cos_sin_cache") == ("cos_sin",)
    assert wf("model.layers.4.self_attn.rotary_emb.cos_sin_cache") is None
    assert PROF.root_class("model.layers.4.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
    assert PROF.root_class("model.layers.4.self_attn.attn.k_scale") == "ignored"
    assert PROF.root_class("model.layers.4.self_attn.qkv_proj.bias") == "unknown"   # BIAS=False: no field
    assert PROF.root_class("something.else") == "unknown"
    assert PROF.check_header(header()) == []
    assert PROF.PROFILE.config["BN"] == 128 and PROF.PROFILE.id == "vllm-d9105ea80-sm89-eager-bf16"
