"""Rope fused with the KV-cache write: on each layer that runs vLLM's `LlamaAttention.forward` over FA's `FlashAttentionImpl`,
one Triton launch after the qkv GEMM (`rope_cache`) does what `_C.rotary_embedding` and `_C_cache_ops.reshape_and_cache_flash`
do in two:

    q, k  rotated in place in the qkv buffer           rotary_embedding_kernel<bf16, bf16, neox>'s arithmetic
    k, v  written into the paged cache at slot_mapping  reshape_and_cache_flash_kernel<bf16, bf16, auto>'s copy

so a token's q, k and v are read once.  For each pair (x, y) = (row[i], row[i + D/2]) of a head and its cos/sin row
(c, s) = (cache[pos, i], cache[pos, D/2 + i]), vLLM's kernel computes, as nvcc contracts ``x*c - y*s`` and ``y*c + x*s``:

    x' = RN_bf16(fma(x, c, -RN32(y*s)))      y' = RN_bf16(fma(x, s, RN32(y*c)))      no flush to zero

and the kernel here computes the same with tl.fma and mul.rn.f32 (`OPTS`: no contraction, no flush to zero; `ptx_check`
reads the PTX for it).  It writes every byte the two ops write and nothing else: q and k in the qkv buffer, and k and v at
each token's slot (block slot // block_size, offset slot % block_size, by the cache's strides) for the first
slot_mapping.numel() tokens whose slot is not negative.  `tests/engine/test_rope_cache.py` holds it to the two ops bit for
bit, edge values, position 0 and the last position, partial blocks and padded slots included.

The layer's forward (an instance ``forward`` patched through `engine.hooks`) runs vLLM's body statement by statement
(`_BODY`, checked against the pinned vLLM's source at install); where that body calls the rope and then the attention
layer, it launches `rope_cache` and calls the attention layer, whose ``do_kv_cache_update`` (patched on the impl instance)
skips exactly the write the kernel made: the same key, value, cache and slot mapping, else it raises.  The layer runs vLLM's
two ops instead whenever its call isn't the served one the kernel reproduces: a dispatch or function mode, a compile or a
CUDA graph capture is active; the rope or the attention layer has a hook or an instance ``forward`` (the Commit's taps,
`commit.committer.native_collect`); a function either op or the attention layer reaches is no longer the one installed
over (an observer's patch); or a dtype, layout or the forward context differs.  Nothing is installed when
``VERITY_ROPE_CACHE_FUSED`` is off, when the pinned body differs, or on a layer whose rope isn't NeoX over the whole head
on `_C.rotary_embedding`, or whose attention isn't a decoder's on FA with a bf16 (``auto``) cache of its own.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import torch
import triton
import triton.language as tl
from torch.nn.modules import module as _nn_module

from verity_vllm import settings
from verity_vllm.engine import hooks

ENV = "VERITY_ROPE_CACHE_FUSED"
TAG = "rope_cache"
OPTS = {"enable_fp_fusion": False, "enable_reflect_ftz": False}
# Tokens a program and warps a program: the fastest of BLOCK_T 1..8 x warps 1..8 at T 8,192, 16,384 and 32,768 on an H200.
BLOCK_T, WARPS = 1, 2

# vLLM d9105ea80's LlamaAttention.forward, statement by statement as ast.unparse prints them.
_ARGS = ("self", "positions", "hidden_states")
_BODY = ("qkv, _ = self.qkv_proj(hidden_states)",
         "q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)",
         "q, k = self.rotary_emb(positions, q, k)",
         "attn_output = self.attn(q, k, v)",
         "output, _ = self.o_proj(attn_output)",
         "return output")


@triton.jit
def _rotate(x, y, c, s):
    # `* -1.0`, not unary minus: Triton lowers `-a` to `0 - a`, which turns -RN32(y*s) = -0 into +0.
    return tl.fma(x, c, (y * s) * -1.0).to(tl.bfloat16), tl.fma(x, s, y * c).to(tl.bfloat16)


@triton.jit
def rope_cache_kernel(pos_ptr, q_ptr, k_ptr, v_ptr, cs_ptr, slot_ptr, kc_ptr, vc_ptr, T, n_slots, stride_q, stride_k,
                      stride_v, block_size, c_block, c_page, c_head, H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
                      HALF: tl.constexpr, HP: tl.constexpr, KVHP: tl.constexpr, HALFP: tl.constexpr,
                      BLOCK_T: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    live = t < T
    t = t.to(tl.int64)
    i = tl.arange(0, HALFP)
    im = i < HALF
    pos = tl.load(pos_ptr + t, mask=live, other=0)
    cs = cs_ptr + pos[:, None] * D + i[None, :]
    m2 = live[:, None] & im[None, :]
    c = tl.load(cs, mask=m2, other=0.0).to(tl.float32)[:, None, :]
    s = tl.load(cs + HALF, mask=m2, other=0.0).to(tl.float32)[:, None, :]

    h = tl.arange(0, HP)
    qo = q_ptr + t[:, None, None] * stride_q + h[None, :, None] * D + i[None, None, :]
    mq = live[:, None, None] & (h < H)[None, :, None] & im[None, None, :]
    x = tl.load(qo, mask=mq, other=0.0).to(tl.float32)
    y = tl.load(qo + HALF, mask=mq, other=0.0).to(tl.float32)
    qx, qy = _rotate(x, y, c, s)
    tl.store(qo, qx, mask=mq)
    tl.store(qo + HALF, qy, mask=mq)

    g = tl.arange(0, KVHP)
    mk = live[:, None, None] & (g < KVH)[None, :, None] & im[None, None, :]
    ko = k_ptr + t[:, None, None] * stride_k + g[None, :, None] * D + i[None, None, :]
    x = tl.load(ko, mask=mk, other=0.0).to(tl.float32)
    y = tl.load(ko + HALF, mask=mk, other=0.0).to(tl.float32)
    kx, ky = _rotate(x, y, c, s)
    tl.store(ko, kx, mask=mk)
    tl.store(ko + HALF, ky, mask=mk)
    vo = v_ptr + t[:, None, None] * stride_v + g[None, :, None] * D + i[None, None, :]
    v0 = tl.load(vo, mask=mk)
    v1 = tl.load(vo + HALF, mask=mk)

    slot = tl.load(slot_ptr + t, mask=live & (t < n_slots), other=-1)
    blk = slot // block_size
    dst = blk * c_block + (slot - blk * block_size) * c_page
    mw = mk & (slot >= 0)[:, None, None]
    at = dst[:, None, None] + g[None, :, None] * c_head + i[None, None, :]
    tl.store(kc_ptr + at, kx, mask=mw)
    tl.store(kc_ptr + at + HALF, ky, mask=mw)
    tl.store(vc_ptr + at, v0, mask=mw)
    tl.store(vc_ptr + at + HALF, v1, mask=mw)


def rope_cache(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos_sin_cache: torch.Tensor,
               key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor, *, block_t: int = BLOCK_T,
               num_warps: int = WARPS):
    """``_C.rotary_embedding(positions, q, k, D, cos_sin_cache, True)`` then ``_C_cache_ops.reshape_and_cache_flash(k, v,
    key_cache, value_cache, slot_mapping, "auto", ...)`` in one launch, in place: positions [T] int64, q [T, H*D], k and v
    [T, KVH*D] bf16 rows with unit stride along a row, cos_sin_cache [max_pos, D] bf16 contiguous, key_cache and
    value_cache [blocks, block_size, KVH, D] bf16 with the same strides and unit stride along D, slot_mapping [n <= T]
    int64.  Returns the launch (`ptx_check` reads it), None for T = 0."""
    n = positions.numel()
    _, block_size, kvh, d = key_cache.shape
    assert positions.dtype == torch.int64 and positions.dim() == 1 and positions.stride(0) == 1, positions.dtype
    assert all(t.dtype == torch.bfloat16 and t.dim() == 2 and t.shape[0] == n and t.stride(1) == 1 for t in (q, k, v))
    assert q.shape[1] % d == 0 and k.shape[1] == v.shape[1] == kvh * d, (q.shape, k.shape, v.shape, key_cache.shape)
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == d
    assert cos_sin_cache.is_contiguous() and d % 2 == 0
    assert key_cache.dtype == value_cache.dtype == torch.bfloat16 and key_cache.stride(3) == 1
    assert value_cache.shape == key_cache.shape and value_cache.stride() == key_cache.stride()
    assert slot_mapping.dtype == torch.int64 and slot_mapping.dim() == 1 and slot_mapping.stride(0) == 1
    assert slot_mapping.numel() <= n, (slot_mapping.numel(), n)
    if n == 0:
        return None
    h = q.shape[1] // d
    return rope_cache_kernel[(triton.cdiv(n, block_t),)](
        positions, q, k, v, cos_sin_cache, slot_mapping, key_cache, value_cache, n, slot_mapping.numel(), q.stride(0),
        k.stride(0), v.stride(0), block_size, key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), H=h, KVH=kvh,
        D=d, HALF=d // 2, HP=triton.next_power_of_2(h), KVHP=triton.next_power_of_2(kvh),
        HALFP=triton.next_power_of_2(d // 2), BLOCK_T=block_t, num_warps=num_warps, **OPTS)


def ptx_check(compiled) -> dict:
    """What the PTX of a compiled `rope_cache_kernel` (a launch's return value) says about its float semantics: fma.rn and
    mul.rn in float32, round-to-nearest bf16 stores, nothing flushed to zero or approximated."""
    ptx = compiled.asm["ptx"]
    doc = {"kernel": compiled.metadata.name, "ftz": ptx.count(".ftz"), "approx": ptx.count(".approx"),
           "fma_rn_f32": ptx.count("fma.rn.f32"), "mul_rn_f32": ptx.count("mul.rn.f32"),
           "mad_f32": ptx.count("mad.f32") + ptx.count("mad.rn.f32"), "cvt_rn_bf16": ptx.count("cvt.rn.bf16"),
           "global_scratch": int(getattr(compiled.metadata, "global_scratch_size", 0) or 0)}
    doc["ok"] = (doc["ftz"] == 0 and doc["approx"] == 0 and doc["mad_f32"] == 0 and doc["fma_rn_f32"] >= 2
                 and doc["mul_rn_f32"] >= 2 and doc["cvt_rn_bf16"] >= 1 and doc["global_scratch"] == 0)
    return doc


def body_matches(forward: Any) -> bool:
    """Whether `forward`'s source is `_ARGS` and `_BODY`, the body the fused forward reproduces."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(forward))).body[0]
    return (isinstance(fn, ast.FunctionDef) and tuple(a.arg for a in fn.args.args) == _ARGS
            and not (fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs)
            and tuple(ast.unparse(s) for s in fn.body) == _BODY)


class _Site:
    """One layer's fused forward and its attention impl's cache update, sharing the pending write between them."""

    def __init__(self, m: Any, update: Any, pins: tuple, context: tuple) -> None:
        self.m, self.rope, self.attn = m, m.rotary_emb, m.attn
        self.update = update                      # the impl's do_kv_cache_update before the patch
        self.pins = pins                          # (owner, name, value): what getattr gave at install
        self.own: tuple = ()                      # (instance, name, value): what the instance itself held after install
        self.available, self.context = context    # vLLM's is_forward_context_available and get_attention_context
        self.d, self.kvh = m.head_dim, m.num_kv_heads
        self.pending: tuple | None = None

    def ready(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple | None:
        """(cos_sin_cache, key_cache, value_cache, slot_mapping, kv_cache) when this call is the served one `rope_cache`
        reproduces, else None."""
        if (torch.compiler.is_compiling() or torch._C._len_torch_dispatch_stack() or torch._C._len_torch_function_stack()
                or torch.cuda.is_current_stream_capturing()):
            return None
        rope, attn = self.rope, self.attn
        if (rope._forward_hooks or rope._forward_pre_hooks or attn._forward_hooks or attn._forward_pre_hooks
                or _nn_module._global_forward_hooks or _nn_module._global_forward_pre_hooks
                or any(getattr(o, n, None) is not x for o, n, x in self.pins)
                or any(hooks.own(o, n) is not x for o, n, x in self.own)):
            return None
        n = q.shape[0]
        if (n == 0 or not q.is_cuda or not all(t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(1) == 1 for t in (q, k, v))
                or positions.dtype != torch.int64 or positions.dim() != 1 or positions.numel() != n
                or positions.stride(0) != 1 or positions.device != q.device or not self.available()):
            return None
        _, layer, kv_cache, slots = self.context(attn.layer_name)
        if (layer is not attn or slots is None or not isinstance(kv_cache, torch.Tensor) or kv_cache.dtype != torch.bfloat16
                or kv_cache.dim() != 4 or kv_cache.device != q.device or slots.dtype != torch.int64 or slots.dim() != 1
                or slots.stride(0) != 1 or slots.numel() > n or slots.device != q.device):
            return None
        kc, vc = kv_cache.transpose(1, 2).split(self.d, dim=-1)
        if kc.shape[2] != self.kvh or kc.shape[3] != self.d or kc.stride(3) != 1 or vc.shape != kc.shape or vc.stride() != kc.stride():
            return None
        cs = rope._match_cos_sin_cache_dtype(q)
        if cs.dtype != torch.bfloat16 or cs.dim() != 2 or cs.shape[1] != self.d or not cs.is_contiguous() or cs.device != q.device:
            return None
        return cs, kc, vc, slots, kv_cache

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        m = self.m
        qkv, _ = m.qkv_proj(hidden_states)
        q, k, v = qkv.split([m.q_size, m.kv_size, m.kv_size], dim=-1)
        got = self.ready(positions, q, k, v)
        if got is None:
            q, k = m.rotary_emb(positions, q, k)
            attn_output = m.attn(q, k, v)
        else:
            cs, kc, vc, slots, kv_cache = got
            rope_cache(positions, q, k, v, cs, kc, vc, slots)
            self.pending = (k.data_ptr(), v.data_ptr(), k.shape[0], kv_cache.data_ptr(), slots.data_ptr(), slots.numel())
            try:
                attn_output = m.attn(q, k, v)
            finally:
                left, self.pending = self.pending, None
            if left is not None:
                raise RuntimeError(f"{m.attn.layer_name}: the attention layer never made the cache update the fused rope "
                                   "kernel wrote for it")
        output, _ = m.o_proj(attn_output)
        return output

    def do_kv_cache_update(self, layer: Any, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
        p, self.pending = self.pending, None
        if p is None:
            return self.update(layer, key, value, kv_cache, slot_mapping)
        if layer is not self.attn or p != (key.data_ptr(), value.data_ptr(), key.shape[0], kv_cache.data_ptr(),
                                           slot_mapping.data_ptr(), slot_mapping.numel()):
            raise RuntimeError(f"{self.attn.layer_name}: the cache update asked for is not the one the fused rope kernel "
                               "just wrote (key, value, cache or slot mapping differ)")
        return None


def _layer_ok(m: Any, rope_cls: type, attn_cls: type, impl_cls: type, decoder: Any) -> bool:
    rope, attn = getattr(m, "rotary_emb", None), getattr(m, "attn", None)
    impl = getattr(attn, "impl", None)
    return (isinstance(rope, rope_cls) and type(rope).forward_cuda is rope_cls.forward_cuda
            and getattr(rope._forward_method, "__func__", None) is rope_cls.forward_cuda
            and getattr(rope._forward_method, "__self__", None) is rope and not rope.use_flashinfer and rope.is_neox_style
            and rope.rotary_dim == rope.head_size == m.head_dim and isinstance(attn, attn_cls)
            and type(attn).forward is attn_cls.forward and type(impl) is impl_cls and attn.attn_type == decoder
            and impl.attn_type == decoder and attn.kv_sharing_target_layer_name is None and attn.query_quant is None
            and attn.kv_cache_dtype == "auto" and impl.kv_cache_dtype == "auto"
            and not attn.attn_backend.forward_includes_kv_cache_update
            and attn.head_size == impl.head_size == m.head_dim and attn.num_kv_heads == impl.num_kv_heads == m.num_kv_heads
            and attn.num_heads == m.num_heads and m.q_size == m.num_heads * m.head_dim
            and m.kv_size == m.num_kv_heads * m.head_dim and hooks.own(m, "forward") is hooks.ABSENT
            and hooks.own(attn, "forward") is hooks.ABSENT and hooks.own(rope, "forward") is hooks.ABSENT
            and hooks.own(impl, "do_kv_cache_update") is hooks.ABSENT)


def install(model: Any) -> dict | None:
    """Install the fused rope and cache write on `model`'s layers that run vLLM's `LlamaAttention.forward` over FA; a record
    of what was installed, or None when nothing was (``VERITY_ROPE_CACHE_FUSED`` off, the pinned body differs, an op is
    patched already, no such layer).  The patches carry `TAG` (`hooks.live(TAG)`)."""
    if not settings.get(ENV):
        return None
    from vllm import _custom_ops as ops
    from vllm.forward_context import is_forward_context_available
    from vllm.model_executor.layers.attention import attention as attn_mod
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.models.llama import LlamaAttention
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends import flash_attn as fa
    if (not body_matches(LlamaAttention.forward) or not hasattr(fa, "reshape_and_cache_flash")
            or not isinstance(torch.ops._C.rotary_embedding, torch._ops.OpOverloadPacket)
            or not isinstance(torch.ops._C_cache_ops.reshape_and_cache_flash, torch._ops.OpOverloadPacket)):
        return None
    sites = [m for _, m in model.named_modules() if type(m).forward is LlamaAttention.forward
             and _layer_ok(m, RotaryEmbedding, attn_mod.Attention, fa.FlashAttentionImpl, AttentionType.DECODER)]
    if not sites:
        return None
    shared = ((ops, "rotary_embedding"), (ops, "reshape_and_cache_flash"), (fa, "reshape_and_cache_flash"),
              (fa.FlashAttentionImpl, "do_kv_cache_update"), (attn_mod, "unified_kv_cache_update"),
              (attn_mod, "get_attention_context"), (attn_mod, "unified_attention_with_output"),
              (torch.ops._C, "rotary_embedding"), (torch.ops._C_cache_ops, "reshape_and_cache_flash"))
    per = {m: shared + ((type(m.rotary_emb), "forward"), (type(m.rotary_emb), "forward_cuda"),
                        (type(m.rotary_emb), "_match_cos_sin_cache_dtype"), (type(m.attn), "forward")) for m in sites}
    pinned = {(id(o), n) for pins in per.values() for o, n in pins}
    if any(isinstance(e, hooks.Patch) and (id(e.owner), e.name) in pinned for e in hooks.live()):
        return None
    context = (is_forward_context_available, attn_mod.get_attention_context)
    h = hooks.Hooks(TAG)
    for m, pins in per.items():
        rope, attn = m.rotary_emb, m.attn
        site = _Site(m, attn.impl.do_kv_cache_update, tuple((o, n, getattr(o, n)) for o, n in pins), context)
        h.patch(attn.impl, "do_kv_cache_update", site.do_kv_cache_update)
        h.patch(m, "forward", site.forward)
        site.own = ((rope, "forward", hooks.ABSENT), (attn, "forward", hooks.ABSENT),
                    (rope, "_forward_method", hooks.own(rope, "_forward_method")),
                    (attn.impl, "do_kv_cache_update", hooks.own(attn.impl, "do_kv_cache_update")))
    _INSTALLED.append(h)
    return {"layers": len(sites), "block_t": BLOCK_T, "num_warps": WARPS}


def uninstall() -> None:
    """Put every layer's forward and cache update back."""
    while _INSTALLED:
        _INSTALLED.pop().uninstall()


_INSTALLED: list[hooks.Hooks] = []

__all__ = ["ENV", "TAG", "install", "ptx_check", "rope_cache", "uninstall"]
