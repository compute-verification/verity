"""Capture-v2 identity -> canonical-circuit value for W1d's **compiled-fx** capture (lane CF2).

The compiled capture (`verity_vllm_adapter.compiled_fx`, `compiled-fx-capture.md` §4) materialises B1's decoder block
at a different partition from the eager capture TA1's `CircuitMap` was written against: the embedding gather and the
layer-0 input norm are one node (`embed_tokens.embedding`, ports 0 = rows, 1 = normed), the qkv bias add and the k-rope
are one node (`self_attn.qkv_proj.rope`, ports 0 = bias-added qkv, 1 = roped k), q-rope is its own node
(`self_attn.rotary_emb.rope`, port 0), the post-attention norm emits only the normed value (the mid-layer residual is
never in memory), and the next layer's input norm emits (normed, residual).  The *program* is unchanged — CF1's
`Serve_v2` under a binding record — so this module is a second **mapping** from identities to the same named root
calls (`t=<t>/embed`, `t=<t>/layers.<L>/{pre,attn,post}`, `t=<t>/final`) and child selectors, nothing more: no
definition, static or primitive is touched, and the mapping's table is data (`COMPILED_LAYER_OPS`).

The value the compiled graph never materialises (the residual `r2 = bf16(o_proj + residual_in)` that `LayerPost_v2`
returns and `LayerPre_v2` consumes) is obtained by the correspondence tool through *exact recomputation from other
committed values* (`verity_vllm.correspondence.core.check_call(..., recompute_missing=True)`, W1cF's stated path); the
records say which regions were recomputed.
"""
from __future__ import annotations

from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, NotInProgram, _returns_component, _select_child
from verity.ir.layout import FunctionCall

#: (op suffix after "layers.L.", port) -> (root call role, child function selector, which return component)
COMPILED_LAYER_OPS: dict[tuple[str, int], tuple[str, str | None, int | None]] = {
    ("input_layernorm.rmsnorm", 0): ("pre", "RMSNorm*", 0),          # L >= 1: `triton_red_fused_fused_add_rms_norm_2` normed
    ("input_layernorm.rmsnorm", 1): ("pre", "RMSNormFusedCuda", 1),  # L >= 1: its residual output bf16(down + (o_proj + res))
    ("self_attn.qkv_proj.gemm", 0): ("pre", "Gemm", 0),              # the user-Triton GEMM HOP: pre-bias
    ("self_attn.qkv_proj.rope", 0): ("pre", "BiasAdd", 0),           # combo kernel port 0: bf16(qkv + bias)
    ("self_attn.qkv_proj.rope", 1): ("pre", "RoPE#1", 0),            # combo kernel port 1: roped k (KVH x D)
    ("self_attn.rotary_emb.rope", 0): ("pre", "RoPE#0", 0),          # q-rope kernel: roped q (NH x D)
    ("self_attn.attn.attn", 0): ("attn", None, 0),
    ("self_attn.o_proj.gemm", 0): ("post", "Gemm#0", 0),
    ("post_attention_layernorm.rmsnorm", 0): ("post", "RMSNormFusedCuda", 0),  # normed only (1 port)
    ("mlp.gate_up_proj.gemm", 0): ("post", "Gemm#1", 0),
    ("mlp.act_fn.act_mul", 0): ("post", "SiluMul", 0),
    ("mlp.down_proj.gemm", 0): ("post", "Gemm#2", 0),
}

#: leaves that hold every position's K and V rows (the attention operands): the combo kernel's two ports
COMPILED_KV_SOURCES = frozenset({("self_attn.qkv_proj.rope", 0), ("self_attn.qkv_proj.rope", 1)})
#: layer L-1 leaves needed to evaluate layer L's input norm (h = down output; the residual is recomputed from o_proj + residual_in)
COMPILED_PREV_LAYER_SOURCES = ("mlp.down_proj.gemm", "self_attn.o_proj.gemm", "input_layernorm.rmsnorm")


class CircuitMapCompiled(CircuitMap):
    """`CircuitMap` for the compiled-fx capture's identities; the same named root calls, a different table."""

    def locate(self, ident: CaptureIdentity) -> tuple[FunctionCall, tuple[int, ...]]:
        t = ident.t
        if ident.layer == "embed":
            if ident.output == 0:  # the gathered rows
                c = self.named[f"t={t}/embed"]
                return c, c.returns
            # port 1: the layer-0 input norm fused into the same kernel
            c = _select_child(self.named[f"t={t}/layers.0/pre"], "RMSNorm*", self.names)
            return c, _returns_component(c, 0)
        if ident.layer in ("norm", "lm_head", "sampler") and f"t={t}/final" not in self.named:
            raise NotInProgram(f"t={t}: production computes {ident.op} at every prefill position; the canonical program "
                               f"has a Final call only at sampled positions (this value has no consumer)")
        if ident.layer == "norm":
            fin = self.named[f"t={t}/final"]
            c = _select_child(fin, "RMSNormFusedCuda", self.names)
            return c, _returns_component(c, ident.output)
        if ident.layer == "lm_head":
            c = _select_child(self.named[f"t={t}/final"], "Gemm", self.names)
            return c, c.returns
        if ident.layer == "sampler":
            c = _select_child(self.named[f"t={t}/final"], "TokenSelect", self.names)
            return c, c.returns
        L = int(ident.layer)
        suffix = ident.op[len(f"layers.{L}."):]
        try:
            role, sel, which = COMPILED_LAYER_OPS[(suffix, ident.output)]
        except KeyError:
            raise NotInProgram(f"{ident.op}/port={ident.output}: no counterpart in the compiled mapping table") from None
        root = self.named[f"t={t}/layers.{L}/{role}"]
        c = root if sel is None else _select_child(root, sel, self.names)
        return c, _returns_component(c, which or 0)


def identities(names: dict[str, str] | None = None) -> dict:
    names = names or {}
    return {
        "layer_ops": {f"layers.<L>.{suffix}/port={port}": {"call": f"t=<t>/layers.<L>/{role}", "select": names.get(sel, sel) if sel else None, "returns_component": which}
                      for (suffix, port), (role, sel, which) in COMPILED_LAYER_OPS.items()},
        "embed_tokens.embedding/port=0": {"call": "t=<t>/embed", "note": "the gathered rows (fused kernel port 0)"},
        "embed_tokens.embedding/port=1": {"call": "t=<t>/layers.0/pre", "select": "RMSNorm*", "returns_component": 0, "note": "layer-0 input norm fused into the embedding kernel"},
        "norm.rmsnorm/port=0": {"call": "t=<t>/final", "select": "RMSNormFusedCuda", "returns_component": 0, "note": "sampled positions only; 1 port (normed) in the compiled capture"},
        "lm_head.gemm/port=0": {"call": "t=<t>/final", "select": names.get("Gemm", "Gemm"), "note": "sampled positions only"},
        "sampler.select/port=0": {"call": "t=<t>/final", "select": "TokenSelect", "note": "sampled positions only"},
        "kv_sources": sorted(f"{s}/port={p}" for s, p in COMPILED_KV_SOURCES),
        "prev_layer_sources": list(COMPILED_PREV_LAYER_SOURCES),
        "unmaterialised_recomputed": ["LayerPost_v2 output 1 (r2 = RMSNormFusedCuda(o_proj, residual_in) port 1): the mid-layer residual, never in memory in the compiled graph; "
                                      "recomputed by the IR from the committed o_proj output and the committed residual entering the layer"],
        "call_names": names,
        "identical_to_ta1": False,
        "table": "verity_vllm.correspondence.compiled.COMPILED_LAYER_OPS",
    }


__all__ = ["COMPILED_LAYER_OPS", "COMPILED_KV_SOURCES", "COMPILED_PREV_LAYER_SOURCES", "CircuitMapCompiled", "identities"]
