"""Capture-v2 identity -> `ServeTP2` value for the **compiled** TP=2 capture (lane COMP, CX1: compiled x TP).

The program is TP's `ServeTP2` unchanged (`registry.b1_tp2`); the capture is W1d's compiled-fx capture run in every
rank's worker process (`verity_vllm_adapter.compiled_fx.tp_worker`).  This module is a *mapping* only: TP's
`CircuitMapTP2` (rank kinds member / rank-call / replicated, the same named root calls) over the compiled partition's
(op suffix, port) table, i.e. CF2's `COMPILED_LAYER_OPS` with the rank kinds attached and the TP nodes added.

Written before any compiled TP=2 output existed, against the *Ampere compiled TP=1* capture's node/port names
(`B1-bdd4a899b1dd-20260907T2138Z`) and the *eager TP=2* capture's rank kinds (`correspondence_tp2.TP2_OPS`).  The
predicted differences from the compiled TP=1 partition (`prediction.json` "capture"):

  * the embedding kernel is vLLM's `_C.vocab_parallel_embedding` (an extern custom op, one port: the rank's masked
    partial rows), followed by `vllm.all_reduce` (extern; its output is the residual entering layer 0) and a
    *standalone* Inductor `rms_norm` kernel for layer 0's input norm (the all-reduce sits between the gather and the
    norm, so W1d's fused embedding+norm node cannot form) -> three nodes where compiled TP=1 had one with two ports;
  * two more extern nodes per decoder layer: `self_attn.o_proj.all_reduce`, `mlp.down_proj.all_reduce` (each a
    materialised buffer, so W1d's "materialised" policy records them without a policy change);
  * the eager tail records the lm_head shard (`lm_head.gemm`, before the all-gather) and the gathered row
    (`lm_head.all_gather`) as the eager TP capture does.

Two alternates are admitted in the table, clearly labelled: the B0-style rope partition (roped q and k as two ports of
one kernel) in case Inductor's fusion decision differs at 6+1 heads per rank.  Admitting a lookup key is not fitting:
which partition occurred is itself a scored row of the prediction.
"""
from __future__ import annotations

from verity_vllm.correspondence.core import NotInProgram, _returns_component
from verity_vllm.correspondence.tp2 import CaptureIdentityTP, CircuitMapTP2, _op_key
from verity.ir.layout import FunctionCall

#: (op suffix after `layers.L.`, port) -> (root call role, body node name in ServeTP2, return component, rank kind)
TP2_COMPILED_OPS: dict[tuple[str, int], tuple[str, str, int, str]] = {
    # piece 0 head: extern embedding gather (rank's partial), its all-reduce, the standalone layer-0 norm (1 port)
    ("embed_tokens.embedding", 0): ("embed", "embed_tokens.embedding", 0, "rank-call"),
    ("embed_tokens.all_reduce", 0): ("embed", "embed_tokens.all_reduce", 0, "replicated"),
    # L >= 1: Inductor fused add+rmsnorm, 2 ports (normed, residual out); L = 0: standalone rms_norm, port 0 only
    ("input_layernorm.rmsnorm", 0): ("pre", "input_layernorm.rmsnorm", 0, "replicated"),
    ("input_layernorm.rmsnorm", 1): ("pre", "input_layernorm.rmsnorm", 1, "replicated"),
    ("self_attn.qkv_proj.gemm", 0): ("pre", "self_attn.qkv_proj.gemm", 0, "member"),      # user-Triton GEMM HOP, pre-bias (rank's 1024 columns)
    ("self_attn.qkv_proj.rope", 0): ("pre", "self_attn.qkv_proj.add", 0, "member"),       # combo kernel port 0: bf16(qkv + bias)
    ("self_attn.qkv_proj.rope", 1): ("pre", "self_attn.rotary_emb.rope.k", 0, "member"),  # combo kernel port 1: roped k (1 head)
    ("self_attn.rotary_emb.rope", 0): ("pre", "self_attn.rotary_emb.rope", 0, "member"),  # q-rope kernel (6 heads)
    ("self_attn.rotary_emb.rope", 1): ("pre", "self_attn.rotary_emb.rope.k", 0, "member"),  # ALTERNATE (B0-style partition): roped k as port 1 of the rope kernel
    ("self_attn.attn.attn", 0): ("attn", "self_attn.attn.attn", 0, "member"),
    ("self_attn.o_proj.gemm", 0): ("post", "self_attn.o_proj.gemm", 0, "member"),         # the rank's pre-all-reduce partial
    ("self_attn.o_proj.all_reduce", 0): ("post", "self_attn.o_proj.all_reduce", 0, "replicated"),
    ("post_attention_layernorm.rmsnorm", 0): ("post", "post_attention_layernorm.rmsnorm", 0, "replicated"),  # 1 port: normed (r2 never in memory)
    ("mlp.gate_up_proj.gemm", 0): ("post", "mlp.gate_up_proj.gemm", 0, "member"),
    ("mlp.act_fn.act_mul", 0): ("post", "mlp.act_fn.act_mul", 0, "member"),
    ("mlp.down_proj.gemm", 0): ("post", "mlp.down_proj.gemm", 0, "member"),
    ("mlp.down_proj.all_reduce", 0): ("post", "mlp.down_proj.all_reduce", 0, "replicated"),
    ("norm.rmsnorm", 0): ("final", "norm.rmsnorm", 0, "replicated"),                       # 1 port (compiled: the residual output is dead)
    ("lm_head.gemm", 0): ("final", "lm_head.gemm", 0, "member"),                           # eager tail: the rank's logits shard
    ("lm_head.all_gather", 0): ("final", "lm_head.all_gather", 0, "replicated"),           # eager tail: the gathered row
    ("sampler.select", 0): ("final", "sampler.select", 0, "replicated"),
}

#: leaves holding every position's K and V rows (attention operands): the combo kernel's two ports (B1 compiled partition);
#: the alternate partition's k port is listed too (a key that never occurs costs nothing)
TP2_COMPILED_KV_SOURCES = frozenset({("self_attn.qkv_proj.rope", 0), ("self_attn.qkv_proj.rope", 1), ("self_attn.rotary_emb.rope", 1)})
#: layer L-1 leaves needed to evaluate layer L's input norm: h = down all-reduce output; the residual r2 (never materialised) is
#: recomputed from the committed o_proj all-reduce output and the committed residual entering L-1 (input norm port 1 / embed all-reduce)
TP2_COMPILED_PREV_LAYER_SOURCES = ("mlp.down_proj.all_reduce", "self_attn.o_proj.all_reduce", "input_layernorm.rmsnorm")


class CircuitMapTP2Compiled(CircuitMapTP2):
    """`CircuitMapTP2` over the compiled partition's table (`TP2_COMPILED_OPS`)."""

    OPS = TP2_COMPILED_OPS

    def locate(self, ident: CaptureIdentityTP) -> tuple[FunctionCall, tuple[int, ...], str]:
        t = ident.t
        key = _op_key(ident)
        if key not in self.OPS:
            raise NotInProgram(f"{ident.op}/port={ident.output}: no node of the TP=2 program in the compiled-partition table")
        role, node, which, kind = self.OPS[key]
        if role == "embed":
            root_name = f"t={t}/embed"
        elif role == "final":
            root_name = f"t={t}/final"
            if root_name not in self.named:
                raise NotInProgram(f"t={t}: production computes {ident.op} at every prefill position; the canonical program "
                                   f"has a Final call only at sampled positions (this value has no consumer)")
        else:
            root_name = f"t={t}/layers.{ident.layer}/{role}"
        root = self.named[root_name]
        kids = self._children(root)
        r = ident.shard or 0
        if kind == "member":
            c = kids[(node, r)]
        elif kind == "rank-call":
            c = kids[(f"{node}/s={r}", 0)]
        else:
            c = kids[(node, 0)]
        return c, _returns_component(c, which), kind


def identities() -> dict:
    table = {f"{suffix}/port={port}/s=<r>": {"call": f"t=<t>/{role}", "select": sel, "returns_component": which, "rank_kind": kind}
             for (suffix, port), (role, sel, which, kind) in TP2_COMPILED_OPS.items()}
    table["_identity"] = "p=<profile>/r=<request>/t=<t>/ph=<phase>/l=<layer>/op=<op>/port=<k>/s=<rank>: W1d's compiled-fx identities per rank (the record pass runs in every worker), joined by tp_merge"
    table["_partition"] = "W1d's compiled partition (CF2 COMPILED_LAYER_OPS) + the two all-reduce extern nodes per layer + embedding partial / all-reduce / standalone layer-0 norm"
    table["_kv_sources"] = sorted(f"{s}/port={p}" for s, p in TP2_COMPILED_KV_SOURCES)
    table["_prev_layer_sources"] = list(TP2_COMPILED_PREV_LAYER_SOURCES)
    table["_unmaterialised_recomputed"] = "LayerPostTP2 output 1 (r2 = RMSNormFusedCuda(o_proj.all_reduce, residual_in) port 1): recomputed by the IR from committed values (correspondence.Recomputer)"
    table["_alternate_keys"] = ["self_attn.rotary_emb.rope/port=1 (B0-style rope partition; not predicted for B1)"]
    return table


__all__ = ["TP2_COMPILED_OPS", "TP2_COMPILED_KV_SOURCES", "TP2_COMPILED_PREV_LAYER_SOURCES", "CircuitMapTP2Compiled", "identities"]
