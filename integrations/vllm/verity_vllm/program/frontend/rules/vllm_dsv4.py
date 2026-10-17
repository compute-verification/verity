"""[DSV4] DeepSeek-V4's sparse MLA as the H200 serves it (vLLM d9105ea80 `models/deepseek_v4/attention.py`, `nvidia/flashmla.py`),
over the leaves `dsv4_leaves.dsv4_attention_leaves` exports in place of its kernels (`vllm_iface`):

    dsv4_qnorm_rope_kv_insert(q [T, NH, D], kv [T, D], positions, cos_sin, padded_heads, eps, quant_block)
        -> (q [T, NHP, D], NoPE [T, DN] e4m3, scale bytes [T, DN/QB + 1], RoPE [T, DR] bf16)
        `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` per token: `Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}` of its live
        heads, its kv row and its position's f32 cos_sin row (`registry.dsv4_qnorm_insert`)
    dsv4_compressor_state(kv_score [T, 2W] f32, positions, ape [R, W] f32) -> (kv [T, W], score [T, W])
        `save_partial_states` per token: kv the first W words, score `Dsv4CompressorScore_v1{W}` of the last W and ape[pos % R]
        (`registry.dsv4_indexer`)

A role the vocabulary lacks refuses by name.
"""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import array_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_dsa import _missing, _positions, _rows, _static_float, _table, _value
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.rules.vllm_mhc import _args
from verity_vllm.program.registry import dsv4_qnorm_insert as Q

BF16, F32 = torch.bfloat16, torch.float32

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"dsv4_qnorm_rope_kv_insert": "the query norm, RoPE and window cache insert (Dsv4QNormRopeKvInsert_v1)",
                "dsv4_compressor_score": "save_partial_states' score row (Dsv4CompressorScore_v1)"}


class Dsv4QNormRopeKvInsertRule(OpRule):
    """`vllm_iface.dsv4_qnorm_rope_kv_insert` per token: `Dsv4QNormRopeKvInsert_v1{NH,NHP,DN,DR,QB,EPS}` of q[t], kv[t] and
    cos_sin[positions[t]]; the node is (q padded, NoPE, scale bytes, RoPE) over the tokens."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4QNormRopeKvInsert[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.dsv4_qnorm_rope_kv_insert",
                    preconditions=("positions a static 1-D index of T rows; q [T, NH, D], kv [T, D] bf16 values",
                                   "cos_sin an f32 rotary cache [MAX_POS, DR] holding every position",
                                   "padded_heads, quant_block static ints and eps a static float as the Definition's statics check them",
                                   "not under registered_values"),
                    evidence_domain="structural; numerics: Dsv4QNormRopeKvInsert_v1 (registry.dsv4_qnorm_insert)")

    def targets(self):
        return {iface.dsv4_qnorm_rope_kv_insert.default}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["dsv4_qnorm_rope_kv_insert"], ROLE_KERNELS)):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: the insert takes eps as a static; no registered-values twin"
        a = _args(node)
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        if (r := _value(ctx, a["q"], BF16, (T, None, None), "q")):
            return r
        _, NH, D = shape(ctx, a["q"])
        DR = (shape(ctx, a["cos_sin"]) or (0, 0))[-1]
        if (r := _value(ctx, a["kv"], BF16, (T, D), "kv") or _table(ctx, a["cos_sin"], DR, pos, "cos_sin", F32)):
            return r
        NHP, QB, eps = a["padded_heads"], a["quant_block"], a["eps"]
        if not (isinstance(NHP, int) and isinstance(QB, int) and _static_float(eps)):
            return "padded_heads, quant_block and eps are not statics"
        try:
            Q.check_insert_statics(NH, NHP)
            Q.check_q_statics(D, DR)
            Q.check_kv_statics(D - DR, DR, QB)
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = _args(node)
        pos = ctx.static_index(a["positions"]).flat()
        T, h = len(pos), ctx.hint_name(node)
        _, NH, D = shape(ctx, a["q"])
        DR = shape(ctx, a["cos_sin"])[1]
        S = {"NH": NH, "NHP": a["padded_heads"], "DN": D - DR, "DR": DR, "QB": a["quant_block"], "EPS": float(a["eps"])}
        fn = ctx.kind("dsv4_qnorm_rope_kv_insert", node, **S)
        q, kv, cs = _rows(ctx, a["q"], 16), _rows(ctx, a["kv"], 16), ctx.coll(a["cos_sin"])
        outs = [ctx.call(fn, q[i], kv[i], cs[pos[i]], node=node, rule=self, name=f"{h}[t={pos[i]}]") for i in range(T)]
        ctx.define(node, [array_of([o[k] for o in outs]) for k in range(4)])
        ctx.bind_weight_role(a["cos_sin"], self, "rope.cos_sin_cache", D=DR)
        ctx.record(self, node, {"T": T, "positions": [pos[0], pos[-1]], **S}, emitted=f"{fn.id} x{T}")


class Dsv4CompressorStateRule(OpRule):
    """`vllm_iface.dsv4_compressor_state` per token: the kv row is kv_score[t, :W] (no gates), the score row
    `Dsv4CompressorScore_v1{W}` of kv_score[t, W:] and ape[positions[t] % R]; the node is (kv, score) over the tokens."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4CompressorState[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.dsv4_compressor_state",
                    preconditions=("positions a static 1-D index of T rows of non-negative positions",
                                   "kv_score an f32 value [T, 2W]; ape an f32 value [R, W]"),
                    evidence_domain="structural; numerics: Dsv4CompressorScore_v1 (registry.dsv4_indexer)")

    def targets(self):
        return {iface.dsv4_compressor_state.default}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["dsv4_compressor_score"], ROLE_KERNELS)):
            return r
        a = _args(node)
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        if any(p < 0 for p in pos):
            return "a negative position"
        if (r := _value(ctx, a["ape"], F32, (None, None), "ape")):
            return r
        W = shape(ctx, a["ape"])[1]
        return _value(ctx, a["kv_score"], F32, (len(pos), 2 * W), "kv_score") or True

    def emit(self, node, ctx):
        a = _args(node)
        pos = ctx.static_index(a["positions"]).flat()
        h = ctx.hint_name(node)
        R, W = shape(ctx, a["ape"])
        fn = ctx.kind("dsv4_compressor_score", node, W=W)
        rows, ape = _rows(ctx, a["kv_score"], 32), ctx.coll(a["ape"])
        score = [ctx.call(fn, rows[i][W:], ape[p % R], node=node, rule=self, name=f"{h}[t={p}]") for i, p in enumerate(pos)]
        ctx.define(node, [array_of([r[:W] for r in rows]), array_of(score)])
        ctx.bind_weight_role(a["ape"], self, "dsv4.compressor.ape", R=R, W=W)
        ctx.record(self, node, {"T": len(pos), "positions": [pos[0], pos[-1]], "R": R, "W": W}, emitted=f"{fn.id} x{len(pos)}")


VLLM_DSV4_RULES = [Dsv4QNormRopeKvInsertRule(), Dsv4CompressorStateRule()]

__all__ = ["Dsv4CompressorStateRule", "Dsv4QNormRopeKvInsertRule", "VLLM_DSV4_RULES"]
