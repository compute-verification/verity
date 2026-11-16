"""GLM-5.3's DSA attention over an FP8 KV cache (`fp8_ds_mla`, `--kv-cache-dtype fp8`, vLLM d9105ea80's FLASHMLA_SPARSE): the
query concat and FlashMLA's sparse fp8 decode, over the cache `dsa.DsaNormRope` wrote and the request-index conversion
`dsa.DsaIndexShare` staged."""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

from dataclasses import dataclass

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.dsa_cache import (
    RECORD_SMS,
    DsMlaCache,
    _module,
    _module_state,
    _state,
)
from verity_vllm.observe.fold.patterns.geometry import GeometryError, is_row_major
from verity_vllm.observe.fold.patterns.kv_cache import kv_slots
from verity_vllm.observe.fold.patterns.ops import bound_args, op_name
from verity_vllm.observe.fold.patterns.resolutions import unsupported
from verity_vllm.observe.fold.patterns.target import target_of, vocab_kind
from verity_vllm.observe.fold.resolver import (
    Compute,
    Const,
    GatherMany,
    Group,
    PerRow,
    Pieces,
    Resolution,
    StepContext,
    Transparent,
)
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.program.frontend.rules.vllm_dsa_fp8 import (
    FLASHMLA_FP8_GEOMETRY,
    FP8_PADDED_HEADS,
    decode_splits,
)

CONCAT_Q_OP = "concat_mla_q"
FLASHMLA_OP = "sparse_decode_fwd"


@dataclass
class DsaConcatQ:
    """``_C_cache_ops.concat_mla_q`` (FlashMLA's query over an `fp8_ds_mla` cache): each head's absorbed NoPE row, then its rotated
    q_pe, as one row [NH, 512 + 64]; a copy of both into the output's halves."""

    name = "dsa.concat_mla_q"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != CONCAT_Q_OP or not node.op.startswith("_C_cache_ops."):
            return None
        a = bound_args(node)
        nope, pe, out = a.get("ql_nope"), a.get("q_pe"), a.get("q_out")
        if not (isinstance(nope, ViewDesc) and isinstance(pe, ViewDesc) and isinstance(out, ViewDesc)):
            return unsupported(node, "concat_mla_q pointers are not bound", ql_nope=repr(nope), q_pe=repr(pe), q_out=repr(out))
        M = len(ctx.start.rows)
        if nope.shape != (M, 8, 512) or pe.shape != (M, 8, 64) or out.shape != (M, 8, 576) or not nope.dtype == pe.dtype == out.dtype == "bfloat16" \
                or nope.strides[2] != 1 or pe.strides[2] != 1 or not is_row_major(out):
            return unsupported(node, "concat_mla_q views differ from bf16 [M,8,512] and [M,8,64] into a row-major [M,8,576]",
                               ql_nope=repr(nope), q_pe=repr(pe), q_out=repr(out), step_rows=M)
        halves = [ViewDesc(out.alloc, out.offset + o, (M, 8, w), out.strides, out.dtype, out.device) for o, w in ((0, 512), (512, 64))]
        mod = _module(node)
        return Group((Transparent("move", (node.id,), src=PerRow(nope), dst=PerRow(halves[0]), rows=M, hint=f"{mod}/q_nope",
                                  kernels=(CONCAT_Q_OP,), pattern=self.name),
                      Transparent("move", (), src=PerRow(pe), dst=PerRow(halves[1]), rows=M, hint=f"{mod}/q_pe", pattern=self.name)),
                     tuple(sorted(node.ids())), hint=mod, pattern=self.name)


@dataclass
class DsaFlashMlaDecode:
    """FlashMLA's sparse fp8 decode over the `fp8_ds_mla` cache (``_flashmla_C.sparse_decode_fwd``, vLLM's
    ``FlashMLASparseImpl._forward_fp8_kv_mixed_batch``): every row of the step in one call at `FP8_PADDED_HEADS` padded heads, so the
    planner gives all of them one SPLITS (`decode_splits`).  Row i of n keys is `FlashMlaV32Decode_v1` over its live heads' query and
    TOPK cache entries, its request's n slots in key order and then slot 0's under index -1, which the kernel zeroes: the short-row
    top-k's selection, which `DsaTopK` folds without reading its values."""

    name = "dsa.flashmla_sparse_decode"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != FLASHMLA_OP or not node.op.startswith("_flashmla_C."):
            return None
        a, outs = bound_args(node), node.output_views()
        q, kv, idx = a.get("q"), a.get("kv"), a.get("indices")
        if not (isinstance(q, ViewDesc) and isinstance(kv, ViewDesc) and isinstance(idx, ViewDesc)) or not outs:
            return unsupported(node, "FlashMLA sparse decode q/kv/indices/out pointers are not bound")
        out, M, (DR, DN), TOPK = outs[0], len(ctx.start.rows), next(iter(FLASHMLA_FP8_GEOMETRY)), 2048
        G, NH, NHP = FLASHMLA_FP8_GEOMETRY[(DR, DN)], 8, FP8_PADDED_HEADS
        if q.shape != (1, M, NHP, DN + DR) or out.shape != (1, M, NHP, DN) or q.dtype != "bfloat16" or out.dtype != "bfloat16" \
                or q.strides[3] != 1 or out.strides[3] != 1 or idx.shape != (1, M, TOPK) or idx.dtype != "int32":
            return unsupported(node, "FlashMLA sparse decode views differ from bf16 [1,M,64,576] into [1,M,64,512] over i32 [1,M,2048]",
                               q=repr(q), out=repr(out), indices=repr(idx), step_rows=M)
        # the step's first call plans the split schedule and returns it beside out and lse; every later layer passes that plan back
        state = _state(ctx)
        plan = state.get("flashmla_plan")
        pins = {"topk_length": None, "attn_sink": None, "tile_scheduler_metadata": None if plan is None else plan[0],
                "num_splits": None if plan is None else plan[1], "extra_kv": None, "extra_indices": None, "extra_topk_length": None,
                "d_v": DN, "sm_scale": 0.0625, "out_": None}
        bad = {key: (a.get(key), val) for key, val in pins.items() if a.get(key) != val}
        if bad:
            return unsupported(node, "FlashMLA sparse decode arguments differ from GLM-5.3's fp8 mixed batch: the step's first call plans "
                               "the splits, each later one passes that plan", mismatches=bad)
        if plan is None and len(outs) != 4:
            return unsupported(node, "the step's first FlashMLA sparse decode does not return its split plan", outputs=[repr(o) for o in outs])
        ms = _module_state(ctx, node)
        cache = ms.get("ds_mla")
        if not isinstance(cache, DsMlaCache) or (kv.alloc, kv.offset, kv.shape, kv.strides) != (
                cache.nope.alloc, cache.nope.offset, (cache.blocks, cache.BS, 1, cache.W), (cache.BS * cache.W, cache.W, cache.W, 1)):
            return unsupported(node, "FlashMLA does not read the fp8_ds_mla cache fused_norm_rope wrote", kv=repr(kv),
                               established=repr(cache.nope) if isinstance(cache, DsMlaCache) else None)
        page = (idx.alloc, idx.offset, idx.shape[2], idx.strides[1:])
        if page != ms.get("page_table"):
            return unsupported(node, "FlashMLA indices are not its layer's request-index conversion", indices=page,
                               established=ms.get("page_table"))
        tp = target_of(ctx)[0]
        if int(tp.num_sms) != RECORD_SMS:
            return unsupported(node, "GLM-5.3 FlashMLA split plan is pinned to the record target's 132-SM launch", num_sms=tp.num_sms)
        entries, consts = [], []
        for req, position in ctx.start.rows:
            n = int(position) + 1
            if n > TOPK:
                return unsupported(node, "a FlashMLA row of more keys than TOPK reads the rows its top-k selects, which no Definition routes",
                                   position=position, keys=n, TOPK=TOPK)
            try:
                slots = kv_slots(ctx, req, n, layer=_module(node))
            except GeometryError as e:
                return unsupported(node, str(e), request=req, T=n)
            if any(s >= cache.slots for s in slots):
                return unsupported(node, "a selected cache slot is outside the fp8_ds_mla cache", T=n, cache_slots=cache.slots)
            entries.append((*slots, *(slots[0],) * (TOPK - n)))
            consts.append(tuple(Const(32, bits, "idx", shared=True, count=k) for bits, k in ((0, n), (0xFFFFFFFF, TOPK - n)) if k))
        spec = vocab_kind(ctx, "flashmla_v32_decode", NH=NH, SPLITS=decode_splits(M, TOPK, int(tp.num_sms), G["BN"]), DN=DN, DR=DR,
                          QT=G["QT"], BN=G["BN"], DOT=G["DOT"], SCALE=0.0625)
        qv, ov = (ViewDesc(v.alloc, v.offset, (M, NH, v.shape[3]), (v.strides[1], v.strides[2], 1), v.dtype, v.device) for v in (q, out))
        if plan is None:
            state["flashmla_plan"] = (outs[2], outs[3])
        return Compute(spec, M, (PerRow(qv), GatherMany(cache.nope, tuple(entries)), GatherMany(cache.scale, tuple(entries)),
                                 GatherMany(cache.rope, tuple(entries)), Pieces(tuple(consts))),
                       (PerRow(ov),), tuple(sorted(node.ids())), hint=f"{_module(node)}/flashmla", kernels=(FLASHMLA_OP,),
                       attribution=tuple(ctx.start.rows), pattern=self.name)


__all__ = ["DsaConcatQ", "DsaFlashMlaDecode"]
