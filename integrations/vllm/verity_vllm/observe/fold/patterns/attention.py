"""The attention scope pattern: `unified_attention_with_output` around one FA2 or FA3 op."""

from __future__ import annotations

import collections
import dataclasses
import math
from dataclasses import dataclass

from verity_vllm.program.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_slot_rows, rows_operand
from verity_vllm.observe.fold.patterns.kv_cache import _kv_moves, kv_layer_of, kv_slots
from verity_vllm.observe.fold.patterns.ops import as_float, bound_args, first, op_name, op_parts
from verity_vllm.observe.fold.patterns.resolutions import _group_compute, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import _targets, target_of
from verity_vllm.observe.fold.patterns.transparent import find_descendants, scope_transparents
from verity_vllm.observe.fold.resolver import Compute, GatherMany, Resolution, StepContext, Supplied, Unsupported
from verity_vllm.observe.fold.tree import TreeNode

FA2_OP_NAMES = frozenset({"varlen_fwd", "flash_attn_varlen_func", "fwd"})


def flash_attn_op_version(op: str) -> int | None:
    """[R13 hopper] The FA library a dispatcher op belongs to, from its namespace: `_vllm_fa2_C.varlen_fwd` -> 2, `_vllm_fa3_C.fwd` -> 3
    (vllm.vllm_flash_attn.flash_attn_varlen_func dispatches on `fa_version`); None when the namespace names neither."""
    ns = op_parts(op)[0] or ""
    if "_vllm_fa3_C" in ns or "fa3" in ns:
        return 3
    if "_vllm_fa2_C" in ns or "fa2" in ns:
        return 2
    return None


@dataclass
class AttentionScope:
    """`vllm.unified_attention_with_output(query, key, value, output!, layer_name, …)` ⊃ the FA2 op
    (`_vllm_fa2_C.varlen_fwd`) → one `Compute(Attention_v3{T,NH,KVH,D,BN}, rows=|group|, (rows(q), GatherMany(k_cache,
    slots 0..T-1), GatherMany(v_cache, …)), (rows(out),))` per distinct `T_i = position_i + 1`, grouped as a
    `ComputeGroup` (primary = first group, `.also` = the rest); the scope's transparent children are consumed (moves
    emitted in `.before`); an embedded `reshape_and_cache_flash` (older layouts) is emitted in `.before` too.

    The FA2 arguments must match the registered body: `causal=True`, `softmax_scale = D**-0.5`, no window / alibi /
    softcap / dropout, `num_splits == 1` (the aligned split-KV path whose kBlockN is the pinned `bn`,
    `b1.fa2_kblock_n(D)`), bf16 everywhere.  Anything else is Unsupported."""

    bn: int
    num_splits: int | None = 1
    name: str = "attention.fa2_varlen_fwd"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        if op_name(node.op) != "unified_attention_with_output":
            return None
        fas = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) in FA2_OP_NAMES)
        if len(fas) != 1:
            return unsupported(node, "unified_attention_with_output without exactly one FA2/FA3 op child", found=[n.op for n in fas],
                               descendants=[n.op for n in node.walk() if n is not node][:16])
        fa = fas[0]
        # [R13 hopper] the FA library the launch went to must be the record target's EFFECTIVE version (`TargetProfile.flash_attn_version`,
        # else vLLM's selector default for the capability): the FA2 and FA3 ops bind different registered chains (`frontend.targets`).
        tp = target_of(ctx)[0]
        fa_v = tp.effective_flash_attn_version()
        op_v = flash_attn_op_version(fa.op)
        if op_v is not None and op_v != fa_v:
            return unsupported(node, f"attention op is FA{op_v} but the record target's effective flash_attn_version is {fa_v}", op=fa.op,
                               target=tp.to_json())
        if fa_v == 3:
            # FA3-only arguments the registered chain does not model: appended KV, q_v, s_aux, context parallelism.  The FP8 descales
            # (`q_descale`/`k_descale`/`v_descale`) vLLM ALWAYS passes (`layer._q_scale.expand(...)`, stride-0 views of the layer's
            # 1.0 scales) are read by the kernel only under `Is_FP8` (vllm-flash-attn 506341a1 hopper/flash_fwd_kernel_sm90.h:426
            # `if constexpr (Is_FP8 && !Has_softcap) softmax_scale_log2 *= q_descale * k_descale`; mainloop_fwd_sm90_tma_gmma_ws.hpp:1109
            # the softcap variant, :1329 `v_descale = !Is_FP8 || ptr == nullptr ? 1.0f : ...`); for the bf16 launch (checked below:
            # q/k/v/out bf16) they are dead arguments, whatever their values -- so they are not a reason to refuse a bf16 record.
            for key in ("k_new", "v_new", "q_v", "s_aux", "cp_tot_seqused_k", "leftpad_k", "kv_batch_idx", "rotary_cos", "rotary_sin"):
                if a_ := bound_args(fa).get(key):
                    if a_ is not None and a_ != 0:
                        return unsupported(node, f"FA3 op argument {key!r} is bound: not the registered chain", **{key: repr(a_)})
            for key in ("q_descale", "k_descale", "v_descale"):
                a_ = bound_args(fa).get(key)
                if a_ is not None and a_ != 0 and getattr(a_, "dtype", "float32") != "float32":
                    return unsupported(node, f"FA3 op argument {key!r} is not a float32 scale view", **{key: repr(a_)})
            cp = bound_args(fa).get("cp_world_size")
            if cp not in (None, 0, 1):
                return unsupported(node, "FA3 context parallelism is not modelled", cp_world_size=cp)
        rcs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "reshape_and_cache_flash")
        keep = {fa.id} | {n.id for n in rcs}
        before, bad = scope_transparents(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        a = bound_args(fa)
        layer_s = kv_layer_of(node) or kv_layer_of(fa)      # KV-cache group key (hybrid KV cache); NOT the instance hint below
        layer = a.get("layer_name") or bound_args(node).get("layer_name") or ""      # hint text, unchanged (instance names)
        for rc in rcs:
            moves = _kv_moves(rc, ctx, self.name, (), layer=layer_s)
            if isinstance(moves, Unsupported):
                return moves
            before.extend(moves)
        q, k, v, out = a.get("q"), a.get("k"), a.get("v"), a.get("out")
        if not all(isinstance(x, ViewDesc) for x in (q, k, v, out)):
            return unsupported(node, "FA2 op arguments q/k/v/out not bound by name (schema unrecognised)",
                               schema=fa.enter.schema, names=list(a)[:24], views=[repr(x) for x in fa.input_views()][:8])
        causal = first(a.get("is_causal"), a.get("causal"))
        if causal is not True:
            return unsupported(node, "FA2 op is not causal", causal=causal)
        scale = as_float(a.get("softmax_scale"))
        if q.rank != 3 or out.rank != 3:
            return unsupported(node, "q/out are not [M, NH, D] views", q=repr(q), out=repr(out))
        M, NH, D = q.shape
        expect = float(D) ** -0.5
        if scale is None or not math.isclose(scale, expect, rel_tol=1e-9, abs_tol=0.0):
            return unsupported(node, "softmax_scale is not D**-0.5", softmax_scale=scale, expected=expect, D=D)
        for key in ("window_size_left", "window_size_right"):
            wv = a.get(key)
            if wv is not None and isinstance(wv, int) and wv >= 0:
                return unsupported(node, "sliding window is not modelled", **{key: wv})
        softcap = as_float(a.get("softcap"))
        if softcap not in (None, 0.0):
            return unsupported(node, "softcap is not modelled", softcap=softcap)
        if a.get("alibi_slopes") is not None:
            return unsupported(node, "alibi is not modelled")
        pdrop = as_float(first(a.get("p_dropout"), a.get("dropout_p")))
        if pdrop not in (None, 0.0):
            return unsupported(node, "dropout is not modelled", p_dropout=pdrop)
        if self.num_splits is not None:
            # paged (block-table) varlen KV forces `num_splits = 1` inside FA2 (`flash_api.cpp:763`, b1.fa2_kblock_n's
            # D54 note): the aligned split-KV kernel whose kBlockN is the pinned `bn`.  A bound `num_splits` of 0
            # (heuristic) or 1 is that path; anything else is a geometry we have not measured.
            if not isinstance(first(a.get("block_table"), a.get("page_table")), ViewDesc):
                return unsupported(node, "FA op without a block table (non-paged KV): the pinned kBlockN is the paged aligned path's",
                                   names=list(a)[:24])
            ns = a.get("num_splits")
            if ns is not None and ns not in (0, self.num_splits):
                return unsupported(node, "num_splits differs from the pinned aligned split-KV path", num_splits=ns, pinned=self.num_splits)
        if not (q.dtype == k.dtype == v.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "Attention_v3 is bf16", dtypes=[q.dtype, k.dtype, v.dtype, out.dtype])
        try:
            kcs, vcs = as_slot_rows(k), as_slot_rows(v)
        except GeometryError as e:
            return unsupported(node, str(e), k=repr(k), v=repr(v))
        KVH = kcs.shape[1]
        if kcs.shape[2] != D or vcs.shape != kcs.shape or NH % KVH:
            return unsupported(node, "KV cache geometry does not match q", q=repr(q), k_cache=repr(kcs), v_cache=repr(vcs))
        if q.strides[1:] != (D, 1) or out.shape != (M, NH, D) or out.strides[1:] != (D, 1):
            return unsupported(node, "q/out rows are not [NH][D] row-major", q=repr(q), out=repr(out))
        if M != len(ctx.start.rows):
            return unsupported(node, "FA2 query rows differ from the step's token rows", M=M, rows=len(ctx.start.rows))
        if fa_v == 2 and self.bn != b1.fa2_kblock_n(D):
            return unsupported(node, "pinned kBlockN is not the FA2 head-dim dispatch rule's", bn=self.bn, rule=b1.fa2_kblock_n(D), D=D)
        # [R13 hopper] the attention Definition of the record's target (`frontend.targets.attention_spec`): Attention_v3{BN} on the
        # accepted target (Ampere step, FA2); Attention_v2{BN, DOT=<target step>, INV} otherwise — forced FA2 on Hopper: BN = fa2_kblock_n(D),
        # INV = Fa2InvSum; native FA3: BN = fa3_kblock_n(D, max_seqlen_q, NH/KVH) (the sm90 tile rule), INV = Fa3InvSum.
        kernel_name = "flash_fwd_splitkv_kernel" if fa_v == 2 else "FlashAttnFwdSm90"
        pattern_name = self.name if fa_v == 2 else "attention.fa3_fwd"
        # under FA3's per-iteration Check_inf chain (`fa3_construction`) a row's tile is part of its Definition: rows group by (T, the row's
        # index among its request's rows in this step); otherwise by T alone
        check_inf = fa_v == 3 and _targets().fa3_check_inf(tp)
        # FA2's per-iteration Check_inf chain also reads how many query rows the row's request has in this step
        check_inf2 = fa_v == 2 and _targets().fa2_check_inf(tp)
        first_pos: dict = {}
        for req, position in ctx.start.rows:
            first_pos[req] = min(int(position), first_pos.get(req, int(position)))
        own = collections.Counter(req for req, _ in ctx.start.rows)
        groups: dict[tuple[int, int | None, int | None], list[int]] = {}
        for i, (req, position) in enumerate(ctx.start.rows):
            T = int(position) + 1
            seq = ctx.start.seq_lens.get(req)
            if seq is not None and T > seq:
                return unsupported(node, "row position exceeds the request's KV length", row=i, request=req, T=T, seq_len=seq)
            groups.setdefault((T, int(position) - first_pos[req] if check_inf or check_inf2 else None, own[req] if check_inf2 else None),
                              []).append(i)
        computes: list[Compute] = []
        registered = getattr(ctx.profile, "registered_values", False)
        for (T, row_in_request, own_rows), rows in groups.items():
            try:
                slots = tuple(kv_slots(ctx, ctx.start.rows[i][0], T, layer=layer_s) for i in rows)
            except GeometryError as e:
                return unsupported(node, str(e), T=T, layer=layer_s)
            if any(s >= kcs.shape[0] for ss in slots for s in ss):
                return unsupported(node, "a KV slot is outside the cache", T=T, cache_slots=kcs.shape[0])
            try:
                # the launch's `max_seqlen_q` (FA3's tile rule reads it): the most query rows any ONE request has in this step
                max_seqlen_q = max(own.values())
                spec = _targets().attention_spec(tp, T, NH, KVH, D, max_seqlen_q, row=row_in_request, own_rows=own_rows,
                                                 scale_input=registered)
            except _targets().UnregisteredTarget as e:
                return unsupported(node, f"target has no registered attention binding: {e}")
            scale = (Supplied("registration", f"scale_log2_d{D}", 32, b1._scale_log2_bits(D)),) if registered else ()
            computes.append(Compute(spec, len(rows),
                                    (rows_operand(q, rows), GatherMany(kcs, slots), GatherMany(vcs, slots)) + scale,
                                    (rows_operand(out, rows),), (), hint=hint_of(node, str(layer) if isinstance(layer, str) else "attn"),
                                    kernels=(kernel_name,), attribution=tuple(ctx.start.rows[i] for i in rows),
                                    pattern=pattern_name))
        primary = dataclasses.replace(computes[0], consumed=tuple(sorted(node.ids())))
        return _group_compute(primary, before=tuple(before), also=tuple(computes[1:]))
