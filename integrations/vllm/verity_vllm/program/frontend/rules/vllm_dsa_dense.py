"""[GLM-5.3] The prefill a DeepSeek sparse attention layer serves by dense MHA (vLLM d9105ea80 `SparseMLACommonImpl.forward_mha`,
`FlashAttnPrefillBackend`), over the leaf `export_compat.dsa_leaves` exports in `dsa_indexer_attn`'s place (`vllm_iface`):

    dsa_dense_mha(positions, q, k, v, kv_c, k_pe, index_k?, index_k_scale?, layer_name, topk, scale) -> attention rows [T, NH, DV]
        a prefill above the decode threshold of at most TOPK keys: the call's cache rows open the layer's history (`vllm_dsa`'s), and
        per query row i `Attention_v4{T=i+1,NH,KVH=NH,D,BN,DOT,INV,MASKED_FROM}` (FA3's causal varlen prefill) over the call's own k, v
        rows 0..i; no logits or top-k
"""

from __future__ import annotations

from verity.ml.circuits.refs import Coll, PartLog, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_target_profile
from verity_vllm.program.frontend.rules.vllm_dsa import (BF16, E4M3, F32, _append_rows, _args, _decode_threshold, _f32, _missing,
                                                         _observed_sparse_mla, _open_history, _positions, _rows, _static_float,
                                                         _value)
from verity_vllm.program.frontend.rules.vllm_iface import iface

_DENSE = ("positions", "q", "k", "v", "kv_c", "k_pe", "index_k", "index_k_scale", "layer_name", "topk", "scale")
PREFILL_BACKEND = "vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"


def _observed_prefill_backend(layer_name: str, scale: float) -> str | None:
    """None when the layer's observed dense-MHA prefill backend is `FlashAttnPrefillBackend` running FA3 with v unpadded at the
    call's scale, and `sparse_mla_force_mqa` is off."""
    obs = _OBSERVED["attention_impls"][layer_name]
    if obs.get("prefill_backend") != PREFILL_BACKEND:
        return (f"numerics_unregistered: {layer_name!r}'s dense-MHA prefill backend is {obs.get('prefill_backend')!r}; Attention_v4 "
                f"states {PREFILL_BACKEND}'s FA3")
    if obs.get("sparse_mla_force_mqa") is not False:
        return f"{layer_name!r} observes sparse_mla_force_mqa={obs.get('sparse_mla_force_mqa')!r}, which serves every call by sparse MQA"
    if obs.get("prefill_fa_version") != 3:
        return f"numerics_unregistered: {layer_name!r}'s prefill backend runs FA version {obs.get('prefill_fa_version')!r}; Attention_v4 is FA3"
    if obs.get("prefill_requires_v_padding") is not False:
        return f"numerics_unregistered: {layer_name!r}'s prefill backend pads v to the qk head dim (requires_v_padding)"
    ps = obs.get("prefill_scale")
    if not _static_float(ps) or _f32(ps) != _f32(scale):
        return f"{layer_name!r}'s prefill backend scale {ps!r} is not the call's {scale!r}"
    return None


class DsaDenseMhaRule(StateRule):
    """`vllm_iface.dsa_dense_mha` (`SparseMLACommonImpl.forward_mha` -> `FlashAttnPrefillBackend.run_prefill_new_tokens`): a prefill the
    layer serves by dense MHA, FA3's causal varlen self-attention over the call's own rows (`flash_attn_varlen_func`, `FlashAttnFwdSm90`
    at the 128 x 80 tile of head 256), per query row i `Attention_v4{T=i+1, NH, KVH=NH, D, BN, DOT, INV, MASKED_FROM}` over q[i] and the
    call's k, v rows 0..i, the tile's `MASKED_FROM` of row i (`targets.fa3_masked_from`).  The indexer selects nothing (vLLM's
    `sparse_attn_indexer` returns before its logits when the batch has no decode token); the call's cache rows open the layer's history
    as `DsaIndexerAttn[vllm-dsa]`'s do, for the decode steps after it."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaDenseMha[vllm-dsa]", 1, "state", applies_to="vllm_iface.dsa_dense_mha(...) (" + ", ".join(_DENSE) + ")",
                    preconditions=("the layer's OBSERVED implementation is FlashAttnMLASparseImpl on a bf16 cache, without context parallelism; "
                                   "its prefill backend FlashAttnPrefillBackend on FA3 with v unpadded at the call's scale; sparse_mla_force_mqa off",
                                   "the target is compute capability 9.0 under FA3's check-inf-per-iteration construction (Attention_v4)",
                                   "the call is a prefill: T above the decode threshold at the observed query heads per rank (DECODE_THRESHOLD), "
                                   "T <= TOPK, and it opens the layer's history (no chunked context)",
                                   "q, k [T, NH, D] and v [T, NH, D] bf16 (one head dim, D registered by targets.fa3_kblock_n); "
                                   "kv_c [T, L], k_pe [T, R] bf16; scale == f32(D^-0.5), the word Attention_v4 folds in",
                                   "FA3 runs the sequence as one split (`fa3_model.split_plan` at the target's SMs, or batch invariance's "
                                   "num_splits = 1): FlashAttnFwdCombine has no Definition",
                                   "no declared launch context other than the call's own rows (a varlen launch's split plan reads every "
                                   "sequence's length)",
                                   "the index operands both given (index_k [T, Dk] e4m3, index_k_scale [T] f32) or both None",
                                   "layer_name static and written from one module; not under registered_values"),
                    evidence_domain="structural; numerics: Attention_v4 (glm53_fa3_prefill_capture)")

    def targets(self):
        return {iface.dsa_dense_mha.default}

    def applies(self, node, ctx):
        from verity_vllm.program.frontend import targets as TG
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_launch_context
        a = _args(node, _DENSE)
        layer, topk, scale = a["layer_name"], a["topk"], a["scale"]
        if not isinstance(layer, str) or not isinstance(topk, int) or topk < 1 or not _static_float(scale):
            return "layer_name, topk, scale are not static"
        if (r := _missing(ctx, ["attention_dot"])):
            return r
        if ctx.registered_values:
            return "numerics_unregistered: Attention_v4 folds its scale; the registered-values twin is not bound here"
        if (r := _observed_sparse_mla(layer)) or (r := _observed_prefill_backend(layer, float(scale))):
            return r
        tp = active_target_profile()
        if tp.compute_capability != (9, 0) or not TG.fa3_check_inf(tp):
            return (f"numerics_unregistered: Attention_v4 is FA3 at compute capability 9.0 under check-inf-per-iteration; the target is "
                    f"{tp.capability_str} at FA{tp.effective_flash_attn_version()}")
        pos = _positions(ctx, a["positions"])
        if isinstance(pos, str):
            return pos
        T = len(pos)
        limit = _decode_threshold(layer)
        if isinstance(limit, str):
            return limit
        if T <= limit:
            return f"a call of {T} query rows is a decode (decode threshold {limit}), which takes the top-k MQA path"
        if T > topk:
            return f"a prefill of {T} rows reaches past TOPK {topk} keys, which vLLM serves by sparse MQA, not dense MHA"
        NH, D = (shape(ctx, a["q"]) or (0, 0, 0))[1:3]
        DV = (shape(ctx, a["v"]) or (0,))[-1]
        L = (shape(ctx, a["kv_c"]) or (0, 0))[-1]
        R = (shape(ctx, a["k_pe"]) or (0, 0))[-1]
        for r in (_value(ctx, a["q"], BF16, (T, NH, D), "q"), _value(ctx, a["k"], BF16, (T, NH, D), "k"),
                  _value(ctx, a["v"], BF16, (T, NH, DV), "v"), _value(ctx, a["kv_c"], BF16, (T, L), "kv_c"),
                  _value(ctx, a["k_pe"], BF16, (T, R), "k_pe")):
            if r:
                return r
        if DV != D:
            return f"numerics_unregistered: Attention_v4 states one head dim for q, k and v; v head {DV} vs qk head {D}"
        if _f32(scale) != _f32(D ** -0.5):
            return f"numerics_unregistered: scale {scale!r} is not f32(D^-0.5) at D={D}, the word Attention_v4 folds in"
        try:
            TG.attention_geometry(tp, D, T, NH, NH)
            launch_sq, _ = TG.launch_seqlen_q(T, 0, active_launch_context())
        except (TG.UnregisteredTarget, TG.LaunchContextConflict, TG.LaunchContextInvalid) as e:
            return f"numerics_unregistered: {e}"
        if launch_sq != T:
            return (f"numerics_unregistered: a declared launch context of max_seqlen_q {launch_sq} at a {T}-row prefill: FA3's split "
                    f"plan of a varlen launch reads every sequence's length, which the context does not declare")
        try:
            ns = _dense_splits(tp, T, NH, D)
        except ValueError as e:
            return f"numerics_unregistered: FA3's split count of a {T}-row prefill: {e}"
        if ns > 1:
            return (f"numerics_unregistered: FA3 runs a {T}-row prefill at {NH} heads, head {D} on {tp.num_sms} SMs as {ns} splits "
                    f"(fa3_model.split_plan) combined by FlashAttnFwdCombine, which no Definition states")
        ix = (a.get("index_k"), a.get("index_k_scale"))
        if (ix[0] is None) != (ix[1] is None):
            return "the index operands are neither both given nor both None"
        if ix[0] is not None:
            for r in (_value(ctx, ix[0], E4M3, (T, None), "index_k"), _value(ctx, ix[1], F32, (T,), "index_k_scale")):
                if r:
                    return r
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets as TG
        a = _args(node, _DENSE)
        has = a.get("index_k") is not None
        layer, TOPK, scale = a["layer_name"], a["topk"], float(a["scale"])
        mp, h = ctx.module_path(node), ctx.hint_name(node)
        pos = ctx.static_index(a["positions"]).flat()
        T = len(pos)
        _, NH, D = shape(ctx, a["q"])
        hist, P0 = _open_history(ctx, node, layer, pos, has)
        if P0:
            refuse("numerics_unregistered", node, mp, f"{layer!r}: a dense-MHA prefill of {T} rows after {P0} history rows runs over a "
                                                     f"chunked context (`_compute_prefill_context`, `merge_attn_states`), which no "
                                                     f"Definition states")
        _append_rows(ctx, hist, a, has)
        tp = active_target_profile()
        geom = TG.attention_geometry(tp, D, T, NH, NH)
        q, k, v = _rows(ctx, a["q"], 16), _rows(ctx, a["k"], 16), _rows(ctx, a["v"], 16)
        klog, vlog = PartLog(), PartLog()
        outs, masked = [], []
        for i in range(T):
            klog.append(k[i].refs)
            vlog.append(v[i].refs)
            masked.append(TG.fa3_masked_from(tp, i + 1, NH, NH, D, T, i))
            fn = ctx.kind("attention_dot", node, T=i + 1, NH=NH, KVH=NH, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"],
                          MASKED_FROM=masked[-1])
            rows = Array(i + 1, Array(NH, Array(D, Value(16))))
            outs.append(ctx.call(fn, q[i], Coll(rows, klog.cut(i + 1)), Coll(rows, vlog.cut(i + 1)), node=node, rule=self,
                                 name=f"{h}[t={pos[i]}]"))
        ctx.define(node, array_of(outs))
        ctx.record(self, node, {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_rows_after": T, "NH": NH, "D": D,
                                "BN": geom["BN"], "BM": TG.fa3_tile_m(D, T, 1), "DOT": geom["DOT"].name, "INV": geom["INV"].name,
                                "fa_version": geom["fa_version"], "masked_from": [min(masked), max(masked)], "splits": 1, "scale": scale,
                                "TOPK": TOPK, "indexer": has, "selection": "none (dense MHA reads every row; the indexer returns early)",
                                "attention_impl_observed": (_OBSERVED["attention_impls"] or {}).get(layer)},
                   emitted=f"Attention_v4 x{T} (T=1..{T})")


def _dense_splits(tp, T: int, NH: int, D: int) -> int:
    """The split count FA3 runs a B1 causal self-attention prefill of T rows at: 1 under batch invariance (`num_splits = 1`), else
    `fa3_model.split_plan`'s count for the sequence (vLLM passes `num_splits = 0`)."""
    if tp.batch_invariant:
        return 1
    from verity_vllm.program.kernels import fa3_model
    return fa3_model.split_plan([T], NH, NH, D, num_sms=int(tp.num_sms), Dv=D)[1][0]


VLLM_DSA_DENSE_RULES = [DsaDenseMhaRule()]

__all__ = ["DsaDenseMhaRule", "PREFILL_BACKEND", "VLLM_DSA_DENSE_RULES"]
