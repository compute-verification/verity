"""[GLM-5.3] The prefill a DeepSeek sparse attention layer serves by dense MHA (vLLM d9105ea80 `SparseMLACommonImpl.forward_mha`,
`FlashAttnPrefillBackend`), over the leaf `export_compat.dsa_leaves` exports in `dsa_indexer_attn`'s place (`vllm_iface`):

    dsa_dense_mha(positions, q_nope, q_pe, k_nope, v, kv_c, k_pe, index_k?, index_k_scale?, layer_name, topk, scale)
        -> attention rows [T, NH, DV]
        a prefill above the decode threshold of at most TOPK keys: the call's cache rows open the layer's history (`vllm_dsa`'s), and
        per query row i and head h `Attention_v4{T=i+1,NH=1,KVH=1,D,BN,DOT,INV,MASKED_FROM}` (FA3's causal varlen prefill, one head
        of it) over the call's own k, v rows 0..i, with q = [q_nope | q_pe] and k = [k_nope | k_pe] (k_pe shared by the heads)
        assembled as references; where FA3 splits the sequence, `AttentionSplit_v1{..., PER}` (`fa3_split`: each split's online
        softmax and end, FlashAttnFwdCombine); no logits or top-k
"""

from __future__ import annotations

from verity.ml.circuits.refs import Affine, Coll, Concat, PartLog, Strided, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_target_profile
from verity_vllm.program.frontend.rules.vllm_dsa import (_DENSE, BF16, DECODE_THRESHOLD, DECODE_THRESHOLD_DEFAULT, E4M3, F32,
                                                         _append_rows, _args, _decode_threshold, _f32, _missing,
                                                         _observed_sparse_mla, _open_history, _positions, _rows, _static_float,
                                                         _value)
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity.ml.definitions.registry import fa3_split

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


def takes_dense_mha(layer, T: int, rows: int, topk: int) -> bool:
    """Whether vLLM serves a sparse attention call of T query rows, its layer's history then `rows` rows long, by dense MHA
    (`DeepseekV32Attention._use_sparse_mha`, `use_dense_mha`): the layer has a prefill backend, `sparse_mla_force_mqa` is off, the
    metadata builder counts the call a prefill (T above its decode threshold at the layer's heads) and the prefill's rows have at
    most TOPK keys.  Masked MHA, the route past TOPK, is FA4's and not the H200's.  `export_compat.dsa_leaves` asks it."""
    cfg = getattr(getattr(layer, "_vllm_config", None), "attention_config", None)
    if getattr(layer, "prefill_backend", None) is None or getattr(cfg, "sparse_mla_force_mqa", False):
        return False
    return T > DECODE_THRESHOLD.get(int(layer.num_heads), DECODE_THRESHOLD_DEFAULT) and rows <= topk


class DsaDenseMhaRule(StateRule):
    """`vllm_iface.dsa_dense_mha` (`SparseMLACommonImpl.forward_mha` -> `FlashAttnPrefillBackend.run_prefill_new_tokens`): a prefill the
    layer serves by dense MHA, FA3's causal varlen self-attention over the call's own rows (`flash_attn_varlen_func`, `FlashAttnFwdSm90`
    at the 128 x 80 tile of head 256), per query row i and head h `Attention_v4{T=i+1, NH=1, KVH=1, D, BN, DOT, INV, MASKED_FROM}`
    over q[i, h] and the call's k, v rows 0..i of head h, the tile's `MASKED_FROM` of row i (`targets.fa3_masked_from`).  That is
    `Attention_v4{NH, KVH=NH}`'s own head h (`AttentionHead_v4` at the one tile every head shares), bound per head because a head's
    k row is two parts, [k_nope | k_pe]: one call per row would add 2 NH parts to its key sequence per row, past the alias window
    of the Program descriptor (`codec.ALIAS_WINDOW`), which would then write every call's whole key sequence.  Where FA3 runs the
    sequence as NS > 1 splits (`fa3_model.split_plan`), row i binds `AttentionSplit_v1{..., PER}` instead, PER the blocks a split of
    its tile (`fa3_split.split_per`): the row's splits, each ended with its partial output and lse, merged by FlashAttnFwdCombine.
    The indexer selects nothing (vLLM's `sparse_attn_indexer` returns before its logits when the batch has no decode token); the
    call's cache rows open the layer's history as `DsaIndexerAttn[vllm-dsa]`'s do, for the decode steps after it."""
    modes = frozenset({"profile"})
    info = RuleInfo("DsaDenseMha[vllm-dsa]", 1, "state", applies_to="vllm_iface.dsa_dense_mha(...) (" + ", ".join(_DENSE) + ")",
                    preconditions=("the layer's OBSERVED implementation is FlashAttnMLASparseImpl on a bf16 cache, without context parallelism; "
                                   "its prefill backend FlashAttnPrefillBackend on FA3 with v unpadded at the call's scale; sparse_mla_force_mqa off",
                                   "the target is compute capability 9.0 under FA3's check-inf-per-iteration construction (Attention_v4)",
                                   "the call is a prefill: T above the decode threshold at the observed query heads per rank (DECODE_THRESHOLD), "
                                   "T <= TOPK, and it opens the layer's history (no chunked context)",
                                   "q's and k's parts q_nope, k_nope [T, NH, P], q_pe [T, NH, R], k_pe [T, R], and v [T, NH, D] bf16 "
                                   "at D = P + R (one head dim, D registered by targets.fa3_kblock_n); kv_c [T, L] bf16; "
                                   "scale == f32(D^-0.5), the word Attention_v4 folds in",
                                   "FA3's split count of the sequence (`fa3_model.split_plan` at the target's SMs, or batch invariance's "
                                   "num_splits = 1) under a launch bound of at most 32 splits, the FlashAttnFwdCombine instantiation "
                                   "SparseMlaCombine_v1 states (kLogMaxSplits 5)",
                                   "no declared launch context other than the call's own rows (a varlen launch's split plan reads every "
                                   "sequence's length)",
                                   "the index operands both given (index_k [T, Dk] e4m3, index_k_scale [T] f32) or both None",
                                   "layer_name static and written from one module; not under registered_values"),
                    evidence_domain="structural; numerics: Attention_v4 and AttentionSplit_v1 (glm53_fa3_prefill_capture)")

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
        NH, P = (shape(ctx, a["q_nope"]) or (0, 0, 0))[1:3]
        R = (shape(ctx, a["k_pe"]) or (0, 0))[-1]
        D = P + R
        DV = (shape(ctx, a["v"]) or (0,))[-1]
        L = (shape(ctx, a["kv_c"]) or (0, 0))[-1]
        for r in (_value(ctx, a["q_nope"], BF16, (T, NH, P), "q_nope"), _value(ctx, a["q_pe"], BF16, (T, NH, R), "q_pe"),
                  _value(ctx, a["k_nope"], BF16, (T, NH, P), "k_nope"), _value(ctx, a["k_pe"], BF16, (T, R), "k_pe"),
                  _value(ctx, a["v"], BF16, (T, NH, DV), "v"), _value(ctx, a["kv_c"], BF16, (T, L), "kv_c")):
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
            _dense_splits(tp, T, NH, D)
        except ValueError as e:
            return f"numerics_unregistered: FA3's split count of a {T}-row prefill: {e}"
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
        _, NH, P = shape(ctx, a["q_nope"])
        D = P + shape(ctx, a["k_pe"])[-1]
        hist, P0 = _open_history(ctx, node, layer, pos, has)
        if P0:
            refuse("numerics_unregistered", node, mp, f"{layer!r}: a dense-MHA prefill of {T} rows after {P0} history rows runs over a "
                                                     f"chunked context (`_compute_prefill_context`, `merge_attn_states`), which no "
                                                     f"Definition states")
        _append_rows(ctx, hist, a, has)
        tp = active_target_profile()
        geom = TG.attention_geometry(tp, D, T, NH, NH)
        BM, ns = TG.fa3_tile_m(D, T, 1), _dense_splits(tp, T, NH, D)
        qn, qp, kn, kpe = (_rows(ctx, a[n], 16) for n in ("q_nope", "q_pe", "k_nope", "k_pe"))
        v = _rows(ctx, a["v"], 16)
        head = Array(1, Array(D, Value(16)))
        klogs, vlogs = [PartLog() for _ in range(NH)], [PartLog() for _ in range(NH)]
        outs, masked, pers = [], [], []
        for i in range(T):
            masked.append(TG.fa3_masked_from(tp, i + 1, NH, NH, D, T, i))
            split = {} if ns == 1 else {"PER": fa3_split.split_per(T, i, BM, geom["BN"], ns)}
            pers += split.values()
            fn = ctx.kind("attention_dot", node, T=i + 1, NH=1, KVH=1, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"],
                          MASKED_FROM=masked[-1], **split)
            rows = Array(i + 1, Array(1, Array(D, Value(16))))
            heads = []
            for hd in range(NH):
                klog, vlog = klogs[hd], vlogs[hd]
                for r in (kn[i][hd].refs, kpe[i].refs):
                    for p in (r.parts if isinstance(r, Concat) else (r,)):
                        klog.append(_run(p))
                vlog.append(_run(v[i][hd].refs))
                q = Coll(head, concat([qn[i][hd].refs, qp[i][hd].refs]))
                heads.append(ctx.call(fn, q, Coll(rows, klog.cut(len(klog))), Coll(rows, vlog.cut(i + 1)), node=node, rule=self,
                                      name=f"{h}[t={pos[i]},h={hd}]"))
            outs.append(Coll(Array(NH, Array(D, Value(16))), concat([o.refs for o in heads])))
        ctx.define(node, array_of(outs))
        kind = "Attention_v4" if ns == 1 else "AttentionSplit_v1"
        ctx.record(self, node, {"layer_name": layer, "positions": [pos[0], pos[-1]], "history_rows_after": T, "NH": NH, "D": D,
                                "binding": f"per head: {kind}{{NH=1, KVH=1}}",
                                "BN": geom["BN"], "BM": BM, "DOT": geom["DOT"].name, "INV": geom["INV"].name,
                                "fa_version": geom["fa_version"], "masked_from": [min(masked), max(masked)], "splits": ns,
                                "blocks_per_split": [min(pers), max(pers)] if pers else None, "scale": scale,
                                "TOPK": TOPK, "indexer": has, "selection": "none (dense MHA reads every row; the indexer returns early)",
                                "attention_impl_observed": (_OBSERVED["attention_impls"] or {}).get(layer)},
                   emitted=f"{kind} x{T * NH} (T=1..{T}, one call per head)")


def _run(p):
    """`p` as an `Affine` when it is one run of consecutive leaves (a head's row of a `Strided` view): the same references, in the
    form whose `PartLog` cuts the correspondence emitter walks once per log (`emit._arg_runs`), not once per call."""
    if isinstance(p, Strided) and p.inner_stride == 1 and p.count <= p.inner:
        return Affine(p.space, p.idx, p.base, p.count)
    return p


def _dense_splits(tp, T: int, NH: int, D: int) -> int:
    """The split count FA3 runs a B1 causal self-attention prefill of T rows at: 1 under batch invariance (`num_splits = 1`), else
    `fa3_model.split_plan`'s count for the sequence (vLLM passes `num_splits = 0`).  A launch bound past the combine's 32 lanes
    raises (`fa3_split.check_combine_bound`)."""
    if tp.batch_invariant:
        return 1
    from verity.ml.definitions.numerics import fa3_model
    static, dyn = fa3_model.split_plan([T], NH, NH, D, num_sms=int(tp.num_sms), Dv=D)
    if dyn[0] > 1:
        fa3_split.check_combine_bound(static)
    return dyn[0]


VLLM_DSA_DENSE_RULES = [DsaDenseMhaRule()]

__all__ = ["DsaDenseMhaRule", "PREFILL_BACKEND", "VLLM_DSA_DENSE_RULES"]
