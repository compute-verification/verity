"""The MLA KV cache write and attention read (two ops) of an `MLAAttention` layer: TRITON_MLA decode, FA2 prefill."""

from __future__ import annotations

import numpy as np

from verity.ir.refs import Coll, PartLog, array_of, concat, strided_view
from verity.ir.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.torch_frontend import _tensor_type
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_bindings.attention import _vllm_op
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_launch_context, active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value
from verity_vllm.program.frontend.rules.vllm_bindings.pins import _target_kinds, gemm_pins_for
from verity_vllm.program.registry.mla_word import BLOCK_N

BF16 = Value(16)

MLA_PINS = {"attention": "mla", "impl": "vllm.v1.attention.backends.mla.triton_mla.TritonMLAImpl", "backend": "TRITON_MLA",
            "kv_cache_dtype": "auto", "layer_kv_cache_dtype": "auto",
            "prefill_backend": "vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend", "vllm_flash_attn_version": 2,
            "batch_invariant_enabled": True, "dcp_world_size": 1, "is_sparse": False, "use_sparse": False, "use_pcp": False,
            "dcp_q_replicate": False, "q_pad_num_heads": None, "sliding_window": None, "non_causal_multi_token_decode": False,
            "is_amx_bmm_enabled": False, "is_aiter_triton_fp8_bmm_enabled": False, "is_aiter_triton_fp4_bmm_enabled": False,
            "kv_b_proj_quant_method": "vllm.model_executor.layers.linear.UnquantizedLinearMethod"}
"""What the MLA kinds model, every field read from the instantiated layer (`vllm_meta.mla_attention_impls`): `forward_mqa` on
`TritonMLAImpl` (one split under batch invariance) after the bf16 absorb bmm, and `forward_mha` on FA2's varlen prefill over the
call's own rows.  The FA version pin is the active target's (`registry.targets.check_flash_attn_registered`)."""


def _f32_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def mla_obs(layer_name: str) -> dict | None:
    impls = _OBSERVED["attention_impls"]
    obs = (impls or {}).get(layer_name)
    return obs if isinstance(obs, dict) and obs.get("attention") == "mla" else None


def _mla_pins_ok(layer_name: str) -> str | None:
    if _OBSERVED["attention_impls"] is None:
        return (f"attention implementation of {layer_name!r} was not observed at construction "
                "(observe_attention_impls(vllm_meta.attention_impls(model)) before derive) — refusing rather than assuming a backend")
    obs = mla_obs(layer_name)
    if obs is None:
        return f"layer {layer_name!r} has no observed MLAAttention implementation"
    from verity_vllm.program.registry import targets as T
    tp = active_target_profile()
    try:
        want_fav = T.check_flash_attn_registered(tp)
    except T.UnregisteredTarget as e:
        return f"numerics_unregistered: {e}"
    for k, want in MLA_PINS.items():
        want = want_fav if k == "vllm_flash_attn_version" else want
        got = obs.get(k, "<absent>")
        if got != want:
            return f"numerics_unregistered: MLA attention {k}={got!r} (observed on layer {layer_name!r}) differs from the pinned {want!r}"
    dims = {k: obs.get(k) for k in ("num_heads", "impl_num_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")}
    if any(not isinstance(v, int) or v <= 0 for v in dims.values()) or dims["num_heads"] != dims["impl_num_heads"]:
        return f"MLA layer {layer_name!r}: observed geometry {dims} is not one positive head count and dims"
    if len({obs.get("scale"), obs.get("impl_scale"), obs.get("prefill_scale")}) != 1:
        return f"MLA layer {layer_name!r}: the layer, decode and prefill scales differ ({obs.get('scale')}, {obs.get('impl_scale')}, {obs.get('prefill_scale')})"
    return None


class _MlaKvWrite:
    """The dummy result of `unified_mla_kv_cache_update`: not a value; the pending (kv_c, k_pe) write of one layer."""

    def __init__(self, layer_name: str, kv_c, k_pe, node) -> None:
        self.layer_name, self.kv_c, self.k_pe, self.node = layer_name, kv_c, k_pe, node

    def __repr__(self) -> str:
        return f"_MlaKvWrite({self.layer_name!r})"


def _rows(ctx, a) -> int:
    return int(shape(ctx, a)[0])


def _width(ctx, a) -> int:
    return int(np.prod(shape(ctx, a)[1:], dtype=np.int64))


class MlaKvCacheUpdateRule(StateRule):
    """`vllm::unified_mla_kv_cache_update(kv_c_normed[T, L], k_pe[T, 1, R], layer_name, kv_cache_dtype, k_scale) -> Tensor[0]`: the
    rows `[kv_c | k_pe]` this step appends to `layer_name`'s latent cache, recorded as the layer's pending write; the attention rule
    that consumes the dummy checks it appends the same rows.  No gates."""
    modes = frozenset({"profile"})
    info = RuleInfo("MlaKvCacheUpdate[vllm]", 1, "state",
                    applies_to="vllm.unified_mla_kv_cache_update(kv_c_normed, k_pe, layer_name: str, kv_cache_dtype: str, k_scale) -> Tensor",
                    preconditions=("kv_c_normed [T, L] and k_pe [T, 1, R] (or [T, R]) are bf16 values of equal row count",
                                   "layer_name is a static string; kv_cache_dtype is 'auto' (the bf16 cache: the row is stored as is)",
                                   "the dummy result is consumed only by this layer's unified_mla_attention_with_output"),
                    evidence_domain="structural: the latent cache write is made explicit by the MLA attention rule's history")

    def targets(self):
        return _vllm_op("unified_mla_kv_cache_update")

    def _args(self, node):
        a = list(node.args) + [None] * (5 - len(node.args))
        return a[0], a[1], a[2], node.kwargs.get("kv_cache_dtype", a[3])

    def applies(self, node, ctx):
        kv_c, k_pe, layer_name, kv_dtype = self._args(node)
        r = _row_value(ctx, kv_c, "kv_c") or _row_value(ctx, k_pe, "k_pe")
        if r:
            return r
        if isinstance(ctx.val(kv_c), Alloc) or isinstance(ctx.val(k_pe), Alloc):
            return "kv_c/k_pe is an unwritten allocation"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        if kv_dtype != "auto":
            return f"numerics_unregistered: kv_cache_dtype {kv_dtype!r}; only the bf16 cache ('auto') is registered"
        if _rows(ctx, kv_c) != _rows(ctx, k_pe):
            return f"kv_c {shape(ctx, kv_c)} vs k_pe {shape(ctx, k_pe)}"
        return True

    def emit(self, node, ctx):
        kv_c, k_pe, layer_name, _ = self._args(node)
        pending = ctx.state.setdefault("mla_pending", {})
        if layer_name in pending:
            refuse("side_input", node, ctx.module_path(node), f"two MLA cache writes to {layer_name!r} without an attention read between them")
        pending[layer_name] = _MlaKvWrite(layer_name, kv_c, k_pe, node)
        ctx.env[node] = pending[layer_name]
        ctx.record(self, node, {"layer_name": layer_name, "rows": _rows(ctx, kv_c), "width": _width(ctx, kv_c) + _width(ctx, k_pe)},
                   emitted="(pending MLA cache write; no gates)")


class MlaAttentionWithOutputRule(StateRule):
    """`vllm::unified_mla_attention_with_output(q[T, NH, DN+DR], kv_c_normed[T, L], k_pe[T, 1, DR], output![T, NH*DV], layer_name,
    output_scale?, output_block_scale?, kv_cache_dummy_dep?)`: the latent cache is implicit state keyed by `layer_name`, made
    explicit as the layer's history of rows `[kv_c | k_pe]`; `output` (an allocation) is rebound.  `W = kv_b_proj.weight`
    [NH*(DN+DV), L], read as views (`W_UK_T[h][p] = W[h*(DN+DV)+p]`, `W_UV[h] = W[h*(DN+DV)+DN : (h+1)*(DN+DV)]`):

    * one query row (vLLM's decode: query_len 1 at any history, `forward_mqa`): `MlaBmmWord{NH, DN, L}` of q_nope against W_UK_T,
      `TritonMlaDecodeWord{NH, P, L, DR, 32, 1, SCALE}` of `[ql_nope | q_pe]` over the P history rows, `MlaBmmWord{NH, L, DV}` against
      W_UV;
    * T > 1 rows opening the history (prefill without context, `forward_mha`): `Gemm{L, NH*(DN+DV)}` per row (`kv_b_proj`), then per
      row FA2's causal varlen attention `Attention_v5{P, NH, NH, DN+DR}` over this call's keys `[k_nope | k_pe]` and values (DV = DN+DR:
      FA2's v padding is the identity).

    Anything else (a prefill with context, registered values, another scale on the prefill) is refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("MlaAttentionWithOutput[vllm]", 1, "state",
                    applies_to="vllm.unified_mla_attention_with_output(q, kv_c_normed, k_pe, Tensor! output, layer_name: str, output_scale=None, output_block_scale=None, kv_cache_dummy_dep=None)",
                    preconditions=("the layer's OBSERVED MLAAttention implementation (decode impl + backend, prefill backend + FA version, kv dtype, "
                                   "batch invariance, and every flag that selects another forward path) equals MLA_PINS",
                                   "q [T, NH, DN+DR], kv_c_normed [T, L], k_pe [T, 1, DR] bf16 with the observed NH, L, DN, DR",
                                   "output is an unwritten allocation [T, NH*DV]; output_scale and output_block_scale are None",
                                   "kv_cache_dummy_dep is this layer's pending write of the SAME kv_c, k_pe rows",
                                   "layer_name is owned by exactly one module; kv_b_proj.weight is a parameter [NH*(DN+DV), L] of that module",
                                   "T == 1 (decode) or the call opens the history (prefill without context); DV == DN + DR on the prefill",
                                   "the prefill's scale is (DN+DR)^-0.5 (FA2's folded constant); the decode's scale is the observed f32 word",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: MlaBmmWord_v1 / TritonMlaDecodeWord_v1 (exact-model-tested twins of the MLA lane's Boolean "
                                    "MlaBmm_v1 / TritonMlaDecode_v1), Gemm_v2 and Attention_v5 (FA2 Check_inf, BN = fa2_kblock_n(D))")

    def targets(self):
        return _vllm_op("unified_mla_attention_with_output")

    def _args(self, node):
        a = list(node.args) + [None] * (8 - len(node.args))
        kw = node.kwargs
        return (a[0], a[1], a[2], a[3], a[4], kw.get("output_scale", a[5]), kw.get("output_block_scale", a[6]), kw.get("kv_cache_dummy_dep", a[7]))

    def applies(self, node, ctx):
        q, kv_c, k_pe, out, layer_name, oscale, obscale, dep = self._args(node)
        for nm, a in (("query", q), ("kv_c", kv_c), ("k_pe", k_pe)):
            r = _row_value(ctx, a, nm)
            if r:
                return r
            if isinstance(ctx.val(a), Alloc):
                return f"{nm} is an unwritten allocation"
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        r = _mla_pins_ok(layer_name)
        if r:
            return r
        if oscale is not None or obscale is not None:
            return "numerics_unregistered: output_scale / output_block_scale set (quantised output) — no registered kind"
        obs = mla_obs(layer_name)
        NH, L, DN, DR, DV = (obs[k] for k in ("num_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        T = _rows(ctx, q)
        want = {"query": (q, NH * (DN + DR)), "kv_c": (kv_c, L), "k_pe": (k_pe, DR), "output": (out, NH * DV)}
        for nm, (a, w) in want.items():
            if _rows(ctx, a) != T or _width(ctx, a) != w:
                return f"{nm} {shape(ctx, a)} is not [{T}, {w}] (observed NH={NH}, L={L}, DN={DN}, DR={DR}, DV={DV})"
        if dep is None:
            return "no kv_cache_dummy_dep: the latent cache write this read depends on is not in the graph"
        d = ctx.val(dep)
        if not isinstance(d, _MlaKvWrite):
            return f"kv_cache_dummy_dep is {type(d).__name__}, not a pending MLA cache write"
        if d.layer_name != layer_name:
            return f"kv_cache_dummy_dep belongs to {d.layer_name!r}, not {layer_name!r}"
        if ctx.storage_base(d.kv_c) is not ctx.storage_base(kv_c) or ctx.storage_base(d.k_pe) is not ctx.storage_base(k_pe):
            return "the pending MLA cache write's rows are not this call's kv_c / k_pe rows"
        return True

    def _kv_b_proj(self, ctx, node, NH, DN, DV, L):
        name = f"{ctx.stack_path(node)}.kv_b_proj.weight"
        leaf = next((w for w in ctx.weights if w.name == name or name in w.aliases), None)
        if leaf is None:
            refuse("hidden_state", node, ctx.module_path(node), f"no parameter {name!r}: the MLA layer's kv_b_proj weight is not a field of the weights record")
        if tuple(leaf.shape) != (NH * (DN + DV), L) or leaf.dtype != "bfloat16":
            refuse("numerics_unregistered", node, ctx.module_path(node),
                   f"{leaf.name} is {leaf.dtype} {list(leaf.shape)}, not bfloat16 [{NH * (DN + DV)}, {L}]")
        if leaf.name not in ctx.weight_roles:
            ctx.weight_roles[leaf.name] = {"rule": self.rule_id, "role": "mla.kv_b_proj.weight", "NH": NH, "DN": DN, "DV": DV, "L": L}
        return leaf, ctx.weights_coll[leaf.index].reshape(Array(NH * (DN + DV), Array(L, BF16)))

    def emit(self, node, ctx):
        q, kv_c, k_pe, out, layer_name, _, _, dep = self._args(node)
        mp = ctx.module_path(node)
        obs = mla_obs(layer_name)
        NH, L, DN, DR, DV = (obs[k] for k in ("num_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        D, H = DN + DR, DN + DV
        T = _rows(ctx, q)
        qrows = ctx.coll(q).reshape(Array(T, Array(NH, Array(D, BF16))))
        kvc = ctx.coll(kv_c).reshape(Array(T, Array(L, BF16)))
        kpe = ctx.coll(k_pe).reshape(Array(T, Array(DR, BF16)))
        owner = ctx.state.setdefault("mla_owner", {})
        if layer_name in owner and owner[layer_name] != mp:
            refuse("side_input", node, mp, f"MLA cache {layer_name!r} is written from two different modules ({owner[layer_name]} and {mp})")
        owner.setdefault(layer_name, mp)
        ctx.state.setdefault("mla_pending", {}).pop(layer_name, None)
        hist = ctx.state.setdefault("mla_history", {}).setdefault(layer_name, PartLog())
        P0 = len(hist)
        for i in range(T):
            hist.append(concat([kvc[i].refs, kpe[i].refs]))
        ctx.check_dim("T", T, node)
        ctx.check_dim("CONTEXT", P0 + T, node)
        if ctx.registered_values:
            refuse("numerics_unregistered", node, mp, "registered_values: the MLA kinds take the scale as a static; no value-reading twin is registered")
        accepted, dot = _target_kinds()
        if accepted:
            refuse("numerics_unregistered", node, mp, "the MLA kinds are bound with a target's k16 DOT step; the accepted (Ampere) kinds have none")
        _leaf, W = self._kv_b_proj(ctx, node, NH, DN, DV, L)
        scale = float(obs["scale"])
        hint = ctx.hint_name(node)
        if T == 1:
            path, (y, detail) = "decode", self._decode(ctx, node, hint, qrows[0], hist, P0 + 1, W, NH, L, DN, DR, DV, H, scale, dot)
        elif P0 == 0:
            path, (y, detail) = "prefill", self._prefill(ctx, node, mp, hint, qrows, kvc, kpe, W, T, NH, L, DN, DR, DV, D, H, scale, dot)
        else:
            refuse("numerics_unregistered", node, mp, f"layer {layer_name!r}: {T} query rows after {P0} history rows is a prefill with context "
                                                      f"(chunked prefill: `_compute_prefill_context` + merge_attn_states); no kind is registered")
        ctx.rebind(out, y.reshape(_tensor_type(tuple(shape(ctx, out)), 16)), self, what="output")
        ctx.record(self, node, {"layer_name": layer_name, "path": path, "positions": [P0, P0 + T - 1], "history_rows_after": P0 + T,
                                "NH": NH, "L": L, "DN": DN, "DR": DR, "DV": DV, "kv_b_proj": _leaf.name, "kv_write_consumed": True,
                                "assumed_causal": True, "mla_pins": {**MLA_PINS, "vllm_flash_attn_version": active_target_profile().effective_flash_attn_version()},
                                "attention_impl_observed": {k: obs.get(k) for k in (*MLA_PINS, "scale", "prefill_requires_v_padding")}, **detail},
                   emitted=f"{path}: {detail['emitted']}")

    def _decode(self, ctx, node, hint, q0, hist, P, W, NH, L, DN, DR, DV, H, scale, dot):
        x = Coll(Array(NH, Array(DN, BF16)), concat([q0[h][0:DN].refs for h in range(NH)]))
        w_uk = Coll(Array(NH, Array(L, Array(DN, BF16))), concat([strided_view(W, h * H * L, DN, 1, L, L * DN) for h in range(NH)]))
        absorb = ctx.kind("mla_bmm", node, NHEADS=NH, K=DN, N=L, DOT=dot)
        ql = ctx.call(absorb, x, w_uk, node=node, rule=self, name=f"{hint}/absorb")
        mqa_q = Coll(Array(NH, Array(L + DR, BF16)), concat([r for h in range(NH) for r in (ql[h].refs, q0[h][DN:DN + DR].refs)]))
        kv = Coll(Array(P, Array(L + DR, BF16)), hist.cut(P))
        SCALE = _f32_bits(scale)
        dec = ctx.kind("mla_decode", node, H=NH, T=P, DL=L, DR=DR, SPLITS=1, SM_SCALE=SCALE, DOT=dot)
        o = ctx.call(dec, mqa_q, kv, node=node, rule=self, name=f"{hint}[t={P - 1}]")
        w_uv = Coll(Array(NH, Array(DV, Array(L, BF16))), concat([W[h * H + DN:(h + 1) * H].refs for h in range(NH)]))
        vup = ctx.kind("mla_bmm", node, NHEADS=NH, K=L, N=DV, DOT=dot)
        y = ctx.call(vup, o, w_uv, node=node, rule=self, name=f"{hint}/v_up")
        detail = {"decode": {"T": P, "BN": BLOCK_N, "SPLITS": 1, "SCALE": f"0x{SCALE:08X}", "DOT": dot.name,
                             "absorb": absorb.id, "attention": dec.id, "v_up": vup.id},
                  "emitted": f"{absorb.id} + {dec.id} + {vup.id}"}
        return y, detail

    def _prefill(self, ctx, node, mp, hint, qrows, kvc, kpe, W, T, NH, L, DN, DR, DV, D, H, scale, dot):
        from verity_vllm.program.registry import targets as TG
        if DV != D:
            refuse("numerics_unregistered", node, mp, f"v_head_dim {DV} != qk_head_dim {D}: FA2's v padding is not the identity; no kind is registered")
        if scale != float(D ** -0.5):
            refuse("numerics_unregistered", node, mp, f"MLA prefill scale {scale!r} != D^-0.5 = {D ** -0.5!r} (D={D}): the FA2 kinds fold softmax_scale = D^-0.5")
        tp = active_target_profile()
        if not TG.fa2_check_inf(tp):
            refuse("numerics_unregistered", node, mp, f"the MLA prefill binds FA2's per-iteration Check_inf chain (Attention_v5); "
                                                      f"{tp.tensor_core_family} under FA{tp.effective_flash_attn_version()} does not register it")
        try:
            launch_sq, launch_src = TG.launch_seqlen_q(T, 0, active_launch_context())
        except TG.LaunchContextConflict:
            launch_sq, launch_src = T, "own rows (context conflict ignored: the FA2 tile does not read seqlen_q)"
        except TG.LaunchContextInvalid as e:
            refuse("numerics_unregistered", node, mp, f"MLA prefill: invalid declared launch context -- {e}", launch_context="invalid", own_rows=T)
        geom = TG.attention_geometry(tp, D, launch_sq, NH, NH)
        gemm = ctx.kind("gemm_dot", node, K=L, N=NH * H, DOT=dot)
        klog, vlog, outs = PartLog(), PartLog(), []
        for i in range(T):
            kvn = ctx.call(gemm, kvc[i], W, node=node, rule=self, name=f"{hint}/kv_b_proj[{i}]")
            klog.append(concat([r for h in range(NH) for r in (kvn[h * H:h * H + DN].refs, kpe[i].refs)]))
            vlog.append(concat([kvn[h * H + DN:(h + 1) * H].refs for h in range(NH)]))
        masked = []
        for i in range(T):
            P = i + 1
            mf = TG.fa2_masked_from(tp, P, D, launch_sq, i, T)
            masked.append(mf)
            fn = ctx.kind("attention_dot", node, T=P, NH=NH, KVH=NH, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"], FA2_MASKED_FROM=mf)
            kcat = Coll(Array(P, Array(NH, Array(D, BF16))), klog.cut(P))
            vcat = Coll(Array(P, Array(NH, Array(D, BF16))), vlog.cut(P))
            outs.append(ctx.call(fn, qrows[i], kcat, vcat, node=node, rule=self, name=f"{hint}[t={P - 1}]"))
        pins, src = gemm_pins_for(T, NH * H, L)
        detail = {"prefill": {"kv_b_proj_gemm": gemm.id, "kv_b_proj_pins": pins, "kv_b_proj_pin_source": src,
                              "kv_b_proj_launch": "inside the op (not observed in the export): the pins are the target's launch geometry for this shape",
                              "BN": geom["BN"], "DOT": geom["DOT"].name, "INV": geom["INV"].name, "fa_version": geom["fa_version"],
                              "launch_seqlen_q": launch_sq, "launch_seqlen_q_source": launch_src, "fa2_masked_from": [masked[0], masked[-1]],
                              "kernel": "flash_fwd_kernel (num_splits 1, unpaged K/V: the non-split dispatch; kBlockM 64, kBlockN fa2_kblock_n(D))"},
                  "emitted": f"{gemm.id} x{T} + Attention_v5 x{T}"}
        return array_of(outs), detail
