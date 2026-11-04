"""The KV cache write and the attention read (two ops)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import attention_scale, kv_logs, shape
from verity_vllm.program.frontend.rules.vllm_bindings import kv_prefix as KP
from verity_vllm.program.frontend.rules.vllm_bindings.observations import (ATTN_PINS, _OBSERVED, _attn_pins_ok, active_launch_context,
                                                                           active_target_profile, attn_kind_for, attn_lookup_key)
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value


# -- KV cache write + attention read (two ops) -----------------------------------------------------------------------------


class _KvWrite:
    """The dummy result of `unified_kv_cache_update`: not a value; the pending (k, v) write of one layer."""

    def __init__(self, layer_name: str, k, v, node) -> None:
        self.layer_name, self.k, self.v, self.node = layer_name, k, v, node

    def __repr__(self) -> str:
        return f"_KvWrite({self.layer_name!r})"


def _vllm_op(name: str) -> set:
    try:
        return {getattr(torch.ops.vllm, name).default}
    except Exception:  # noqa: BLE001 — vLLM not importable (laptop): the rule has no targets
        return set()


class KvCacheUpdateRule(StateRule):
    """`vllm::unified_kv_cache_update(key[T,KVH,D], value[T,KVH,D], layer_name) -> Tensor[0]`: the rows this step appends to
    `layer_name`'s cache. Recorded as the layer's pending write; the attention rule that consumes the dummy checks it appends
    the same rows. No gates."""
    modes = frozenset({"profile"})
    info = RuleInfo("KvCacheUpdate[vllm]", 1, "state", applies_to="vllm.unified_kv_cache_update(key, value, layer_name: str) -> Tensor",
                    preconditions=("key, value are bf16 values [T, KVH, D] (or [T, KVH*D]) of equal shape", "layer_name is a static string",
                                   "the dummy result is consumed only by this layer's unified_attention_with_output"),
                    evidence_domain="structural: the KV write is made explicit by the attention rule's history")

    def targets(self):
        return _vllm_op("unified_kv_cache_update")

    def applies(self, node, ctx):
        k, v, layer_name = node.args[:3]
        r = _row_value(ctx, k, "key") or _row_value(ctx, v, "value")
        if r:
            return r
        if isinstance(ctx.val(k), Alloc) or isinstance(ctx.val(v), Alloc):
            return "key/value is an unwritten allocation"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        if shape(ctx, k) != shape(ctx, v):
            return f"key {shape(ctx, k)} vs value {shape(ctx, v)}"
        return True

    def emit(self, node, ctx):
        k, v, layer_name = node.args[:3]
        pending = ctx.state.setdefault("kv_pending", {})
        if layer_name in pending:
            refuse("side_input", node, ctx.module_path(node), f"two KV writes to {layer_name!r} without an attention read between them")
        pending[layer_name] = _KvWrite(layer_name, k, v, node)
        ctx.env[node] = pending[layer_name]
        ctx.record(self, node, {"layer_name": layer_name, "rows": shape(ctx, k)[0]}, emitted="(pending KV write; no gates)")


class UnifiedAttentionWithOutputRule(StateRule):
    """`vllm::unified_attention_with_output(q, k, v, output!, layer_name, output_scale?, output_block_scale?, kv_cache_dummy_dep?)`:
    the KV cache is implicit state keyed by `layer_name` (vLLM's forward context). Made explicit as in `UnifiedAttention[vllm]`:
    the rows k, v of this call extend the layer's history at positions len(history)..; the row at global position P reads
    `K[0:P+1], V[0:P+1]` as `Attention{T=P+1,NH,KVH,D}`; `output` (an allocation) is rebound to the attention rows.
    Causality is the op's contract (decoder attention), recorded as an assumed precondition."""
    modes = frozenset({"profile"})
    info = RuleInfo("UnifiedAttentionWithOutput[vllm]", 2, "state",
                    applies_to="vllm.unified_attention_with_output(query, key, value, Tensor! output, layer_name: str, output_scale=None, output_block_scale=None, kv_cache_dummy_dep=None)",
                    preconditions=("the layer's OBSERVED attention implementation (instantiated `Attention.impl` class, backend, FA version, window, soft cap, "
                                   "alibi, sinks, kv dtype, batch_invariant_enabled) equals ATTN_PINS — derived from the model instance, never assumed",
                                   "q [T, NH, D], k, v [T, KVH, D] bf16 (or flat rows with config D); NH % KVH == 0",
                                   "output is an unwritten allocation of q's shape", "layer_name is a static string owned by exactly one module",
                                   "output_scale and output_block_scale are None (no fp8 output quantisation)",
                                   "kv_cache_dummy_dep, if present, is this layer's pending write of the SAME k, v rows",
                                   "assumed (op contract): causal attention over the history + this call's rows"),
                    evidence_domain="structural; numerics: the vocabulary's attention kind per position (FA2 causal GQA)")

    def targets(self):
        return _vllm_op("unified_attention_with_output")

    def _args(self, node):
        a = list(node.args) + [None] * (8 - len(node.args))
        kw = node.kwargs
        return (a[0], a[1], a[2], a[3], a[4], kw.get("output_scale", a[5]), kw.get("output_block_scale", a[6]), kw.get("kv_cache_dummy_dep", a[7]))

    def impl_refusal(self, layer_name: str) -> str | None:
        """Why the layer's observed implementation is not the one this rule binds (None: it is)."""
        return _attn_pins_ok((_OBSERVED["attention_impls"] or {}).get(layer_name) if _OBSERVED["attention_impls"] is not None else None, layer_name)

    def applies(self, node, ctx):
        q, k, v, out, layer_name, oscale, obscale, dep = self._args(node)
        for nm, a in (("query", q), ("key", k), ("value", v)):
            r = _row_value(ctx, a, nm)
            if r:
                return r
            if isinstance(ctx.val(a), Alloc):
                return f"{nm} is an unwritten allocation"
        if not isinstance(ctx.val(out), (Alloc, Coll)):
            return "output is not an allocation/value"
        if not isinstance(layer_name, str):
            return "layer_name is not a static string"
        r = self.impl_refusal(layer_name)
        if r:
            return r
        if oscale is not None or obscale is not None:
            return "numerics_unregistered: output_scale / output_block_scale set (fp8 output) — no registered kind"
        qs, ks, vs, os_ = shape(ctx, q), shape(ctx, k), shape(ctx, v), shape(ctx, out)
        if qs[0] != ks[0] or ks != vs:
            return f"row counts differ q{qs} k{ks} v{vs}"
        if os_ != qs:
            return f"output {os_} vs query {qs}"
        if dep is not None:
            d = ctx.val(dep)
            if not isinstance(d, _KvWrite):
                return f"kv_cache_dummy_dep is {type(d).__name__}, not a pending KV write"
            if d.layer_name != layer_name:
                return f"kv_cache_dummy_dep belongs to {d.layer_name!r}, not {layer_name!r}"
            if ctx.storage_base(d.k) is not ctx.storage_base(k) or ctx.storage_base(d.v) is not ctx.storage_base(v):
                return "the pending KV write's rows are not this call's key/value rows"
        return True

    def extend_history(self, node, ctx) -> SimpleNamespace:
        """Append the call's k, v rows to its layer's history: the query rows `qrows` [T][NH][D] (`qs` q's shape), the history's
        logs `klog`, `vlog` with the call's rows at positions P0..P0+T-1, whether the call is the `prefill` (it opens the history,
        after any cached prefix), and the module path `mp`."""
        q, k, v, out, layer_name, _, _, dep = self._args(node)
        qc, kc, vc = ctx.coll(q), ctx.coll(k), ctx.coll(v)
        D = ctx.config.get("D") or ctx.allowed_dims.get("D")
        qs, ks = shape_of(qc.type)[0], shape_of(kc.type)[0]
        T = qs[0]
        if len(qs) == 3:
            NH, D_ = qs[1], qs[2]
        else:
            if D is None:
                refuse("shape", node, ctx.module_path(node), "head dim D is needed to split flat q/k/v rows: pass config={'D': ...} or 3-D q/k/v")
            NH, D_ = qs[1] // D, D
        if D is not None and D_ != D:
            refuse("shape", node, ctx.module_path(node), f"head dim {D_} vs config D={D}")
        D = D_
        KVH = (ks[1] * (ks[2] if len(ks) > 2 else 1)) // D
        if NH % KVH:
            refuse("shape", node, ctx.module_path(node), f"NH={NH} not a multiple of KVH={KVH}")
        qrows = qc.reshape(Array(T, Array(NH, Array(D, Value(16)))))
        krows = kc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        vrows = vc.reshape(Array(T, Array(KVH, Array(D, Value(16)))))
        hist = ctx.state.setdefault("kv_history", {})
        owner = ctx.state.setdefault("kv_owner", {})
        mp = ctx.module_path(node)
        if layer_name in owner and owner[layer_name] != mp:
            refuse("side_input", node, mp, f"KV history {layer_name!r} is written from two different modules ({owner[layer_name]} and {mp})")
        owner.setdefault(layer_name, mp)
        pending = ctx.state.setdefault("kv_pending", {})
        if dep is not None:
            pending.pop(layer_name, None)
        elif layer_name in pending:
            refuse("side_input", node, mp, f"a pending KV write to {layer_name!r} was not consumed by this attention call")
        opened = layer_name not in hist
        krow_list, vrow_list = hist.setdefault(layer_name, ([], []))
        # a cached prefix (KvPrefix[vllm]) opens the history with its rows: this call is still the prefill, from position C
        cached = ctx.state.setdefault("kv_cached_rows", {})
        if opened:
            cached[layer_name] = KP.seed(ctx, node, layer_name, krow_list, vrow_list, KVH, D)
        P0 = len(krow_list)
        prefill = P0 == cached.get(layer_name, 0)
        if not prefill:
            KP.require_consumed(ctx, node)
        for i in range(T):
            krow_list.append(krows[i])
            vrow_list.append(vrows[i])
        klog, vlog = kv_logs(ctx, layer_name, krow_list, vrow_list)
        ctx.check_dim("T", T, node)
        ctx.check_dim("CONTEXT", P0 + T, node)
        return SimpleNamespace(qs=qs, T=T, NH=NH, D=D, KVH=KVH, qrows=qrows, klog=klog, vlog=vlog, P0=P0, prefill=prefill, mp=mp)

    def emit(self, node, ctx):
        q, k, v, out, layer_name, _, _, dep = self._args(node)
        h = self.extend_history(node, ctx)
        qs, T, NH, D, KVH, qrows, klog, vlog, P0, prefill, mp = h.qs, h.T, h.NH, h.D, h.KVH, h.qrows, h.klog, h.vlog, h.P0, h.prefill, h.mp
        outs = []
        obs_sel = (_OBSERVED["attention_impls"] or {}).get(layer_name, {})
        role, extra, window = attn_kind_for(obs_sel)
        # both registered chains fold `softmax_scale = float(D ** -0.5)` into their constants (`b1.scale_log2_bits`,
        # `dense.softcap_mul_bits`); a layer whose selector scale differs (Gemma-2-9B/27B: `query_pre_attn_scalar != head_dim`) has no kind.
        obs_scale = obs_sel.get("scale")
        if obs_scale is not None and (not isinstance(obs_scale, (int, float)) or float(obs_scale) != float(D ** -0.5)):
            refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} has an OBSERVED attention scale {obs_scale!r} != D^-0.5 = {D ** -0.5!r} "
                   f"(D={D}); the registered attention kinds fold softmax_scale = D^-0.5 and no kind takes another scale")
        # target-keyed chain: on the accepted target (Ampere step, FA2) the role stays `attention` / `attention_softcap`
        # (digests unchanged); otherwise `attention_dot` = Attention_v2{BN, DOT, INV} with the block size of the launch the target's FA
        # version issues (FA3's tile rule reads the launch's query rows: this call's T on the prefill, 1 per request on decode).
        from verity_vllm.program.frontend import targets as TG
        tp = active_target_profile()
        accepted = TG.uses_accepted_kinds(tp)
        geom = None
        # the tile rule reads the LAUNCH's max_seqlen_q, not this request's rows: under varlen every sequence of
        # the launch runs the tile of the longest one.  The request's step index: 0 on the prefill (the call that opens the layer's
        # history, after any cached prefix; own rows = T), else P0 - LP + 1 (decode, own rows = 1; LP = the history's rows after the
        # prefill).  The declared launch context (`observe_launch_context`, build --launch-context, from the workload's co-scheduled
        # prompts) raises it; absent, own rows.
        lp_of = ctx.state.setdefault("launch_LP", {})
        if prefill:
            lp_of[layer_name] = P0 + T
        req_step = 0 if prefill else P0 - int(lp_of.get(layer_name, P0)) + 1
        own_rows = T if prefill else 1
        try:
            launch_sq, launch_src = TG.launch_seqlen_q(own_rows, req_step, active_launch_context())
        except TG.LaunchContextConflict as e:
            if tp.effective_flash_attn_version() == 3 and not accepted:
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} request step {req_step}: FA3 tile (kBlockN) is not one value -- {e}",
                       launch_context="conflict", request_step=req_step)
            launch_sq, launch_src = own_rows, "own rows (context conflict ignored: the FA2 tile does not read seqlen_q)"
        except TG.LaunchContextInvalid as e:
            # a declared context that omits this step or contradicts the request's own rows is a wrong
            # structural input: refused on EVERY target (FA2's tile does not read it, but the Program would still carry a false declaration).
            refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} request step {req_step}: invalid declared launch context -- {e}",
                   launch_context="invalid", request_step=req_step, own_rows=own_rows)
        if not accepted:
            if role != "attention" and not (role == "attention_softcap" and TG.fa2_check_inf(tp)):
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r}: kind {role!r} has no target-keyed variant on {tp.tensor_core_family} "
                       f"(FA{tp.effective_flash_attn_version()}); off the accepted target only the plain causal GQA chain is registered, "
                       f"and the softcapped one under FA2's per-iteration Check_inf")
            geom = TG.attention_geometry(tp, D, launch_sq, NH, KVH)
        # FA3's per-iteration Check_inf chain (`fa3_construction = check-inf-per-iteration`): each row's tile decides which of its key blocks
        # the kernel guards -- the row's index among this request's query rows at the step (i on the prefill, 0 on a decode step)
        check_inf = geom is not None and TG.fa3_check_inf(tp)
        # FA2's (`targets.fa2_check_inf`): the row's tile and the launch's causality decide it, from the same row index and own rows
        check_inf2 = geom is not None and TG.fa2_check_inf(tp)
        # under registered_values the chain's twin that reads the scale (a softcapped chain: its two words) as inputs, with no CAP
        scale = attention_scale(ctx, D, node, cap=extra.get("CAP"))
        dot_kind = ("attention_dot" if not scale else "attention_softcap_dot_value" if extra else "attention_dot_value" if check_inf2
                    else "attention_dot_fa3_value" if check_inf else "attention_dot_plain_value")
        statics = {} if scale else extra
        for i in range(T):
            P = P0 + i + 1
            if window is not None and P > window:
                refuse("numerics_unregistered", node, mp, f"layer {layer_name!r} has an OBSERVED sliding window of {window} keys and position {P - 1} "
                       f"has {P} keys in its history: the window is active and no windowed attention kind is registered "
                       f"(positions < {window} are exact under the unwindowed kind)")
            if geom is None:
                fn = ctx.kind(f"{role}_value" if scale else role, node, T=P, NH=NH, KVH=KVH, D=D, **statics)
            else:
                fn = ctx.kind(dot_kind, node, T=P, NH=NH, KVH=KVH, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"],
                              **({"MASKED_FROM": TG.fa3_masked_from(tp, P, NH, KVH, D, launch_sq, i if prefill else 0)} if check_inf else {}),
                              **({"FA2_MASKED_FROM": TG.fa2_masked_from(tp, P, D, launch_sq, i if prefill else 0, own_rows)} if check_inf2 else {}),
                              **statics)
            kcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), klog.cut(P))
            vcat = Coll(Array(P, Array(KVH, Array(D, Value(16)))), vlog.cut(P))
            outs.append(ctx.call(fn, qrows[i], kcat, vcat, *scale, node=node, rule=self, name=f"{ctx.hint_name(node)}[t={P - 1}]"))
        ctx.rebind(out, array_of(outs).reshape(_tensor_type(qs, 16)), self, what="output")
        obs = (_OBSERVED["attention_impls"] or {}).get(layer_name, {})
        ctx.record(self, node, {"layer_name": layer_name, "positions": [P0, P0 + T - 1], "NH": NH, "KVH": KVH, "D": D, "history_rows_after": P0 + T,
                                "kv_write_consumed": dep is not None, "assumed_causal": True,
                                "attention_impl_observed": attn_lookup_key(obs), "attention_pins": {**ATTN_PINS, "vllm_flash_attn_version": tp.effective_flash_attn_version()},
                                "attention_kind_selected": {"role": role if geom is None else "attention_dot", **extra, "inert_window": window,
                                                            **({} if geom is None else {"BN": geom["BN"], "DOT": geom["DOT"].name, "INV": geom["INV"].name,
                                                                                        "fa_version": geom["fa_version"], "tensor_core_family": tp.tensor_core_family,
                                                                                        # what the tile rule read and where it came from
                                                                                        "launch_seqlen_q": launch_sq, "launch_seqlen_q_source": launch_src,
                                                                                        "request_step": req_step, "own_rows": own_rows,
                                                                                        **({"fa3_construction": tp.effective_fa3_construction()} if check_inf else {})})}},
                   emitted=f"{fn.id} x{T} (T={P0 + 1}..{P0 + T})")
