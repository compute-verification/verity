"""[DSV4] DeepSeek-V4's manifold-constrained hyper-connections as the H200 runs them (vLLM d9105ea80 `kernels/mhc/tilelang.py`,
`registry.mhc`), over the four `vllm_iface` ops the model calls (`export_compat.registered_op_bodies_as_ops`,
`mhc_pre_broadcast_leaf`):

    mhc_pre_broadcast_tilelang(residual [T, H], fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                               sinkhorn_repeat, n_splits, norm_weight, norm_eps, fn_broadcast)
        -> (streams [T, HC, H], post [T, HC, 1], comb [T, HC, HC], layer input [T, H])
        layer 0's pre, per token `mhc_pre_broadcast{HC,H,NS,TN,REPEAT,EPS,HC_EPS}` over fn_broadcast and the embedding copied to
        the streams, `mhc_residual_broadcast{HC,H}`
    mhc_fused_post_pre_tilelang(x [T, H], residual [T, HC, H], post, comb, fn, ...) -> (streams', post', comb', layer input)
        per token `mhc_fused_post_pre` at most 16 tokens (`use_small_fma`), `mhc_post_pre_tf32` above
    mhc_post_tilelang(x, residual, post, comb) -> streams', per token `mhc_post{HC,N}`
    hc_head_fused_kernel_tilelang(streams [T, HC, H], fn, scale, base, rms_eps, hc_eps) -> [T, H], per token `hc_head{HC,H,TH,EPS,HC_EPS}`

The split counts are the dispatch's own, whatever `n_splits` / `tile_n` the call passes: the FMA path's 4 (8 below 8 tokens at
H <= 4096), DeepGEMM's `compute_num_split(64, K, cdiv(T, 64))` on the target's SMs (`kernels.mhc_twin.tf32_num_split`).  T, there
and in the FMA path's 16-token bound, is the launch's token count (`launch_tokens_of`), the request's own rows only for a batch of
one; the kernels' words are per token.  The
threads are the kernels' at the hidden size (`MHC_GEOMETRY`).  EPS is `rms_eps` = `norm_eps` and HC_EPS `hc_pre_eps` =
`hc_sinkhorn_eps`; the post multiplier is 2.0 (`s + s` in the Definitions); the pre without a fused norm is not modelled.
"""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import array_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import call_args, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile, launch_tokens_of
from verity_vllm.program.frontend.rules.vllm_dsa import _missing, _rows, _static_float, _value
from verity_vllm.program.frontend.rules.vllm_iface import iface, targets_for
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity.ml.definitions.numerics.mhc_twin import tf32_num_split

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

BF16, F32 = torch.bfloat16, torch.float32

#: the kernels' threads at DeepSeek-V4-Pro's hidden size (registry.mhc's statics; an H200's GO run):
#: `mhc_fused_tilelang`'s TH, the fused norm's TN, `hc_head_fuse_tilelang`'s TH
MHC_GEOMETRY = {7168: {"TH": 256, "TN": 64, "TH_HEAD": 128}}
#: the most tokens `mhc_fused_post_pre_tilelang` runs on its FMA path (`use_small_fma`, mhc/tilelang.py:518)
SMALL_FMA_TOKENS = 16

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"mhc_pre_broadcast": "layer 0's TF32 pre (MhcPreBroadcast_v1)",
                "mhc_residual_broadcast": "layer 0's embedding copied to the streams (MhcResidualBroadcast_v1)",
                "mhc_fused_post_pre": "the FMA post and pre at most 16 tokens (MhcFusedPostPre_v1)",
                "mhc_post_pre_tf32": "the post and TF32 pre above 16 tokens (MhcPostPreTf32_v1)",
                "mhc_post": "mhc_post_tilelang (MhcPost_v1)",
                "hc_head": "hc_head_fuse_tilelang (HcHead_v1)"}

def post_pre_path(M: int, HC: int, H: int, sms: int) -> tuple[str, int]:
    """The role `mhc_fused_post_pre_tilelang` runs at M launch tokens and its split count: the FMA path's 4 (8 below 8 tokens at
    H <= 4096) at most `SMALL_FMA_TOKENS`, above them DeepGEMM's TF32 split count over HC*H on `sms` SMs."""
    if M <= SMALL_FMA_TOKENS:
        return "mhc_fused_post_pre", 8 if M < 8 and H <= 4096 else 4
    return "mhc_post_pre_tf32", tf32_num_split(M, HC * H, sms)


def _geometry(H: int) -> dict | str:
    g = MHC_GEOMETRY.get(H)
    return g if g is not None else f"numerics_unregistered: no pinned mHC thread geometry at hidden size {H} (pinned: {sorted(MHC_GEOMETRY)})"


def _hopper() -> str | None:
    cc = tuple(active_target_profile().compute_capability)
    return None if cc == (9, 0) else f"numerics_unregistered: the TF32 pre is DeepGEMM's cc 9.0 hc-prenorm GEMM; the target is cc {cc}"


def _pre_statics(ctx, a, HC: int, H: int) -> dict | str:
    """The fused pre's statics, or why it is not modelled: scale [3] / base [OUT] f32, norm_weight [H] bf16, the eps and repeat
    statics as the Definitions take them, the thread geometry at H."""
    OUT = HC * (2 + HC)
    for r in (_value(ctx, a["hc_scale"], F32, (3,), "hc_scale"), _value(ctx, a["hc_base"], F32, (OUT,), "hc_base")):
        if r:
            return r
    if a.get("norm_weight") is None:
        return "numerics_unregistered: the mHC pre without its fused RMS norm (mhc_pre_big_fuse_tilelang) is not modelled"
    if (r := _value(ctx, a["norm_weight"], BF16, (H,), "norm_weight")):
        return r
    eps = [a.get(k) for k in ("rms_eps", "norm_eps", "hc_pre_eps", "hc_sinkhorn_eps", "hc_post_mult_value")]
    if not all(_static_float(e) for e in eps) or not isinstance(a.get("sinkhorn_repeat"), int):
        return "the eps, the post multiplier and the Sinkhorn repeat are not statics"
    if eps[0] != eps[1] or eps[2] != eps[3] or float(eps[4]) != 2.0:
        return (f"numerics_unregistered: rms_eps {eps[0]} / norm_eps {eps[1]}, hc_pre_eps {eps[2]} / hc_sinkhorn_eps {eps[3]} and the post "
                f"multiplier {eps[4]}: the Definitions take one EPS, one HC_EPS and the multiplier 2.0")
    g = _geometry(H)
    if isinstance(g, str):
        return g
    return {"HC": HC, "H": H, "TN": g["TN"], "REPEAT": int(a["sinkhorn_repeat"]), "EPS": float(eps[0]), "HC_EPS": float(eps[2])}


def _mixes(ctx, a, T: int, HC: int, H: int) -> str | None:
    for r in (_value(ctx, a["x"], BF16, (T, H), "x"), _value(ctx, a["comb_res_mix"], F32, (T, HC, HC), "comb_res_mix")):
        if r:
            return r
    post = a["post_layer_mix"]
    return _value(ctx, post, F32, (T, HC, 1), "post_layer_mix") and _value(ctx, post, F32, (T, HC), "post_layer_mix")


def _post_rows(ctx, a, T: int, HC: int):
    return ctx.coll(a["post_layer_mix"]).reshape(_tensor_type((T, HC), 32))


def _outputs(outs: list, T: int, HC: int, H: int, layer: int, mixes: int) -> list:
    """(streams [T, HC, H], post [T, HC, 1], comb [T, HC, HC], layer input [T, H]) from per-token results with the layer input at
    `layer` and (post, comb) from `mixes`."""
    post = array_of([o[mixes] for o in outs]).reshape(_tensor_type((T, HC, 1), 32))
    return [None, post, array_of([o[mixes + 1] for o in outs]), array_of([o[layer] for o in outs])]


class MhcPreBroadcastRule(OpRule):
    """`vllm_iface.mhc_pre_broadcast_tilelang` (layer 0's pre) per token: `MhcPreBroadcast_v1` of the embedding over
    fn_broadcast, NS DeepGEMM's split count at the launch's tokens over H, and `MhcResidualBroadcast_v1` for the streams."""
    modes = frozenset({"profile"})
    info = RuleInfo("MhcPreBroadcast[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.mhc_pre_broadcast_tilelang",
                    preconditions=("residual [T, H] bf16; fn [OUT, HC*H], fn_broadcast [OUT, H], hc_scale [3], hc_base [OUT] f32, "
                                   "OUT = HC (2 + HC); norm_weight [H] bf16", "rms_eps = norm_eps, hc_pre_eps = hc_sinkhorn_eps, the post "
                                   "multiplier 2.0, the Sinkhorn repeat static", "a pinned thread geometry at H; a cc 9.0 target"),
                    evidence_domain="structural; numerics: MhcPreBroadcast_v1 and MhcResidualBroadcast_v1 (registry.mhc)")

    def targets(self):
        return {iface.mhc_pre_broadcast_tilelang.default}

    def _statics(self, node, ctx):
        a = call_args(node)
        s = shape(ctx, a["residual"])
        if s is None or len(s) != 2:
            return "residual is not [T, H]"
        T, H = s
        fs = shape(ctx, a["fn"])
        if fs is None or len(fs) != 2 or fs[1] % H:
            return "fn is not [OUT, HC*H]"
        HC = fs[1] // H
        for r in (_value(ctx, a["residual"], BF16, (T, H), "residual"), _value(ctx, a["fn"], F32, (HC * (2 + HC), HC * H), "fn"),
                  a.get("fn_broadcast") is None and "fn_broadcast is None",
                  _value(ctx, a["fn_broadcast"], F32, (HC * (2 + HC), H), "fn_broadcast"), _hopper()):
            if r:
                return r
        S = _pre_statics(ctx, a, HC, H)
        if isinstance(S, str):
            return S
        return a, T, {**S, "NS": tf32_num_split(launch_tokens_of(node, T), H, active_target_profile().num_sms)}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["mhc_pre_broadcast", "mhc_residual_broadcast"], ROLE_KERNELS)):
            return r
        S = self._statics(node, ctx)
        return S if isinstance(S, str) else True

    def emit(self, node, ctx):
        a, T, S = self._statics(node, ctx)
        HC, H, h = S["HC"], S["H"], ctx.hint_name(node)
        pre = ctx.kind("mhc_pre_broadcast", node, **S)
        copy = ctx.kind("mhc_residual_broadcast", node, HC=HC, H=H)
        x, fn, scale, base, w = (_rows(ctx, a["residual"], 16), ctx.coll(a["fn_broadcast"]), ctx.coll(a["hc_scale"]), ctx.coll(a["hc_base"]),
                                 ctx.coll(a["norm_weight"]))
        outs = [ctx.call(pre, x[i], fn, scale, base, w, node=node, rule=self, name=f"{h}/pre[{i}]") for i in range(T)]
        res = [ctx.call(copy, x[i], node=node, rule=self, name=f"{h}/streams[{i}]") for i in range(T)]
        out = _outputs(outs, T, HC, H, 0, 1)
        out[0] = array_of(res)
        ctx.define(node, out)
        for k, role in (("fn_broadcast", "mhc.fn_broadcast"), ("hc_scale", "mhc.scale"), ("hc_base", "mhc.base"),
                        ("norm_weight", "mhc.norm.weight")):
            ctx.bind_weight_role(a[k], self, role, HC=HC, H=H)
        ctx.record(self, node, {"T": T, **S}, emitted=f"{pre.id} x{T} + {copy.id} x{T}")


class MhcFusedPostPreRule(OpRule):
    """`vllm_iface.mhc_fused_post_pre_tilelang` per token: `MhcFusedPostPre_v1{NS=4 (8 below 8 tokens at H <= 4096),TH}` at most
    16 tokens in the launch, `MhcPostPreTf32_v1{NS=DeepGEMM's split count at its tokens over HC*H}` above."""
    modes = frozenset({"profile"})
    info = RuleInfo("MhcFusedPostPre[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.mhc_fused_post_pre_tilelang",
                    preconditions=("x [T, H], residual [T, HC, H] bf16; post_layer_mix [T, HC, 1] or [T, HC], comb_res_mix [T, HC, HC] f32",
                                   "fn [OUT, HC*H], hc_scale [3], hc_base [OUT] f32; norm_weight [H] bf16",
                                   "rms_eps = norm_eps, hc_pre_eps = hc_sinkhorn_eps, the post multiplier 2.0, the Sinkhorn repeat static",
                                   "a pinned thread geometry at H; above 16 tokens a cc 9.0 target"),
                    evidence_domain="structural; numerics: MhcFusedPostPre_v1 / MhcPostPreTf32_v1 (registry.mhc)")

    def targets(self):
        return targets_for("mhc_fused_post_pre_tilelang")

    def _statics(self, node, ctx):
        a = call_args(node)
        s = shape(ctx, a["residual"])
        if s is None or len(s) != 3:
            return "residual is not [T, HC, H]"
        T, HC, H = s
        M = launch_tokens_of(node, T)
        for r in (_value(ctx, a["residual"], BF16, (T, HC, H), "residual"), _mixes(ctx, a, T, HC, H),
                  _value(ctx, a["fn"], F32, (HC * (2 + HC), HC * H), "fn"), M > SMALL_FMA_TOKENS and _hopper()):
            if r:
                return r
        S = _pre_statics(ctx, a, HC, H)
        if isinstance(S, str):
            return S
        role, NS = post_pre_path(M, HC, H, active_target_profile().num_sms)
        return a, T, role, {**S, "NS": NS, **({"TH": MHC_GEOMETRY[H]["TH"]} if role == "mhc_fused_post_pre" else {})}

    def applies(self, node, ctx):
        S = self._statics(node, ctx)
        if isinstance(S, str):
            return S
        return _missing(ctx, [S[2]], ROLE_KERNELS) or True

    def emit(self, node, ctx):
        a, T, role, S = self._statics(node, ctx)
        HC, H, h = S["HC"], S["H"], ctx.hint_name(node)
        fn = ctx.kind(role, node, **S)
        x, res, post, comb = _rows(ctx, a["x"], 16), _rows(ctx, a["residual"], 16), _post_rows(ctx, a, T, HC), _rows(ctx, a["comb_res_mix"], 32)
        w, scale, base, wn = ctx.coll(a["fn"]), ctx.coll(a["hc_scale"]), ctx.coll(a["hc_base"]), ctx.coll(a["norm_weight"])
        outs = [ctx.call(fn, x[i], res[i], post[i], comb[i], w, scale, base, wn, node=node, rule=self, name=f"{h}/post_pre[{i}]")
                for i in range(T)]
        out = _outputs(outs, T, HC, H, 1, 2)
        out[0] = array_of([o[0] for o in outs])
        ctx.define(node, out)
        for k, role in (("fn", "mhc.fn"), ("hc_scale", "mhc.scale"), ("hc_base", "mhc.base"), ("norm_weight", "mhc.norm.weight")):
            ctx.bind_weight_role(a[k], self, role, HC=HC, H=H)
        ctx.record(self, node, {"T": T, **S}, emitted=f"{fn.id} x{T}")


class MhcPostRule(OpRule):
    """`vllm_iface.mhc_post_tilelang` (after the last layer) per token: `MhcPost_v1{HC,N=H}`."""
    modes = frozenset({"profile"})
    info = RuleInfo("MhcPost[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.mhc_post_tilelang",
                    preconditions=("x [T, H], residual [T, HC, H] bf16; post_layer_mix [T, HC, 1] or [T, HC], comb_res_mix [T, HC, HC] f32",),
                    evidence_domain="structural; numerics: MhcPost_v1 (registry.mhc)")

    def targets(self):
        return targets_for("mhc_post_tilelang")

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["mhc_post"], ROLE_KERNELS)):
            return r
        a = call_args(node)
        s = shape(ctx, a["residual"])
        if s is None or len(s) != 3:
            return "residual is not [T, HC, H]"
        return _value(ctx, a["residual"], BF16, s, "residual") or _mixes(ctx, a, *s) or True

    def emit(self, node, ctx):
        a = call_args(node)
        T, HC, H = shape(ctx, a["residual"])
        h = ctx.hint_name(node)
        fn = ctx.kind("mhc_post", node, HC=HC, N=H)
        x, res, post, comb = _rows(ctx, a["x"], 16), _rows(ctx, a["residual"], 16), _post_rows(ctx, a, T, HC), _rows(ctx, a["comb_res_mix"], 32)
        ctx.define(node, array_of([ctx.call(fn, x[i], res[i], post[i], comb[i], node=node, rule=self, name=f"{h}/post[{i}]")
                                   for i in range(T)]))
        ctx.record(self, node, {"T": T, "HC": HC, "H": H}, emitted=f"{fn.id} x{T}")


class HcHeadRule(OpRule):
    """`vllm_iface.hc_head_fused_kernel_tilelang` per token: `HcHead_v1{HC,H,TH,EPS=rms_eps,HC_EPS=hc_eps}`."""
    modes = frozenset({"profile"})
    info = RuleInfo("HcHead[vllm-dsv4]", 1, "op-binding", applies_to="vllm_iface.hc_head_fused_kernel_tilelang",
                    preconditions=("hs_flat [T, HC, H] bf16; fn [HC, HC*H], hc_scale [1], hc_base [HC] f32; the eps static floats",
                                   "a pinned thread geometry at H"),
                    evidence_domain="structural; numerics: HcHead_v1 (registry.mhc)")

    def targets(self):
        return targets_for("hc_head_fused_kernel_tilelang")

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["hc_head"], ROLE_KERNELS)):
            return r
        a = call_args(node)
        s = shape(ctx, a["hs_flat"])
        if s is None or len(s) != 3:
            return "hs_flat is not [T, HC, H]"
        T, HC, H = s
        for r in (_value(ctx, a["hs_flat"], BF16, s, "hs_flat"), _value(ctx, a["fn"], F32, (HC, HC * H), "fn"),
                  _value(ctx, a["hc_scale"], F32, (1,), "hc_scale"), _value(ctx, a["hc_base"], F32, (HC,), "hc_base")):
            if r:
                return r
        if not (_static_float(a["rms_eps"]) and _static_float(a["hc_eps"])):
            return "rms_eps / hc_eps are not static floats"
        g = _geometry(H)
        return g if isinstance(g, str) else True

    def emit(self, node, ctx):
        a = call_args(node)
        T, HC, H = shape(ctx, a["hs_flat"])
        h = ctx.hint_name(node)
        S = {"HC": HC, "H": H, "TH": MHC_GEOMETRY[H]["TH_HEAD"], "EPS": float(a["rms_eps"]), "HC_EPS": float(a["hc_eps"])}
        fn = ctx.kind("hc_head", node, **S)
        x, w, scale, base = _rows(ctx, a["hs_flat"], 16), ctx.coll(a["fn"]), ctx.coll(a["hc_scale"]), ctx.coll(a["hc_base"])
        ctx.define(node, array_of([ctx.call(fn, x[i], w, scale, base, node=node, rule=self, name=f"{h}/head[{i}]") for i in range(T)]))
        ctx.bind_weight_role(a["fn"], self, "hc_head.fn", HC=HC, H=H)
        ctx.bind_weight_role(a["hc_scale"], self, "hc_head.scale", HC=HC)
        ctx.bind_weight_role(a["hc_base"], self, "hc_head.base", HC=HC)
        ctx.record(self, node, {"T": T, **S}, emitted=f"{fn.id} x{T}")


VLLM_MHC_RULES = [MhcPreBroadcastRule(), MhcFusedPostPreRule(), MhcPostRule(), HcHeadRule()]

__all__ = ["HcHeadRule", "MHC_GEOMETRY", "MhcFusedPostPreRule", "MhcPostRule", "MhcPreBroadcastRule", "SMALL_FMA_TOKENS", "VLLM_MHC_RULES",
           "post_pre_path"]
