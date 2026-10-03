"""The block-scaled FP8 linear and the same-dtype cast it ends with."""

from __future__ import annotations

import torch

from verity.ir.refs import Coll
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.elementwise import _C_op
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16, _row_value, _untransposed, aten


# -- [R15 fp8] the block-scaled FP8 linear: dynamic per-token-group activation quant + the blockwise CUTLASS scaled GEMM -----------------

FP8_E4M3 = torch.float8_e4m3fn
#: the quant op's pinned scalars (`per_token_group_quant_fp8(..., group_size=128, eps=1e-10, dtype=e4m3, use_ue8m0=False)` as vLLM's
#: `QuantFP8(group_shape=(1, 128), use_ue8m0=False)` calls it for `CutlassFp8BlockScaledMMKernel`); the Definition `Fp8GroupQuant_v1{K,G}`
#: hard-codes eps and the e4m3 range, so a launch with other values is refused (another kernel configuration, not this Definition).
FP8_QUANT_PINS = {"group_size": 128, "eps": 1e-10, "fp8_min": -448.0, "fp8_max": 448.0, "use_ue8m0": False}


def _f32_equal(a, b: float) -> bool:
    import numpy as _np
    return isinstance(a, (int, float)) and not isinstance(a, bool) and _np.float32(a) == _np.float32(b)


class PerTokenGroupFp8QuantRule(InPlaceRule):
    """`_C.per_token_group_fp8_quant(x bf16 [.., K], x_q! e4m3 [.., K], x_s! f32 [.., K/G], G, eps, fp8_min, fp8_max, use_ue8m0,
    column_major_scales, tma_aligned_scales)` -> `fp8_group_quant{K,G}` (Fp8GroupQuant_v1) per token row; BOTH allocations are rebound
    (`x_q` to port 0, `x_s` to port 1).  `x_s` is usually a permuted view of its allocation (column-major scales): the view node is what the
    GEMM reads, so the view node is rebound (AllocView v2).  The scale layout / TMA alignment flags are storage facts, not values."""
    modes = frozenset({"profile"})
    info = RuleInfo("PerTokenGroupFp8Quant[vllm]", 1, "in-place-binding",
                    applies_to="_C.per_token_group_fp8_quant(x, Tensor! x_q, Tensor! x_s, group_size, eps, fp8_min, fp8_max, use_ue8m0, column_major_scales, tma_aligned_scales)",
                    preconditions=("x is bf16 [.., K] with K % group_size == 0", "x_q is an unwritten allocation float8_e4m3fn [.., K]",
                                   "x_s is an unwritten allocation (or a view of one) f32 [.., K/group_size]",
                                   "group_size == 128, eps == 1e-10f, fp8_min/max == -448/448, use_ue8m0 == False (the pinned Definition)"),
                    evidence_domain="structural; numerics: Fp8GroupQuant_v1 (exact-model-tested on the H100 against _C.per_token_group_fp8_quant)")

    def targets(self):
        return _C_op("per_token_group_fp8_quant")

    @staticmethod
    def _scalars(node) -> dict:
        names = ("x", "x_q", "x_s", "group_size", "eps", "fp8_min", "fp8_max", "use_ue8m0", "column_major_scales", "tma_aligned_scales")
        d = dict(zip(names, node.args))
        d.update(node.kwargs)
        return d

    def applies(self, node, ctx):
        a = self._scalars(node)
        if not all(k in a for k in ("x", "x_q", "x_s", "group_size", "eps", "fp8_min", "fp8_max")):
            return "unexpected operator arity"
        r = _row_value(ctx, a["x"], "x")
        if r:
            return r
        xs_shape = shape(ctx, a["x"])
        G = a["group_size"]
        if not isinstance(G, int) or G != FP8_QUANT_PINS["group_size"]:
            return f"numerics_unregistered: group_size={G!r}; Fp8GroupQuant_v1 is pinned for group 128"
        if xs_shape[-1] % G:
            return f"K={xs_shape[-1]} is not a multiple of the group size {G}"
        for key in ("eps", "fp8_min", "fp8_max"):
            if not _f32_equal(a[key], FP8_QUANT_PINS[key]):
                return f"numerics_unregistered: {key}={a[key]!r} differs from the pinned {FP8_QUANT_PINS[key]!r} (another kernel configuration)"
        if bool(a.get("use_ue8m0", False)):
            return "numerics_unregistered: use_ue8m0=True (power-of-two scales, the DeepGEMM path) has no registered Definition"
        for name in ("x_q", "x_s"):
            if not isinstance(ctx.val(a[name]), Alloc):
                return f"{name} is not an unwritten allocation ({type(ctx.val(a[name])).__name__})"
        if not dtype_is(ctx, a["x_q"], FP8_E4M3):
            return f"numerics_unregistered: x_q has dtype {ctx.dtype_of(a['x_q'])}; the Definition quantises to float8_e4m3fn"
        if not dtype_is(ctx, a["x_s"], torch.float32):
            return f"x_s has dtype {ctx.dtype_of(a['x_s'])}, not float32"
        if shape(ctx, a["x_q"]) != xs_shape:
            return f"x_q {shape(ctx, a['x_q'])} vs x {xs_shape}"
        if shape(ctx, a["x_s"]) != xs_shape[:-1] + (xs_shape[-1] // G,):
            return f"x_s {shape(ctx, a['x_s'])} vs x {xs_shape} / {G}"
        return True

    def emit(self, node, ctx):
        a = self._scalars(node)
        rows, lead = rows_of(ctx.coll(a["x"]))
        K, G = rows[0].type.n, int(a["group_size"])
        fn = ctx.kind("fp8_group_quant", node, K=K, G=G)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(a["x_q"], from_rows([o[0] for o in outs], lead), self, what="x_q")
        ctx.rebind(a["x_s"], from_rows([o[1] for o in outs], lead), self, what="x_s")
        ctx.record(self, node, {"K": K, "G": G, "rows": len(rows), "eps": a["eps"], "fp8_min": a["fp8_min"], "fp8_max": a["fp8_max"],
                                "use_ue8m0": bool(a.get("use_ue8m0", False)), "column_major_scales": bool(a.get("column_major_scales", False)),
                                "tma_aligned_scales": bool(a.get("tma_aligned_scales", False))}, emitted=f"{fn.id} x{len(rows)}")


class CutlassScaledMmFp8BlockRule(InPlaceRule):
    """`_C.cutlass_scaled_mm(out! bf16 [.., N], a e4m3 [.., K], b = W.T ([K, N] view of e4m3 W [N, K]), a_scales f32 [.., K/G],
    b_scales = Ws.T ([K/G, N/G] view of f32 Ws [N/G, K/G]), bias None)` -> `scaled_mm_fp8_block{K,N,G}` per token row (the active target's
    `fp8_scale_construction`: ScaledMmFp8BlockSharedScale_v1 by default, ScaledMmFp8Block_v1 per-coordinate); `out` is rebound.  W and Ws are prescribed weight Inputs (the checkpoint's e4m3 words and f32 block scales); `a` / `a_scales` are the
    values `PerTokenGroupFp8Quant[vllm]` computed.  Binds the BLOCK-scaled shapes only: a per-tensor / per-token scaled_mm (scales [M,1] /
    [1,N] / 0-d) is another kernel (`ScaledMmFp8_v1` of registry.fp8, R14 QZ) and is refused here, named."""
    modes = frozenset({"profile"})
    info = RuleInfo("CutlassScaledMmFp8Block[vllm]", 1, "in-place-binding",
                    applies_to="_C.cutlass_scaled_mm(Tensor! out, a e4m3, b e4m3 [K, N], a_scales f32 [.., K/G], b_scales f32 [K/G, N/G], bias None)",
                    preconditions=("the active target's tensor-core family is hopper (cc 9.x): the Definition is that family's blockwise CUTLASS "
                                   "kernel, and `_C.cutlass_scaled_mm` on another target runs another kernel under the same op and kernel class",
                                   "a is a float8_e4m3fn value [.., K]; out an unwritten bf16 allocation [.., N]",
                                   "b is a transposed view of an e4m3 [N, K] matrix; b_scales a transposed view of an f32 [N/G, K/G] matrix, G = 128",
                                   "a_scales is an f32 value [.., K/G] (the group scales the quant rule computed)", "bias is None",
                                   "K % G == 0 and N % G == 0"),
                    evidence_domain="structural; numerics: ScaledMmFp8Block_v1 (exact-model-tested on the H100 against the blockwise _C.cutlass_scaled_mm, both dispatch "
                                    "variants) and ScaledMmFp8BlockSharedScale_v1, its function restated")

    def targets(self):
        return _C_op("cutlass_scaled_mm")

    def applies(self, node, ctx):
        tp = active_target_profile()
        if tp.tensor_core_family != "hopper":
            return (f"numerics_unregistered: target {tp.tensor_core_family} (cc {tp.capability_str}); ScaledMmFp8Block_v1 is the hopper "
                    f"family's blockwise CUTLASS kernel, and this target's `_C.cutlass_scaled_mm` blockwise kernel has no registered Definition")
        if len(node.args) < 5:
            return "unexpected operator arity"
        out, a, b, a_s, b_s = node.args[:5]
        bias = node.args[5] if len(node.args) > 5 else node.kwargs.get("bias")
        if bias is not None:
            return "numerics_unregistered: a fused bias in the blockwise FP8 epilogue has no registered Definition (vLLM adds the bias outside the kernel)"
        r = _row_value(ctx, a, "a", dtype=FP8_E4M3)
        if r:
            return r
        if not isinstance(ctx.val(out), Alloc):
            return f"out is not an unwritten allocation ({type(ctx.val(out)).__name__})"
        if not dtype_is(ctx, out, BF16):
            return f"numerics_unregistered: out has dtype {ctx.dtype_of(out)}; the Definition's epilogue is cvt.rn.bf16.f32"
        ash, bsh, ash_s, bsh_s, osh = shape(ctx, a), shape(ctx, b), shape(ctx, a_s), shape(ctx, b_s), shape(ctx, out)
        if bsh is None or len(bsh) != 2 or ash[-1] != bsh[0]:
            return f"b {bsh} is not [K, N] for a [.., K={ash[-1]}]"
        K, N = bsh
        G = FP8_QUANT_PINS["group_size"]
        if not dtype_is(ctx, b, FP8_E4M3):
            return f"numerics_unregistered: b has dtype {ctx.dtype_of(b)}; the Definition multiplies e4m3 by e4m3"
        if not (dtype_is(ctx, a_s, torch.float32) and dtype_is(ctx, b_s, torch.float32)):
            return f"scales are {ctx.dtype_of(a_s)} / {ctx.dtype_of(b_s)}, not float32"
        if ash_s is None or bsh_s is None or ash_s == ash[:-1] + (1,) or bsh_s in ((1, N), (N,), (), (1,)):
            return (f"numerics_unregistered: per-tensor / per-token scales a_scales {ash_s}, b_scales {bsh_s} -- the FastAccum per-tensor kernel "
                    f"(ScaledMmFp8_v1, registry.fp8) is not bound by this rule; this rule binds the blockwise kernel (a_scales [.., K/128], b_scales [K/128, N/128])")
        if K % G or N % G:
            return f"numerics_unregistered: K={K}, N={N} not multiples of the block {G} (vLLM pads such layers; not modelled)"
        if ash_s != ash[:-1] + (K // G,) or bsh_s != (K // G, N // G):
            return f"scale shapes a_scales {ash_s} (want [.., {K // G}]), b_scales {bsh_s} (want [{K // G}, {N // G}]) are not the 128x128 block layout"
        if osh != ash[:-1] + (N,):
            return f"out {osh} vs a {ash} x b {bsh}"
        if ctx.weight_of(b) is None or ctx.weight_of(b_s) is None:
            return "b / b_scales are not (views of) parameters; the Definition takes the checkpoint's e4m3 words and block scales as weight Inputs"
        return True

    def emit(self, node, ctx):
        out, a, b, a_s, b_s = node.args[:5]
        K, N = shape(ctx, b)
        G = FP8_QUANT_PINS["group_size"]
        w = _untransposed(ctx, b, K, N, ctx.hint_name(node) + "/w")
        ws = _untransposed(ctx, b_s, K // G, N // G, ctx.hint_name(node) + "/ws")
        ctx.bind_weight_role(b, self, "fp8_w", K=K, N=N)
        ctx.bind_weight_role(b_s, self, "fp8_ws", K=K, N=N, G=G)
        rows, lead = rows_of(ctx.coll(a))
        srows, _ = rows_of(ctx.coll(a_s))
        if len(rows) != len(srows):
            refuse("shape", node, ctx.module_path(node), f"a has {len(rows)} rows but a_scales {len(srows)}")
        scale = active_target_profile().effective_fp8_scale_construction()
        fn = ctx.kind("scaled_mm_fp8_block", node, K=K, N=N, G=G, SCALE=scale)
        outs = [ctx.call(fn, r, s, w, ws, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (r, s) in enumerate(zip(rows, srows))]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"K": K, "N": N, "G": G, "rows": len(rows), "scale": scale, "b_view": str(getattr(b, "target", None)),
                                "b_scales_view": str(getattr(b_s, "target", None))}, emitted=f"{fn.id} x{len(rows)}")


class SameDtypeCastRule(ViewRule):
    """`aten.to.dtype(x, dt)` / `_to_copy(x, dtype=dt)` with `dt == x.dtype`: the identity (ATen returns `self` for `to`; `_to_copy` is a
    copy of the same words).  vLLM's `BlockScaledMMLinearKernel.apply_weights` ends with `output.to(dtype=out_dtype)` on an already-bf16
    output.  Rank/shape unchanged; no gates."""
    info = RuleInfo("SameDtypeCast", 1, "view", applies_to="aten.to.dtype / _to_copy where the target dtype equals the operand's",
                    preconditions=("operand is a value collection", "target dtype == operand dtype"), evidence_domain="structural (no gates)")

    def targets(self):
        return {aten.to.dtype, aten._to_copy.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        src, dst = ctx.dtype_of(node.args[0]), ctx.dtype_of(node)
        if src is None or dst is None or src != dst:
            return f"{src} -> {dst} is not a same-dtype cast"
        if shape(ctx, node) != shape(ctx, node.args[0]):
            return "shape changes"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=None if node.target is aten._to_copy.default else node.args[0])
        ctx.record(self, node, {"dtype": str(ctx.dtype_of(node))}, emitted="(no gates)")
