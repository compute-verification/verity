"""The block-scaled FP8 linear and the same-dtype cast it ends with."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc, refuse
from verity_vllm.program.frontend.rules.base import InPlaceRule, OpRule, RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.elementwise import _C_op
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16, _row_value, _untransposed, aten
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.registry.fp8 import QUANT_GROUPS, weight_blocks


# -- [R15 fp8] block-scaled FP8 linears: vLLM's sm_90 default dynamic route, and the explicit CUTLASS route ------------------------------

FP8_E4M3 = torch.float8_e4m3fn
#: the quant op's pinned scalars (`per_token_group_quant_fp8(..., group_size=128, eps=1e-10, dtype=e4m3, use_ue8m0=False)` as vLLM's
#: `QuantFP8(group_shape=(1, 128), use_ue8m0=False)` calls it for `CutlassFp8BlockScaledMMKernel`); the Definition `Fp8GroupQuant_v1{K,G}`
#: hard-codes eps and the e4m3 range, so a launch with other values is refused (another kernel configuration, not this Definition).
#: `group_size` is the blockwise GEMM's block; the quant takes any of `fp8.QUANT_GROUPS`.
FP8_QUANT_PINS = {"group_size": 128, "eps": 1e-10, "fp8_min": -448.0, "fp8_max": 448.0, "use_ue8m0": False}


def _f32_equal(a, b: float) -> bool:
    import numpy as _np
    return isinstance(a, (int, float)) and not isinstance(a, bool) and _np.float32(a) == _np.float32(b)


def _vllm_op(name: str) -> set:
    try:
        return {getattr(torch.ops.vllm, name).default}
    except (AttributeError, RuntimeError):
        return set()


class DynamicDeepGemmFp8LinearRule(OpRule):
    """vLLM's default Hopper block-FP8 linear, one registered row call per token.

    The custom op owns both branches: FlashInfer for M < 32 and DeepGEMM for
    M >= 32.  GLM-5.3's load-time ``requant_weight_ue8m0_inplace`` output is
    the op's ``weight`` / ``weight_scale`` parameter pair, so those served
    tensors are the declared weight operands.
    """

    modes = frozenset({"profile"})
    info = RuleInfo(
        "DynamicDeepGemmFp8Linear[vllm-default]",
        1,
        "op-binding",
        applies_to="vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(input, weight, weight_scale, 128, True)",
        preconditions=(
            "Hopper without batch invariance and vLLM's default dynamic FlashInfer/DeepGEMM custom op",
            "bf16 input [M,K], e4m3 weight [N,K], f32 weight scales [ceil(N/128),K/128]",
            "group_size=128 and use_deep_gemm_e8m0=True",
        ),
        evidence_domain="deepgemm_sm90_capture/v1; GLM-5.3 f32 checkpoint scales are load-time requantised to ue8m0",
    )

    def targets(self):
        return _vllm_op("dynamic_flashinfer_deepgemm_blockscale_gemm") | {
            iface.dynamic_flashinfer_deepgemm_blockscale_gemm.default
        }

    def applies(self, node, ctx):
        target = active_target_profile()
        if target.compute_capability != (9, 0):
            return f"compute capability {target.compute_capability}, not cc 9.0"
        if target.batch_invariant:
            return "batch invariance selects DeepGEMM for every M"
        if len(node.args) != 5 or node.kwargs:
            return f"unexpected args={node.args[3:]} kwargs={node.kwargs}"
        x, w, ws, group, ue8m0 = node.args
        if group != 128 or ue8m0 is not True:
            return f"expected group_size=128 and use_deep_gemm_e8m0=True, got {group!r}, {ue8m0!r}"
        if not all(isinstance(ctx.val(v), Coll) for v in (x, w, ws)):
            return "input, weight or weight_scale is not a value"
        if ctx.dtype_of(x) != BF16 or ctx.dtype_of(w) != FP8_E4M3 or ctx.dtype_of(ws) != torch.float32:
            return f"expected bf16/e4m3/f32 operands, got {ctx.dtype_of(x)}, {ctx.dtype_of(w)}, {ctx.dtype_of(ws)}"
        xs, wsh, ssh, osh = shape(ctx, x), shape(ctx, w), shape(ctx, ws), shape(ctx, node)
        if xs is None or len(xs) != 2 or wsh is None or len(wsh) != 2 or xs[1] != wsh[1]:
            return f"input {xs} vs weight {wsh}"
        N, K = wsh
        if K % 128:
            return f"K={K} is not a multiple of 128"
        expected_scales = (weight_blocks(N, 128), K // 128)
        if ssh != expected_scales:
            return f"weight_scale {ssh}, expected {expected_scales}"
        if osh != xs[:-1] + (N,) or ctx.dtype_of(node) != BF16:
            return f"output {osh} {ctx.dtype_of(node)}, expected {xs[:-1] + (N,)} bf16"
        if ctx.weight_of(w) is None or ctx.weight_of(ws) is None:
            return "weight / weight_scale are not declared weight operands"
        role = "flashinfer_fp8_linear" if xs[0] < 32 else "deepgemm_fp8_linear"
        if not ctx.vocab.has(role):
            return f"the vocabulary has no {role}"
        return True

    def emit(self, node, ctx):
        x, w, ws, _group, _ue8m0 = node.args
        N, K = shape(ctx, w)
        rows, lead = rows_of(ctx.coll(x))
        branch = "flashinfer" if len(rows) < 32 else "deepgemm"
        fn = ctx.kind(f"{branch}_fp8_linear", node, K=K, N=N)
        calls = [
            ctx.call(fn, row, ctx.coll(w), ctx.coll(ws), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            for i, row in enumerate(rows)
        ]
        outs = calls if branch == "flashinfer" else [call[2] for call in calls]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(w, self, "fp8_w_requant_ue8m0", K=K, N=N)
        ctx.bind_weight_role(ws, self, "fp8_ws_requant_ue8m0", K=K, N=N, G=128)
        ctx.record(
            self,
            node,
            {"M": len(rows), "K": K, "N": N, "G": 128, "branch": branch, "use_deep_gemm_e8m0": True},
            emitted=f"{fn.id} x{len(rows)}",
        )


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
                                   "group_size in fp8.QUANT_GROUPS (64, 128), eps == 1e-10f, fp8_min/max == -448/448, use_ue8m0 == False "
                                   "(the pinned Definition)"),
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
        if not isinstance(G, int) or isinstance(G, bool) or G not in QUANT_GROUPS:
            return f"numerics_unregistered: group_size={G!r}; Fp8GroupQuant_v1 is registered for the groups {list(QUANT_GROUPS)}"
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
    b_scales = Ws.T ([K/G, cdiv(N, G)] view of f32 Ws [cdiv(N, G), K/G]: N may end mid-block, `fp8.weight_blocks`), bias None)` ->
    `scaled_mm_fp8_block{K,N,G}` per token row (the active target's `fp8_scale_construction`: ScaledMmFp8BlockSharedScale_v1 by
    default, ScaledMmFp8Block_v1 per-coordinate; the version 3 forms over the target's step where it is not Hopper's,
    `frontend.targets.fp8_block_dot`); `out` is rebound.  W and Ws are prescribed weight Inputs (the checkpoint's e4m3 words and f32 block scales); `a` / `a_scales` are the
    values `PerTokenGroupFp8Quant[vllm]` computed.  Binds the BLOCK-scaled shapes only: a per-tensor / per-token scaled_mm (scales [M,1] /
    [1,N] / 0-d) is another kernel (`ScaledMmFp8_v1` of registry.fp8, R14 QZ) and is refused here, named.  A target whose GEMM record
    has no blockwise step is refused: `_C.cutlass_scaled_mm` runs that target's own blockwise kernel under the same op and kernel class."""
    modes = frozenset({"profile"})
    info = RuleInfo("CutlassScaledMmFp8Block[vllm]", 1, "in-place-binding",
                    applies_to="_C.cutlass_scaled_mm(Tensor! out, a e4m3, b e4m3 [K, N], a_scales f32 [.., K/G], b_scales f32 [K/G, cdiv(N, G)], bias None)",
                    preconditions=("the active target's GEMM record has a blockwise FP8 step (`GemmTarget.fp8_block_dot`, registry.targets.fp8_block_dot)",
                                   "a is a float8_e4m3fn value [.., K]; out an unwritten bf16 allocation [.., N]",
                                   "b is a transposed view of an e4m3 [N, K] matrix; b_scales a transposed view of an f32 [cdiv(N, G), K/G] matrix, G = 128",
                                   "a_scales is an f32 value [.., K/G] (the group scales the quant rule computed)", "bias is None",
                                   "K % G == 0 (N need not be: the last weight block may be partial)"),
                    evidence_domain="structural; numerics: ScaledMmFp8Block_v1 (exact-model-tested on the H100 against the blockwise _C.cutlass_scaled_mm, both dispatch "
                                    "variants) and ScaledMmFp8BlockSharedScale_v1, its function restated; off hopper their version 3 forms over "
                                    "the target's step (model-untested until that target's capture)")

    def targets(self):
        return _C_op("cutlass_scaled_mm")

    def applies(self, node, ctx):
        from verity_vllm.program.frontend import targets as T
        try:
            T.fp8_block_dot(active_target_profile())
        except T.UnregisteredTarget as e:
            return f"numerics_unregistered: {e}"
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
                    f"(ScaledMmFp8_v1, registry.fp8) is not bound by this rule; this rule binds the blockwise kernel (a_scales [.., K/128], b_scales [K/128, cdiv(N, 128)])")
        if K % G:
            return f"numerics_unregistered: K={K} is not a multiple of the block {G} (the activation groups would end mid-group)"
        NB = weight_blocks(N, G)
        if ash_s != ash[:-1] + (K // G,) or bsh_s != (K // G, NB):
            return f"scale shapes a_scales {ash_s} (want [.., {K // G}]), b_scales {bsh_s} (want [{K // G}, {NB}]) are not the 128x128 block layout"
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
        ws = _untransposed(ctx, b_s, K // G, weight_blocks(N, G), ctx.hint_name(node) + "/ws")
        ctx.bind_weight_role(b, self, "fp8_w", K=K, N=N)
        ctx.bind_weight_role(b_s, self, "fp8_ws", K=K, N=N, G=G)
        rows, lead = rows_of(ctx.coll(a))
        srows, _ = rows_of(ctx.coll(a_s))
        if len(rows) != len(srows):
            refuse("shape", node, ctx.module_path(node), f"a has {len(rows)} rows but a_scales {len(srows)}")
        from verity_vllm.program.frontend import targets as T
        tp = active_target_profile()
        scale = tp.effective_fp8_scale_construction()
        dot = T.fp8_block_dot(tp)
        fn = ctx.kind("scaled_mm_fp8_block", node, K=K, N=N, G=G, SCALE=scale, **({} if dot is None else {"FP8_DOT": dot}))
        outs = [ctx.call(fn, r, s, w, ws, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (r, s) in enumerate(zip(rows, srows))]
        ctx.rebind(out, from_rows(outs, lead), self, what="out")
        ctx.record(self, node, {"K": K, "N": N, "G": G, "rows": len(rows), "scale": scale, "b_view": str(getattr(b, "target", None)),
                                "b_scales_view": str(getattr(b_s, "target", None)), **({} if dot is None else {"dot": dot.id})},
                   emitted=f"{fn.id} x{len(rows)}")


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
