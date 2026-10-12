"""The unfused norm chain: `forward_native` norms as ATen on an f32 copy."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, runs
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo, ViewRule
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, scalar_kind, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, active_target_profile
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16, aten
from verity_vllm.program.frontend.rules.vllm_bindings.pins import MEAN_PINS


# -- the unfused norm chain (`ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` as ATen on an f32 copy) [R15 dense, F-dense-05] -----------
#
# A model whose norm module runs `forward_native` (vLLM `GemmaRMSNorm`; `RMSNorm` off the BI branch) exports the norm as a chain of ATen
# ops on an f32 copy of the bf16 activation:
#
#     w1  = w.to(f32) + 1.0                     `WidenBf16ToF32` (no gates)  ->  `AddScalar` -> AddScalarBf16{N, C=1.0}     (GemmaRMSNorm's (1 + w))
#     x32 = x.to(f32)                           `WidenBf16ToF32` (no gates: the f32 tensor keeps the bf16 WORDS)
#     [s32 = x32 + res.to(f32); res' = s32.to(bf16)]  `AddWidened` -> AddWidenedBf16{N};  `NarrowF32ToBf16` -> NarrowF32ToBf16{N} (member "1")
#     v   = pow(x32 | s32, 2)                   `Square`      -> SquareBf16{N} | SquareF32{N}
#     m   = mean(v, [-1], keepdim=True)         `MeanDimBI`   -> MeanTriton{N}                  (BI `mean_kernel`; dispatch DERIVED, sources pinned)
#     r   = rsqrt(m + eps)                      `AddScalar`   -> AddScalarF32{N=1, C=eps};  `Rsqrt` -> RsqrtF32{N=1}
#     y   = (x32 | s32) * r                     `MulF32`      -> ScaleRowBf16{N} | ScaleRowF32{N}
#     y1  = y.to(f32) * w1                      `ViewDtype` (same width: a view);  `MulF32` -> MulVecF32{N}
#     out = y1.to(bf16)                         `NarrowF32ToBf16` -> NarrowF32ToBf16{N} (member "out" | "0")
#
# The widening is EXACT, so no gate is emitted for it: the f32 tensor's value collection keeps the bf16 words (`Value<16>` leaves under an
# f32 fx tensor) and every Definition that reads it widens internally (`SquareBf16`, `ScaleRowBf16`, `AddScalarBf16`, `AddWidenedBf16` —
# the SAME specialization ids the eager Match fold resolves the chain to, `gen_dense_gemma2_patterns`).  A reader that has no `*Bf16`
# variant (rsqrt, the row-vector multiply, the narrowing, mean) refuses a widened operand explicitly; any OTHER rule that reached such a
# value would fail the Builder's type check (`Value<16>` where the Definition takes `Value<32>`) — fail-closed by construction.  Rows:
# one Definition call per trailing-axis row in row-major order, exactly as the eager kernels' `PerRow` operands.  Member hints (`/out`,
# `/0`, `/1` on the narrowing rows; nothing on the interior) name the values the norm module returns, for the required-value manifest.


def widened_bf16(ctx, a) -> bool | None:
    """True when graph value `a` is an f32 tensor whose bound collection carries bf16 words (the transparent widening of
    `WidenBf16ToF32`, or a view of one), False for a genuine f32 value, None when `a` is not an f32 value collection."""
    if not isinstance(a, torch.fx.Node) or not dtype_is(ctx, a, torch.float32):
        return None
    v = ctx.val(a)
    if not isinstance(v, Coll):
        return None
    return shape_of(v.type)[1] == 16


def _f32_rows(ctx, a, name: str) -> str | None:
    """A refusal string unless `a` is an f32 value collection of rank >= 1 (widened or genuine)."""
    if not isinstance(a, torch.fx.Node):
        return f"{name} is not a graph value"
    v = ctx.val(a)
    if not isinstance(v, Coll):
        return f"{name} is not a value collection ({type(v).__name__})"
    if not dtype_is(ctx, a, torch.float32):
        return f"numerics_unregistered: {name} has dtype {ctx.dtype_of(a)}; the f32 chain kinds take float32 tensors"
    if not shape(ctx, a):
        return f"{name} is a scalar"
    return None


def _static_scalar(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


_CAST_OPS = {aten.to.dtype, aten._to_copy.default, aten.to.dtype_layout, aten.to.device}


class WidenBf16ToF32Rule(ViewRule):
    """`aten.to.dtype(x bf16, float32)` (also `_to_copy` / `to.dtype_layout` / `to.device` with dtype=float32): the EXACT widening of a
    bf16 tensor. No gate: the f32 tensor denotes the same bf16 words, and each reader binds the Definition that widens internally
    (`SquareBf16`, `ScaleRowBf16`, `AddScalarBf16`, `AddWidenedBf16`); a reader without such a variant refuses (rules below) or fails
    the Builder's type check. This is the frontend counterpart of the eager fold's `widened_source` (the widened allocation gets no refs)."""
    modes = frozenset({"profile"})
    info = RuleInfo("WidenBf16ToF32[aten]", 1, "view", applies_to="aten.to.dtype / _to_copy / to.dtype_layout / to.device (x bfloat16 -> float32)",
                    preconditions=("operand is a bf16 value collection of rank >= 1 (16-bit leaves)", "result dtype is float32 (same shape)"),
                    evidence_domain="structural: an exact conversion carries no rounding; readers bind the *Bf16 kinds (Value<16> leaves under an f32 tensor)")

    def targets(self):
        return set(_CAST_OPS)

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return f"operand is not a value collection ({type(ctx.val(x)).__name__ if isinstance(x, torch.fx.Node) else type(x).__name__})"
        if not (dtype_is(ctx, x, BF16) and dtype_is(ctx, node, torch.float32)):
            return f"{ctx.dtype_of(x)} -> {ctx.dtype_of(node)} is not the bfloat16 -> float32 widening"
        if shape_of(ctx.val(x).type)[1] != 16:
            return "operand collection does not carry 16-bit words"
        if not shape(ctx, x) or shape(ctx, node) != shape(ctx, x):
            return f"shape {shape(ctx, x)} -> {shape(ctx, node)} is not a same-shape conversion"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        ctx.define(node, ctx.coll(x), view_of=x)
        ctx.record(self, node, {"shape": list(shape(ctx, x)), "widened_words": 16}, emitted="transparent widening (no gates; readers bind *Bf16 kinds)")


class NarrowF32ToBf16Rule(OpRule):
    """`aten.to.dtype(y f32 [.., N], bfloat16)` -> `narrow_f32_bf16{N}` (NarrowF32ToBf16_v1: `cvt.rn.bf16.f32`, NaN -> 0x7FFF) per row.
    A genuine f32 operand only: narrowing a widened bf16 view is the identity on the words and is not a kernel vLLM runs in this chain
    (refused). Row names carry the member the norm module returns (`/1` for the residual `s32.to(bf16)`, `/0` for the normed output next
    to a residual, `/out` alone) — the required-value manifest's hint."""
    modes = frozenset({"profile"})
    info = RuleInfo("NarrowF32ToBf16[aten]", 1, "op-binding", applies_to="aten.to.dtype / _to_copy / to.dtype_layout / to.device (y float32 -> bfloat16)",
                    preconditions=("operand is a genuine f32 value collection of rank >= 1 (32-bit leaves; a widened bf16 view is refused)",
                                   "result dtype is bfloat16 (same shape)"),
                    evidence_domain="structural; numerics: NarrowF32ToBf16_v1 (F2fpBf16 per element)")

    def targets(self):
        return set(_CAST_OPS)

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return f"operand is not a value collection ({type(ctx.val(x)).__name__ if isinstance(x, torch.fx.Node) else type(x).__name__})"
        if not (dtype_is(ctx, x, torch.float32) and dtype_is(ctx, node, BF16)):
            return f"{ctx.dtype_of(x)} -> {ctx.dtype_of(node)} is not the float32 -> bfloat16 narrowing"
        if widened_bf16(ctx, x):
            return "numerics_unregistered: narrowing a widened bf16 view (the identity on the words) is not a kernel of the modelled chain"
        if not shape(ctx, x) or shape(ctx, node) != shape(ctx, x):
            return f"shape {shape(ctx, x)} -> {shape(ctx, node)} is not a same-shape conversion"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        fn = ctx.kind("narrow_f32_bf16", node, N=N)
        member = self._member(node, ctx)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}/{member}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "member": member}, emitted=f"{fn.id} x{len(rows)}")

    @staticmethod
    def _member(node, ctx) -> str:
        """`1` when the narrowed value is the residual sum (`AddWidened` fired on the operand), `0` when the module also produced one
        (fused form: (normed, residual)), else `out` (a module returning one tensor)."""
        adds = ctx.state.get("widened_adds", {})
        if node.args[0] in adds:
            return "1"
        mp = ctx.module_path(node)
        if mp is not None and any(ctx.module_path(a) == mp for a in adds):
            return "0"
        return "out"


class SquareRule(OpRule):
    """`aten.pow.Tensor_Scalar(x f32 [.., N], 2)` -> `square_bf16{N}` (x a widened bf16 view) | `square_f32{N}` per row: ATen's CUDA
    `pow_tensor_scalar_kernel` special-cases exponent 2 as `base * base` (one FMUL, RN). Any other exponent is refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("Square[aten]", 1, "op-binding", applies_to="aten.pow.Tensor_Scalar(x float32 [.., N], 2)",
                    preconditions=("x is an f32 value collection of rank >= 1 (widened bf16 view -> SquareBf16, genuine f32 -> SquareF32)",
                                   "exponent is the static scalar 2"),
                    evidence_domain="structural; numerics: SquareBf16_v1 / SquareF32_v1 (one FMUL per element)")

    def targets(self):
        return {aten.pow.Tensor_Scalar}

    def applies(self, node, ctx):
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        e = _static_scalar(node.args[1]) if len(node.args) > 1 else None
        if e != 2.0:
            return f"numerics_unregistered: exponent {node.args[1]!r} is not 2 (only the squared form has a registered kind)"
        return True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        w = widened_bf16(ctx, x)
        fn = ctx.kind("square_bf16" if w else "square_f32", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "operand_widened_bf16": bool(w)}, emitted=f"{fn.id} x{len(rows)}")


class MeanDimBIRule(OpRule):
    """`aten.mean.dim(x f32 [.., N], [-1], keepdim=True)` under batch invariance -> `mean_triton{N}` (MeanTriton_v1: the BI Triton
    `mean_kernel`, BLOCK_SIZE=1024, 4 warps) per row, output `[.., 1]`. The dispatch is DERIVED (see `MEAN_PINS`): the export must show
    BI Triton launches (the graph's own evidence that `vllm_is_batch_invariant()` holds in the construction process) and the installed
    vLLM's `mean_kernel` / `mean_dim` / `mean_batch_invariant` sources must hash to the pins; a genuine f32 operand only (no `MeanBf16`)."""
    modes = frozenset({"profile"})
    info = RuleInfo("MeanDimBI[aten]", 1, "op-binding", applies_to="aten.mean.dim(x float32 [.., N], dim=[-1], keepdim=True, dtype=None)",
                    preconditions=("x is a genuine f32 value collection of rank >= 1 (a widened bf16 view is refused)",
                                   "dim is the last axis only; keepdim=True; no dtype argument",
                                   "export facts show batch-invariant Triton launches in this graph (derived, never assumed)",
                                   "installed vLLM's mean_kernel src sha, mean_dim and mean_batch_invariant source shas == MEAN_PINS; "
                                   "enable_batch_invariant_mode registers aten::mean.dim"),
                    evidence_domain="structural; numerics: MeanTriton_v1 (BI mean_kernel reduction order, div.full.f32)")

    def targets(self):
        return {aten.mean.dim}

    @staticmethod
    def _dispatch_ok() -> str | None:
        f = _OBSERVED["export_facts"]
        if f is None:
            return "batch-invariant dispatch was not observed at construction (observe_export_facts before derive) — refusing rather than assuming"
        ev = f.get("evidence") or {}
        if not ev.get("bi_triton_kernels_launched"):
            return ("numerics_unregistered: the export shows no batch-invariant Triton launch, so `aten::mean.dim` is not known to dispatch to "
                    f"`mean_batch_invariant` (evidence {ev!r}); the eager ATen mean reduction has no registered kind")
        mo = ev.get("mean_dim_override") or {}
        if mo.get("error") or not mo:
            return f"batch_invariant.mean_kernel not observable in this process ({mo.get('error', 'no mean_dim_override facts')}) — refusing"
        for k in ("kernel", "src_sha256", "mean_dim_src_sha256", "mean_batch_invariant_src_sha256"):
            if mo.get(k) != MEAN_PINS[k]:
                return f"numerics_unregistered: installed {k}={mo.get(k)!r} differs from the pinned {MEAN_PINS[k]!r} (MeanTriton_v1 models the pinned kernel)"
        if mo.get("enable_batch_invariant_mode_registers_mean_dim") is not True:
            return "numerics_unregistered: the installed enable_batch_invariant_mode does not register aten::mean.dim"
        return None

    def applies(self, node, ctx):
        x = node.args[0]
        r = _f32_rows(ctx, x, "x")
        if r:
            return r
        if widened_bf16(ctx, x):
            return "numerics_unregistered: mean of a widened bf16 view has no Definition (MeanTriton_v1 reads f32)"
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        rank = len(shape(ctx, x))
        if not isinstance(dims, (list, tuple)) or len(dims) != 1 or int(dims[0]) not in (-1, rank - 1):
            return f"dim {dims!r} is not the last axis"
        if keep is not True:
            return "keepdim is not True"
        if node.kwargs.get("dtype") is not None or (len(node.args) > 3 and node.args[3] is not None):
            return "an explicit dtype argument is not the modelled call"
        if shape(ctx, node) != tuple(shape(ctx, x)[:-1]) + (1,):
            return f"output shape {shape(ctx, node)} is not [.., 1]"
        return self._dispatch_ok() or True

    def emit(self, node, ctx):
        x = node.args[0]
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        fn = ctx.kind("mean_triton", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), "pins": MEAN_PINS,
                                "mean_dim_override_observed": (_OBSERVED["export_facts"].get("evidence") or {}).get("mean_dim_override")},
                   emitted=f"{fn.id} x{len(rows)}")


class RsqrtRule(OpRule):
    """`aten.rsqrt.default(x f32 [.., N])` -> `rsqrt_f32{N}` (RsqrtF32_v1: `rsqrt.approx.f32`, MUFU.RSQ + range scaling) per row; a
    genuine f32 operand only."""
    modes = frozenset({"profile"})
    info = RuleInfo("Rsqrt[aten]", 1, "op-binding", applies_to="aten.rsqrt.default(x float32 [.., N])",
                    preconditions=("x is a genuine f32 value collection of rank >= 1 (a widened bf16 view is refused)",),
                    evidence_domain="structural; numerics: RsqrtF32_v1 (RsqrtApprox_v1 per element)")

    def targets(self):
        return {aten.rsqrt.default}

    def applies(self, node, ctx):
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        if widened_bf16(ctx, node.args[0]):
            return "numerics_unregistered: rsqrt of a widened bf16 view has no Definition (RsqrtF32_v1 reads f32)"
        return True

    def emit(self, node, ctx):
        rows, lead = rows_of(ctx.coll(node.args[0]))
        N = rows[0].type.n
        fn = ctx.kind("rsqrt_f32", node, N=N)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class AddScalarRule(OpRule):
    """`aten.add.Tensor(x f32 [.., N], C)` with a static Python scalar -> `add_scalar_bf16{N,C}` (x a widened bf16 view: GemmaRMSNorm's
    `weight.float() + 1.0`) | `add_scalar_f32{N,C}` (`variance + eps`) per row; ATen's CUDA `add_kernel` in opmath f32 with the scalar
    narrowed once (one FADD, RN). The constant is part of the kind. `alpha` must be 1.  When x is a weight and the declared target's
    `weight_only_calls` is `once`, a forward whose add reads the same weight leaves as an earlier forward's reads that forward's value."""
    modes = frozenset({"profile"})
    info = RuleInfo("AddScalar[aten]", 1, "op-binding", applies_to="aten.add.Tensor(x float32 [.., N], C: static scalar), alpha == 1",
                    preconditions=("x is an f32 value collection of rank >= 1 (widened -> AddScalarBf16, genuine -> AddScalarF32)",
                                   "the second operand is a static Python scalar", "alpha == 1"),
                    evidence_domain="structural; numerics: AddScalarBf16_v1 / AddScalarF32_v1 (one FADD per element)")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        if len(node.args) != 2:
            return "not a two-operand add"
        r = _f32_rows(ctx, node.args[0], "x")
        if r:
            return r
        if _static_scalar(node.args[1]) is None:
            return "the second operand is not a static Python scalar"
        if node.kwargs.get("alpha", 1) not in (1, 1.0):
            return "alpha != 1"
        return True

    def emit(self, node, ctx):
        x, c = node.args[0], _static_scalar(node.args[1])
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        w = widened_bf16(ctx, x)
        fn, value = scalar_kind(ctx, "add_scalar_bf16" if w else "add_scalar_f32", node, N, c)
        once = ctx.weight_of(x) is not None and active_target_profile().effective_weight_only_calls() == "once"
        key = (fn.id, tuple(tuple(r) for r in runs(ctx.coll(x).refs))) if once else None
        issued = ctx.state.setdefault("weight_only_calls", {}) if once else {}
        if key in issued:
            ctx.define(node, issued[key][0])
            ctx.record(self, node, {"N": N, "C": c, "rows": len(rows), "operand_widened_bf16": bool(w), "weight_only_calls": "once"},
                       emitted=f"{fn.id} x0 (reads {issued[key][1]})")
            return
        outs = [ctx.call(fn, r, *value, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        if once:
            issued[key] = (ctx.val(node), node.name)
        if w and ctx.weight_of(x) is not None:
            ctx.bind_weight_role(x, self, "norm.weight_plus_one", N=N, C=c)
        ctx.record(self, node, {"N": N, "C": c, "rows": len(rows), "operand_widened_bf16": bool(w)}, emitted=f"{fn.id} x{len(rows)}")


class AddWidenedRule(OpRule):
    """`aten.add.Tensor(x f32 [.., N], r f32 [.., N])`, both the exact widening of bf16 tensors (`fused_add_rms_norm`'s eager fallback:
    `x.to(f32) + residual.to(f32)`) -> `add_widened_bf16{N}` (AddWidenedBf16_v1: one FADD, RN) per row. An f32 + f32 add whose operands
    are not both widened has no Definition and is refused; `alpha` must be 1."""
    modes = frozenset({"profile"})
    info = RuleInfo("AddWidened[aten]", 1, "op-binding", applies_to="aten.add.Tensor(x float32 [.., N], r float32 [.., N]) both widened bf16 views, alpha == 1",
                    preconditions=("both operands are f32 value collections of one shape whose leaves are bf16 words (WidenBf16ToF32 views)",
                                   "alpha == 1"),
                    evidence_domain="structural; numerics: AddWidenedBf16_v1 (one FADD per element)")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        if len(node.args) != 2:
            return "not a two-operand add"
        for nm, a in (("x", node.args[0]), ("r", node.args[1])):
            r = _f32_rows(ctx, a, nm)
            if r:
                return r
        if shape(ctx, node.args[0]) != shape(ctx, node.args[1]):
            return "operands differ in shape (broadcast is not a bound rule)"
        if node.kwargs.get("alpha", 1) not in (1, 1.0):
            return "alpha != 1"
        if not (widened_bf16(ctx, node.args[0]) and widened_bf16(ctx, node.args[1])):
            return "numerics_unregistered: f32 + f32 add whose operands are not both widened bf16 views has no Definition"
        return True

    def emit(self, node, ctx):
        ra, lead = rows_of(ctx.coll(node.args[0]))
        rb, _ = rows_of(ctx.coll(node.args[1]))
        N = ra[0].type.n
        fn = ctx.kind("add_widened_bf16", node, N=N)
        outs = [ctx.call(fn, x, y, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (x, y) in enumerate(zip(ra, rb))]
        ctx.define(node, from_rows(outs, lead))
        ctx.state.setdefault("widened_adds", {})[node] = ctx.module_path(node)   # `NarrowF32ToBf16` names the residual member from this
        ctx.record(self, node, {"N": N, "rows": len(ra)}, emitted=f"{fn.id} x{len(ra)}")


class MulF32Rule(OpRule):
    """The f32 `aten.mul.Tensor` forms of the unfused norm: `x[.., N] * s[.., 1]` (a row scaled by its own f32 scalar) ->
    `scale_row_bf16{N}` (x a widened bf16 view) | `scale_row_f32{N}`; `x[.., N] * w[N]` (a shared f32 row vector, `[1,..,1,N]` too) ->
    `mul_vec_f32{N}` (a genuine f32 x only). One FMUL per element, RN. Any other broadcast is refused."""
    modes = frozenset({"profile"})
    info = RuleInfo("MulF32[aten]", 1, "op-binding", applies_to="aten.mul.Tensor(x float32 [.., N], s float32 [.., 1]) ; aten.mul.Tensor(x float32 [.., N], w float32 [N])",
                    preconditions=("x is an f32 value collection of rank >= 2 (row-scale: widened or genuine; row-vector: genuine only)",
                                   "the other operand is f32: [.., 1] with x's leading shape (row scale) or [N] / [1,..,1,N] (row vector)"),
                    evidence_domain="structural; numerics: ScaleRowBf16_v1 / ScaleRowF32_v1 / MulVecF32_v1 (one FMUL per element)")

    def targets(self):
        return {aten.mul.Tensor}

    def _form(self, node, ctx):
        """-> ("scale", x, s) | ("vec", x, w) | a refusal string."""
        if len(node.args) != 2:
            return "not a two-operand mul"
        x, y = node.args
        for nm, a in (("x", x), ("y", y)):
            r = _f32_rows(ctx, a, nm)
            if r:
                return r
        sx, sy = shape(ctx, x), shape(ctx, y)
        if len(sx) < 2:
            return f"x {sx} has no rows (rank >= 2 needed)"
        N = sx[-1]
        if sy == tuple(sx[:-1]) + (1,):
            if widened_bf16(ctx, y):
                return "numerics_unregistered: the per-row scale is a widened bf16 view (the chain's scale is the f32 rsqrt)"
            return "scale", x, y
        if sy == (N,) or (len(sy) >= 1 and sy[-1] == N and all(s == 1 for s in sy[:-1])):
            if widened_bf16(ctx, x):
                return "numerics_unregistered: row-vector mul of a widened bf16 view has no Definition (MulVecF32_v1 reads f32)"
            if widened_bf16(ctx, y):
                return "numerics_unregistered: the row vector is a widened bf16 view (the chain's vector is the f32 (1 + w))"
            return "vec", x, y
        return f"numerics_unregistered: f32 mul broadcast {sx} * {sy} is neither [.., N] * [.., 1] nor [.., N] * [N]"

    def applies(self, node, ctx):
        f = self._form(node, ctx)
        return f if isinstance(f, str) else True

    def emit(self, node, ctx):
        form, x, y = self._form(node, ctx)
        rows, lead = rows_of(ctx.coll(x))
        N = rows[0].type.n
        if form == "scale":
            srows, _ = rows_of(ctx.coll(y))
            w = widened_bf16(ctx, x)
            fn = ctx.kind("scale_row_bf16" if w else "scale_row_f32", node, N=N)
            outs = [ctx.call(fn, r, s, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, (r, s) in enumerate(zip(rows, srows))]
            det = {"form": "row_scale", "operand_widened_bf16": bool(w)}
        else:
            wv = ctx.coll(y)
            if len(shape_of(wv.type)[0]) != 1:
                wv = wv.reshape(_tensor_type((N,), 32))
            fn = ctx.kind("mul_vec_f32", node, N=N)
            outs = [ctx.call(fn, r, wv, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
            det = {"form": "row_vector"}
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"N": N, "rows": len(rows), **det}, emitted=f"{fn.id} x{len(rows)}")
