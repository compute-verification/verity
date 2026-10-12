"""[GEN dense / ov-dense] Gemma-2's eager-unfused ATen chain -> the lane's quarantined Definitions
(`verity_vllm.program.registry.quarantine.dense`, one `SEMANTIC_ID` per module, each difftested on the pod).

vLLM d9105ea80 runs `Gemma2ForCausalLM` in eager mode with `VLLM_BATCH_INVARIANT=1`; nothing of the norm / GeGLU / softcap
is fused into the b1 kernels the base profile knows, so the raw log shows the ATen ops themselves (one decode step, layer L):

    GemmaRMSNorm.forward_native (vllm/model_executor/layers/layernorm.py), residual-free form (layer 0 input norm, every
    post_attention / post_feedforward norm):
        aten.to.dtype(w[N] bf16 -> f32)           transparent widening (base rule; consumers re-point to the bf16 Root)
        aten.add.Tensor(w_f32, 1.0)                -> AddScalarBf16{N, C=1.0}     rows=1  (Shared(w), Shared(out))
        aten.to.dtype(x[M,N] bf16 -> f32)         transparent widening
        aten.pow.Tensor_Scalar(x_f32, 2)           -> SquareBf16{N}               rows=M
        aten.mean.dim(x2, [-1], True) {mean_kernel}-> MeanTriton{N}               rows=M  ([M,N] -> [M,1])
        aten.add.Tensor(var[M,1], 1e-6)            -> AddScalarF32{N=1, C=EPS}    rows=M
        aten.rsqrt.default(var)                    -> RsqrtF32{N=1}               rows=M
        aten.mul.Tensor(x_f32, r[M,1])             -> ScaleRowBf16{N}             rows=M
        aten.to.dtype(f32 -> f32)                  transparent alias (returned self)
        aten.mul.Tensor(xs, w1[N] f32)             -> MulVecF32{N}                rows=M
        aten.to.dtype(f32 -> bf16)                 -> NarrowF32ToBf16{N}          rows=M
    residual form (`ir.ops.fused_add_rms_norm` fallback: every input / pre_feedforward norm from layer 1 on, the final norm):
        aten.to.dtype(x bf16 -> f32), aten.to.dtype(res bf16 -> f32)      two widenings
        aten.add.Tensor(x_f32, res_f32)            -> AddWidenedBf16{N}           rows=M  (the new residual stream, f32)
        aten.to.dtype(sum f32 -> bf16)             -> NarrowF32ToBf16{N}          (the residual handed to the next layer)
        aten.pow.Tensor_Scalar(sum, 2)             -> SquareF32{N}
        ... mean / add eps / rsqrt as above ...
        aten.mul.Tensor(sum, r)                    -> ScaleRowF32{N}
        ... MulVecF32 / NarrowF32ToBf16 as above ...
    embedding scale (Gemma2Model.forward):  aten.mul.Tensor(e[M,H] bf16, normalizer[] bf16 Root) -> Bf16MulScalarTensor{H}
    GeGLU:                                   _C.gelu_tanh_and_mul(out[M,I], gu[M,2I])                -> GeluTanhMul{I}
    final-logit softcap (LogitsProcessor):   aten.div.Tensor(logits, 30.0) -> Bf16DivScalar{V, 30.0};
                                             aten.tanh.default -> Bf16Tanh{V}; aten.mul.Tensor(t, 30.0) -> Bf16MulScalar{V, 30.0}

Every pattern here checks dtypes / layouts / the scalar and resolves ONLY the shapes above; anything else falls through to the
base patterns (`None`) or is reported with a reason.  `widened_source` (base helper) turns an f32 operand that is the exact
widening of a bf16 view back into that view, so the `*Bf16` Definitions read the narrow source and the widened allocation
receives no refs (an unexplained reader of it fails loudly in the fold, as designed).

NOT here (and stated in the lane report): the FA2 softcap attention body (`Is_softcap=true` -> MUFU.TANH, MECHANISM GAP /
new-primitive request) and the root class of `model.normalizer` (owner decision; the operand is resolved, the Root is not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry.quarantine.dense import (add_scalar_bf16, add_scalar_f32, add_widened_bf16, bf16_div_scalar,
                                                bf16_mul_scalar, bf16_mul_scalar_tensor, bf16_tanh, gelu_tanh_mul, mean_triton,
                                                mul_vec_f32, narrow_f32_bf16, rsqrt_f32, scale_row_bf16, scale_row_f32,
                                                square_bf16, square_f32)

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (BiasAdd, _check_pins, as_float, attribution_for, base_op, bound_args, hint_of, is_row_major,
                                     op_name, triton_name, unsupported, widened_source)
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode

KERNELS_ELEMENTWISE = ("vectorized_elementwise_kernel", "elementwise_kernel", "unrolled_elementwise_kernel")


def _scalar(v: Any) -> float | None:
    """A Python number argument (never a ViewDesc / bool)."""
    if isinstance(v, bool) or isinstance(v, ViewDesc):
        return None
    return as_float(v)


def _rows_n(v: ViewDesc) -> tuple[int, int] | None:
    """(rows, N) for a row-major [M,N] view or a contiguous [N] view (rows=1); None otherwise."""
    if v.rank == 2 and is_row_major(v):
        return v.shape[0], v.shape[1]
    if v.rank == 1 and v.strides == (1,):
        return 1, v.shape[0]
    return None


def _operand(v: ViewDesc):
    """`PerRow` for a [M,N] view, `Shared` for a [N] view (one row)."""
    return PerRow(v) if v.rank == 2 else Shared(v)


def _compute(node: TreeNode, ctx: StepContext, spec, rows: int, operands: tuple, outputs: tuple, name: str, kernels=KERNELS_ELEMENTWISE,
             *hint_extra: str, consumed: tuple[int, ...] | None = None) -> Compute:
    return Compute(spec, rows, operands, outputs, consumed if consumed is not None else (node.id,), hint=hint_of(node, *hint_extra),
                   kernels=kernels, attribution=attribution_for(ctx, rows, node.phase) if rows > 1 or operands and isinstance(operands[0], PerRow) else (),
                   pattern=name)


@dataclass
class ScalarAddF32:
    """`aten.add.Tensor(x f32 [M,N] | [N], C: float)` -> `AddScalarF32{N,C}` / `AddScalarBf16{N,C}` when x is the exact widening
    of a bf16 view (GemmaRMSNorm: `weight.float() + 1.0`, `variance + eps`)."""

    name: str = "dense.add_scalar"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.add":
            return None
        a = bound_args(node)
        x, c = a.get("self", a.get("arg0")), _scalar(a.get("other", a.get("arg1")))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or c is None or not outs or x.dtype != "float32":
            return None
        if a.get("alpha", 1) not in (1, 1.0):
            return unsupported(node, "aten.add scalar with alpha != 1", alpha=a.get("alpha"))
        geo = _rows_n(x)
        out = outs[0]
        if geo is None or out.shape != x.shape or out.dtype != "float32" or not (is_row_major(out) if out.rank == 2 else out.strides == (1,)):
            return unsupported(node, "scalar add over views that are not row-major [M,N] / contiguous [N] f32", x=repr(x), out=repr(out))
        rows, N = geo
        src = widened_source(x, ctx)
        if src is not None:
            spec = bind(add_scalar_bf16.AddScalarBf16, N=N, C=c)
            roots = ctx.roots_of(src)
            return _compute(node, ctx, spec, rows, (_operand(src),), (_operand(out),), self.name, KERNELS_ELEMENTWISE,
                            roots[0].name if roots else "add_scalar")
        return _compute(node, ctx, bind(add_scalar_f32.AddScalarF32, N=N, C=c), rows, (_operand(x),), (_operand(out),), self.name)


@dataclass
class SquarePow:
    """`aten.pow.Tensor_Scalar(x f32 [M,N], 2)` -> `SquareF32{N}` / `SquareBf16{N}` (x the widening of a bf16 view)."""

    name: str = "dense.square"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.pow":
            return None
        a = bound_args(node)
        x, e = a.get("self", a.get("arg0")), _scalar(a.get("exponent", a.get("arg1")))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or e is None or not outs or x.dtype != "float32":
            return None
        if e != 2.0:
            return unsupported(node, "aten.pow with an exponent other than 2", exponent=e)
        geo = _rows_n(x)
        out = outs[0]
        if geo is None or x.rank != 2 or out.shape != x.shape or out.dtype != "float32" or not is_row_major(out):
            return unsupported(node, "pow over views that are not row-major [M,N] f32", x=repr(x), out=repr(out))
        M, N = geo
        src = widened_source(x, ctx)
        if src is not None:
            return _compute(node, ctx, bind(square_bf16.SquareBf16, N=N), M, (PerRow(src),), (PerRow(out),), self.name)
        return _compute(node, ctx, bind(square_f32.SquareF32, N=N), M, (PerRow(x),), (PerRow(out),), self.name)


@dataclass
class MeanDimTriton:
    """`aten.mean.dim(x f32 [M,N], [-1], keepdim=True)` whose single child is the `TritonLaunch batch_invariant.mean_kernel`
    (grid=[M], BLOCK_SIZE=1024, num_warps=4, K=1) -> `MeanTriton{N}` rows=M, `[M,N] -> [M,1]`."""

    name: str = "dense.mean_dim.triton_mean_kernel"
    block_size: int = 1024
    num_warps: int = 4

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.mean":
            return None
        a = bound_args(node)
        x = a.get("self", a.get("arg0"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs or x.dtype != "float32":
            return None
        dims = a.get("dim", a.get("arg1"))
        keep = a.get("keepdim", a.get("arg2"))
        if x.rank != 2 or not is_row_major(x):
            return unsupported(node, "mean over a view that is not row-major [M,N] f32", x=repr(x))
        M, N = x.shape
        if list(dims or []) not in ([-1], [1]) or keep is not True:
            return unsupported(node, "mean.dim that is not a keepdim last-axis reduction", dim=dims, keepdim=keep)
        out = outs[0]
        if out.shape != (M, 1) or out.dtype != "float32":
            return unsupported(node, "mean.dim output is not [M,1] f32", out=repr(out))
        launches = [c for c in node.children if c.is_triton]
        if len(launches) != 1 or len(node.children) != 1 or triton_name(launches[0]) != "mean_kernel":
            return unsupported(node, "mean.dim is not implemented by exactly one mean_kernel launch", children=[c.op for c in node.children])
        k = launches[0]
        ka = bound_args(k)
        consts = dict(k.enter.constexprs)
        got = {"BLOCK_SIZE": consts.get("BLOCK_SIZE", ka.get("BLOCK_SIZE")), "num_warps": k.enter.num_warps, "M": ka.get("M"), "N": ka.get("N"),
               "K": ka.get("K"), "input_stride1": ka.get("input_stride1")}
        bad = _check_pins(node, got, {"BLOCK_SIZE": self.block_size, "num_warps": self.num_warps, "M": M, "N": N, "K": 1, "input_stride1": 1})
        if bad is not None:
            return bad
        if k.enter.grid and tuple(k.enter.grid) != (M,):
            return unsupported(node, "mean_kernel grid is not one program per row", grid=k.enter.grid, M=M)
        return _compute(node, ctx, bind(mean_triton.MeanTriton, N=N), M, (PerRow(x),), (PerRow(out),), self.name, ("mean_kernel",),
                        consumed=tuple(sorted(node.ids())))


@dataclass
class RsqrtF32Pattern:
    """`aten.rsqrt.default(x f32 [M,N])` -> `RsqrtF32{N}` rows=M (GemmaRMSNorm: N = 1)."""

    name: str = "dense.rsqrt"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.rsqrt":
            return None
        a = bound_args(node)
        x = a.get("self", a.get("arg0"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs or x.dtype != "float32":
            return None
        geo = _rows_n(x)
        out = outs[0]
        if geo is None or out.shape != x.shape or out.dtype != "float32":
            return unsupported(node, "rsqrt over views that are not row-major [M,N] f32", x=repr(x), out=repr(out))
        rows, N = geo
        if widened_source(x, ctx) is not None:
            return unsupported(node, "rsqrt of a widened bf16 view has no Definition (RsqrtF32 reads f32)", x=repr(x))
        return _compute(node, ctx, bind(rsqrt_f32.RsqrtF32, N=N), rows, (_operand(x),), (_operand(out),), self.name)


@dataclass
class MulF32:
    """The three f32 `aten.mul.Tensor` forms of GemmaRMSNorm:
    `x[M,N] * s[M,1]` -> `ScaleRowF32{N}` / `ScaleRowBf16{N}` (x a widening);  `x[M,N] * w[N]` -> `MulVecF32{N}`."""

    name: str = "dense.mul_f32"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.mul":
            return None
        a = bound_args(node)
        x, y = a.get("self", a.get("arg0")), a.get("other", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(y, ViewDesc) or not outs:
            return None
        if not (x.dtype == y.dtype == "float32"):
            return None
        out = outs[0]
        if x.rank != 2 or not is_row_major(x) or out.shape != x.shape or out.dtype != "float32" or not is_row_major(out):
            return unsupported(node, "f32 mul over views that are not row-major [M,N]", x=repr(x), y=repr(y), out=repr(out))
        M, N = x.shape
        if y.rank == 2 and y.shape == (M, 1):
            src = widened_source(x, ctx)
            if src is not None:
                return _compute(node, ctx, bind(scale_row_bf16.ScaleRowBf16, N=N), M, (PerRow(src), PerRow(y)), (PerRow(out),), self.name)
            return _compute(node, ctx, bind(scale_row_f32.ScaleRowF32, N=N), M, (PerRow(x), PerRow(y)), (PerRow(out),), self.name)
        if y.rank == 1 and y.shape == (N,) and y.strides == (1,):
            if widened_source(x, ctx) is not None:
                return unsupported(node, "row-vector mul of a widened bf16 view has no Definition (MulVecF32 reads f32)", x=repr(x))
            return _compute(node, ctx, bind(mul_vec_f32.MulVecF32, N=N), M, (PerRow(x), Shared(y)), (PerRow(out),), self.name)
        return unsupported(node, "f32 mul broadcast shape is neither [M,N]*[M,1] nor [M,N]*[N]", x=repr(x), y=repr(y))


@dataclass
class AddWidened:
    """`aten.add.Tensor(x f32 [M,N], r f32 [M,N])`, both the exact widening of bf16 views -> `AddWidenedBf16{N}` rows=M
    (the residual add of `ir.ops.fused_add_rms_norm`'s eager fallback)."""

    name: str = "dense.add_widened"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.add":
            return None
        a = bound_args(node)
        x, r = a.get("self", a.get("arg0")), a.get("other", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not isinstance(r, ViewDesc) or not outs or not (x.dtype == r.dtype == "float32"):
            return None
        if a.get("alpha", 1) not in (1, 1.0):
            return unsupported(node, "aten.add with alpha != 1", alpha=a.get("alpha"))
        out = outs[0]
        if x.rank != 2 or not is_row_major(x) or r.shape != x.shape or not is_row_major(r) or out.shape != x.shape or out.dtype != "float32":
            return unsupported(node, "f32 add over views that are not row-major [M,N] of one shape", x=repr(x), r=repr(r), out=repr(out))
        xs, rs = widened_source(x, ctx), widened_source(r, ctx)
        if xs is None or rs is None:
            return unsupported(node, "f32 + f32 add whose operands are not both widened bf16 views has no Definition", x=repr(x), r=repr(r))
        M, N = x.shape
        return _compute(node, ctx, bind(add_widened_bf16.AddWidenedBf16, N=N), M, (PerRow(xs), PerRow(rs)), (PerRow(out),), self.name)


@dataclass
class NarrowToBf16:
    """`aten.to.dtype(x f32 [M,N] | [N], bfloat16)` (a fresh bf16 allocation of the same shape) -> `NarrowF32ToBf16{N}`.
    Registered ahead of `TransparentAten`, which reports a float dtype change as Unsupported."""

    name: str = "dense.narrow_to_bf16"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) not in ("aten.to", "aten._to_copy"):
            return None
        ins, outs = node.input_views(), node.output_views()
        if len(ins) != 1 or len(outs) != 1:
            return None
        x, out = ins[0], outs[0]
        if x.dtype != "float32" or out.dtype != "bfloat16":
            return None
        geo = _rows_n(x)
        if geo is None or out.shape != x.shape or not (is_row_major(out) if out.rank == 2 else out.strides == (1,)):
            return unsupported(node, "f32 -> bf16 narrowing over views that are not row-major [M,N] / contiguous [N]", x=repr(x), out=repr(out))
        rows, N = geo
        return _compute(node, ctx, bind(narrow_f32_bf16.NarrowF32ToBf16, N=N), rows, (_operand(x),), (_operand(out),), self.name)


@dataclass
class EmbedScale:
    """`aten.mul.Tensor(e[M,N] bf16, s[] bf16)` with a 0-d device tensor scalar (Gemma2Model: `embed * self.normalizer`)
    -> `Bf16MulScalarTensor{N}` rows=M, `(PerRow(e), Shared(s))`.  The Root living in `s` (`model.normalizer`) keeps whatever
    class the root policy gives it; this pattern only resolves the operand."""

    name: str = "dense.embed_scale"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.mul":
            return None
        a = bound_args(node)
        e, s = a.get("self", a.get("arg0")), a.get("other", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(e, ViewDesc) or not isinstance(s, ViewDesc) or not outs or s.rank != 0:
            return None
        if not (e.dtype == s.dtype == "bfloat16"):
            return None
        out = outs[0]
        if e.rank != 2 or not is_row_major(e) or out.shape != e.shape or out.dtype != "bfloat16" or not is_row_major(out):
            return unsupported(node, "bf16 scalar-tensor mul over views that are not row-major [M,N]", e=repr(e), out=repr(out))
        M, N = e.shape
        roots = ctx.roots_of(s)
        return _compute(node, ctx, bind(bf16_mul_scalar_tensor.Bf16MulScalarTensor, N=N), M, (PerRow(e), Shared(s)), (PerRow(out),), self.name,
                        KERNELS_ELEMENTWISE, roots[0].name if roots else "scalar_tensor")


@dataclass
class GeluTanhAndMul:
    """`_C.gelu_tanh_and_mul(out[M,I], gu[M,2I])` -> `GeluTanhMul{I}` rows=M (mirrors the base `SiluAndMul`)."""

    name: str = "dense.act.gelu_tanh_and_mul"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "gelu_tanh_and_mul":
            return None
        a = bound_args(node)
        out = a.get("result", a.get("out", a.get("arg0")))
        x = a.get("input", a.get("x", a.get("arg1")))
        if not isinstance(out, ViewDesc) or not isinstance(x, ViewDesc):
            return unsupported(node, "gelu_tanh_and_mul arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if x.rank != 2 or out.rank != 2 or x.shape[0] != out.shape[0] or x.shape[1] != 2 * out.shape[1] or not is_row_major(x) or not is_row_major(out):
            return unsupported(node, "gelu_tanh_and_mul views are not row-major [M,2I] -> [M,I]", x=repr(x), out=repr(out))
        if not (x.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "GeluTanhMul is bf16", dtypes=[x.dtype, out.dtype])
        if getattr(ctx.profile, "registered_values", False):
            return unsupported(node, "registered_values: GeluTanhMul_v3 reads the registered table gelu_tanh_bf16, which the fold "
                               "does not supply yet")
        M, I = out.shape
        return _compute(node, ctx, bind(gelu_tanh_mul.GeluTanhMul, I=I), M, (PerRow(x),), (PerRow(out),), self.name, ("act_and_mul_kernel",),
                        "gelu_tanh_and_mul")


@dataclass
class Bf16ScalarChain:
    """The final-logit softcap: `aten.div.Tensor(x bf16 [M,N], C)` -> `Bf16DivScalar{N,C}`; `aten.tanh.default(x bf16)` -> `Bf16Tanh{N}`;
    `aten.mul.Tensor(x bf16 [M,N], C)` -> `Bf16MulScalar{N,C}` (C a Python float)."""

    name: str = "dense.bf16_scalar_chain"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        b = base_op(node.op)
        if b not in ("aten.div", "aten.mul", "aten.tanh"):
            return None
        a = bound_args(node)
        x = a.get("self", a.get("arg0"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs or x.dtype != "bfloat16":
            return None
        c = None
        if b != "aten.tanh":
            c = _scalar(a.get("other", a.get("arg1")))
            if c is None:
                return None
            if b == "aten.div" and a.get("rounding_mode") not in (None,):
                return unsupported(node, "aten.div with a rounding mode", rounding_mode=a.get("rounding_mode"))
        out = outs[0]
        if x.rank != 2 or not is_row_major(x) or out.shape != x.shape or out.dtype != "bfloat16" or not is_row_major(out):
            return unsupported(node, "bf16 scalar op over views that are not row-major [M,N]", x=repr(x), out=repr(out))
        M, N = x.shape
        if b == "aten.div":
            spec = bind(bf16_div_scalar.Bf16DivScalar, N=N, C=c)
        elif b == "aten.mul":
            spec = bind(bf16_mul_scalar.Bf16MulScalar, N=N, C=c)
        else:
            spec = bind(bf16_tanh.Bf16Tanh, N=N)
        return _compute(node, ctx, spec, M, (PerRow(x),), (PerRow(out),), self.name)


def gemma2_patterns() -> list:
    return [EmbedScale(), GeluTanhAndMul(), MeanDimTriton(), SquarePow(), RsqrtF32Pattern(), ScalarAddF32(), AddWidened(), MulF32(),
            NarrowToBf16(), Bf16ScalarChain()]


def with_gemma2_patterns(patterns: list) -> list:
    """The profile's pattern list with the Gemma-2 patterns inserted ahead of the base `BiasAdd` / `TransparentAten` tail
    (after every base compute pattern; `NarrowToBf16` must precede `TransparentAten`, which reports float dtype changes)."""
    out: list = []
    inserted = False
    for p in patterns:
        if not inserted and type(p) is BiasAdd:
            out.extend(gemma2_patterns())
            inserted = True
        out.append(p)
    if not inserted:
        out.extend(gemma2_patterns())
    return out
