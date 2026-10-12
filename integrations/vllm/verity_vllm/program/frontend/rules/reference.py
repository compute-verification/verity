"""Reference-mode op rules (GUIDANCE §43.2 (B)): unannotated f32 / bf16 modules -> NUMERIC's reference kinds
(`verity_vllm/program/registry/ref_prims.py`, vocabulary `verity-ref-vocab/1`). Every rule checks dtypes and shapes
structurally; anything else is `Unsupported`. The reference kinds *define* the arithmetic (sequential f32 chains,
one DT rounding at stated points); torch CPU agreement is measured, not assumed (`docs/data/ref-prims/`).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import StaticIndex, refuse
from verity_vllm.program.frontend.torch_frontend import _permute, _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, norm_kind, rows_of, shape

aten = torch.ops.aten

_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}


def _dt_of(ctx, a) -> str | None:
    return _DT.get(ctx.dtype_of(a))


class LinearRule(OpRule):
    """`aten.linear(x, W, b?)` -> one `linear` kind call per row of `x` (+ one bias-add call per row).

    Structural preconditions: x and W are value collections of the mode's dtype; W is 2-D `[N, K]` with K = x's
    last dim; b is absent or `[N]`. W being a parameter is recorded (role `linear.weight`), not required."""

    def __init__(self, mode: str, dtype: torch.dtype, gemm_role: str, bias_role: str, version: int = 1) -> None:
        self.modes = frozenset({mode})
        self.dtype, self.gemm_role, self.bias_role = dtype, gemm_role, bias_role
        self.info = RuleInfo(f"Linear[{mode}]", version, "op-binding", applies_to=f"aten.linear with {str(dtype).removeprefix('torch.')} operands",
                             preconditions=("x, W (and b) are value collections, not static structure or allocations",
                                            f"dtypes are {dtype}", "W is [N, K] with K = x.shape[-1]; b is [N] or absent",
                                            "one kind call per row: rows are independent (batch-invariant linear)"),
                             evidence_domain=f"structural; numerics are the {gemm_role}/{bias_role} kinds' (see vocabulary)")

    def targets(self):
        return {aten.linear.default}

    def applies(self, node, ctx):
        x, w = node.args[0], node.args[1]
        b = node.args[2] if len(node.args) > 2 else None
        for nm, a in (("x", x), ("W", w)) + ((("b", b),) if b is not None else ()):
            if not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
            if not dtype_is(ctx, a, self.dtype):
                return f"numerics_unregistered: {nm} has dtype {ctx.dtype_of(a)}, this rule binds {self.dtype} linears only"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or len(ws) != 2 or not xs or xs[-1] != ws[1]:
            return f"shape: x {xs} vs W {ws}"
        if b is not None and shape(ctx, b) != (ws[0],):
            return f"shape: b {shape(ctx, b)} vs N={ws[0]}"
        return True

    def emit(self, node, ctx):
        x, w = ctx.coll(node.args[0], "x"), ctx.coll(node.args[1], "W")
        b = ctx.coll(node.args[2], "b") if len(node.args) > 2 and node.args[2] is not None else None
        N, K = shape_of(w.type)[0]
        rows, lead = rows_of(x)
        gemm = ctx.kind(self.gemm_role, node, K=K, N=N)
        bias = ctx.kind(self.bias_role, node, N=N) if b is not None else None
        outs = []
        for i, r in enumerate(rows):
            y = ctx.call(gemm, r, w, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
            if bias is not None:
                y = ctx.call(bias, y, b, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]/bias")
            outs.append(y)
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(node.args[1], self, "linear.weight", K=K, N=N)
        if b is not None:
            ctx.bind_weight_role(node.args[2], self, "linear.bias", N=N)
        ctx.record(self, node, {"dtype": str(self.dtype), "K": K, "N": N, "rows": len(rows), "bias": b is not None,
                                "W_is_parameter": (ctx.weight_of(node.args[1]) or None) and ctx.weight_of(node.args[1]).name},
                   emitted=f"{gemm.id} x{len(rows)}" + (f" + {bias.id} x{len(rows)}" if bias else ""))


class LinearRefRule(OpRule):
    """Reference mode: `aten.linear(x, W[N,K], b?)` and GPT-2 `Conv1D`'s `aten.addmm(b, x, W[K,N])` -> one
    `LinearRef_v1{K,N,DT,BIAS}` call per row (bias fused: one f32 add before the DT rounding, as CPU oneDNN/MKL do).
    DT is the operands' common dtype (f32 or bf16); `addmm` binds only with beta = alpha = 1 (else it is not a linear)."""

    modes = frozenset({"reference"})
    info = RuleInfo("Linear[reference]", 2, "op-binding", applies_to="aten.linear (W [N,K]) / aten.addmm (Conv1D: W [K,N], alpha=beta=1) on f32 or bf16",
                    preconditions=("x, W (and b) are value collections of one dtype in {f32, bf16}",
                                   "W is 2-D and its contraction dim equals x.shape[-1]; b is [N] or absent",
                                   "addmm: alpha == beta == 1 and x is 2-D",
                                   "one kind call per row: rows are independent"),
                    evidence_domain="numerics are LinearRef_v1's (sequential f32 fma chain from +0, fused bias add, one DT rounding) — reference-defined")

    def targets(self):
        return {aten.linear.default, aten.addmm.default}

    def _operands(self, node):
        if node.target is aten.linear.default:
            x, w = node.args[0], node.args[1]
            b = node.args[2] if len(node.args) > 2 else None
            return x, w, b, False
        b, x, w = node.args[0], node.args[1], node.args[2]
        return x, w, b, True

    def applies(self, node, ctx):
        x, w, b, transposed = self._operands(node)
        dts = set()
        for nm, a in (("x", x), ("W", w)) + ((("b", b),) if b is not None else ()):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
            dt = _dt_of(ctx, a)
            if dt is None:
                return f"numerics_unregistered: {nm} has dtype {ctx.dtype_of(a)} (reference linears are f32/bf16)"
            dts.add(dt)
        if len(dts) != 1:
            return f"numerics_unregistered: mixed operand dtypes {sorted(dts)}"
        xs, ws = shape(ctx, x), shape(ctx, w)
        if xs is None or ws is None or len(ws) != 2 or not xs:
            return f"shape: x {xs} vs W {ws}"
        K, N = (ws[0], ws[1]) if transposed else (ws[1], ws[0])
        if xs[-1] != K:
            return f"shape: x {xs} contracts {xs[-1]} but W is {ws}"
        if transposed:
            if len(xs) != 2:
                return "addmm needs a 2-D x"
            if node.kwargs.get("alpha", 1) != 1 or node.kwargs.get("beta", 1) != 1:
                return "addmm with alpha/beta != 1 is not a linear"
        if b is not None and shape(ctx, b) != (N,):
            return f"shape: b {shape(ctx, b)} vs N={N} (broadcast bias is not a bound rule)"
        return True

    def emit(self, node, ctx):
        xn, wn, bn, transposed = self._operands(node)
        x, w = ctx.coll(xn, "x"), ctx.coll(wn, "W")
        b = ctx.coll(bn, "b") if bn is not None else None
        DT = _dt_of(ctx, xn)
        if transposed:
            w = _permute(w, (1, 0), "addmm weight")  # [K,N] -> [N,K]: a view, no gates
        N, K = shape_of(w.type)[0]
        rows, lead = rows_of(x)
        fn = ctx.kind("linear", node, K=K, N=N, DT=DT, BIAS=b is not None)
        outs = [ctx.call(fn, r, w, *([b] if b is not None else []), node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]")
                for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(wn, self, "linear.weight", K=K, N=N)
        if bn is not None:
            ctx.bind_weight_role(bn, self, "linear.bias", N=N)
        wl = ctx.weight_of(wn)
        ctx.record(self, node, {"dtype": DT, "K": K, "N": N, "rows": len(rows), "bias": b is not None, "W_transposed_view": transposed,
                                "W_is_parameter": wl.name if wl else None}, emitted=f"{fn.id} x{len(rows)}")


class ElementwiseRefRule(OpRule):
    """Element-wise reference kinds over whole tensors in DT ∈ {f32, bf16}: relu, silu, gelu (erf / tanh), and
    add / mul of two same-shape tensors.

    v3 (G-RH-17, Gemma `embeds * embed_scale`): ONE binary operand may be a **0-d parameter/buffer of the module** (a declared
    weight leaf: `GemmaTextScaledWordEmbedding.embed_scale`, `sqrt(H)` registered as a buffer). It is already a prescribed Input
    (a field of the weights record with its name, shape `[]` and dtype); the rule feeds that one Input leaf to the kind N times
    (zero-stride references — no gates, no copy, no numerics beyond the kind's) and records the operand's provenance in the
    application: field name, kind (parameter | buffer), dtype, and the buffer's VALUE when the source module is materialised.
    Not a silent broadcast: any other shape mismatch (a 0-d computed value, a `[1]` or `[N,1]` operand, a Python scalar) still
    refuses with its own string."""

    _UNARY = {aten.relu.default: "relu", aten.silu.default: "silu", aten.gelu.default: "gelu"}
    _BINARY = {aten.add.Tensor: "add", aten.mul.Tensor: "mul"}

    modes = frozenset({"reference"})
    info = RuleInfo("Elementwise[reference]", 3, "op-binding", applies_to="aten.relu / silu / gelu(approximate none|tanh) / add.Tensor / mul.Tensor on f32 or bf16 values",
                    preconditions=("operands are value collections of one dtype in {f32, bf16}",
                                   "binary operands have identical shapes, OR exactly one of them is a 0-d parameter/buffer of the module (G-RH-17: "
                                   "its one Input leaf is read N times; name / kind / dtype / value recorded); no other broadcast, no Python scalar",
                                   "add.Tensor alpha == 1", "gelu approximate in {'none', 'tanh'}"),
                    evidence_domain="numerics are the ReluRef/SiluRef/GeluRef/AddRef/MulRef kinds' — reference-defined")

    def targets(self):
        return set(self._UNARY) | set(self._BINARY)

    def _scalar_weight_operand(self, node, ctx, ops) -> int | None:
        """Index of the one operand that is a 0-d parameter/buffer (the other has a non-empty shape), else None."""
        shapes = [shape(ctx, a) for a in ops]
        zero = [i for i, s in enumerate(shapes) if s == ()]
        if len(zero) != 1 or shapes[1 - zero[0]] == ():
            return None
        return zero[0] if ctx.weight_of(ops[zero[0]]) is not None else None

    def applies(self, node, ctx):
        arity = 1 if node.target in self._UNARY else 2
        ops = list(node.args[:arity])
        dts = set()
        for a in ops:
            if not isinstance(a, torch.fx.Node):
                return "numerics_unregistered: a Python scalar operand needs a constant kind"
            if not isinstance(ctx.val(a), Coll):
                return f"{a.name} is not a value collection"
            dt = _dt_of(ctx, a)
            if dt is None:
                return f"numerics_unregistered: {a.name} is {ctx.dtype_of(a)}, reference element-wise kinds are f32/bf16"
            dts.add(dt)
        if len(dts) != 1:
            return f"numerics_unregistered: mixed dtypes {sorted(dts)}"
        if arity == 2 and shape(ctx, ops[0]) != shape(ctx, ops[1]) and self._scalar_weight_operand(node, ctx, ops) is None:
            zero = [a for a in ops if shape(ctx, a) == ()]
            if zero and ctx.weight_of(zero[0]) is None:
                return (f"operands differ in shape {shape(ctx, ops[0])} vs {shape(ctx, ops[1])}: {zero[0].name} is a 0-d COMPUTED value, "
                        f"not a parameter/buffer of the module (only a declared 0-d weight leaf is read broadcast, G-RH-17)")
            return f"operands differ in shape {shape(ctx, ops[0])} vs {shape(ctx, ops[1])} (broadcast is not a bound rule)"
        if node.target is aten.add.Tensor and node.kwargs.get("alpha", 1) != 1:
            return "alpha != 1"
        if node.target is aten.gelu.default and node.kwargs.get("approximate", "none") not in ("none", "tanh"):
            return f"numerics_unregistered: gelu approximate={node.kwargs.get('approximate')!r}"
        return True

    def _scalar_weight_value(self, ctx, wl):
        """The 0-d weight's value from the source module when it is materialised (not meta); None otherwise."""
        try:
            tensors = dict(ctx.module.named_buffers())
            tensors.update(dict(ctx.module.named_parameters()))
            for nm in (wl.name, *wl.aliases):
                t = tensors.get(nm)
                if t is not None and t.device.type != "meta":
                    return float(t.detach().to(torch.float32).item())
        except Exception:  # noqa: BLE001 — provenance only; the program does not depend on it
            return None
        return None

    def emit(self, node, ctx):
        arity = 1 if node.target in self._UNARY else 2
        role = self._UNARY.get(node.target) or self._BINARY[node.target]
        ops = list(node.args[:arity])
        colls = [ctx.coll(a) for a in ops]
        DT = _dt_of(ctx, node.args[0])
        sc = self._scalar_weight_operand(node, ctx, ops) if arity == 2 else None
        shp, w = shape_of(colls[1 - sc].type if sc is not None else colls[0].type)
        n = math.prod(shp) if shp else 1
        flat, scalar_rec = [], None
        for i, c in enumerate(colls):
            if i == sc:
                from verity.ml.circuits.refs import broadcast
                space, idx, leaf = c.leaf  # the one Input leaf of the 0-d weight field
                flat.append(Coll(Array(n, Value(w)), broadcast(space, idx, leaf, n)))
                wl = ctx.weight_of(ops[i])
                scalar_rec = {"arg": i, "field": wl.name, "aliases": list(wl.aliases), "weight_kind": wl.kind, "dtype": wl.dtype, "shape": [],
                              "value": self._scalar_weight_value(ctx, wl),
                              "representation": f"the field's single Input leaf referenced {n} times (zero-stride); no gates, no copy"}
                ctx.bind_weight_role(ops[i], self, f"{role}.scalar", N=n)
            else:
                flat.append(c.reshape(Array(n, Value(w))))
        statics = {"N": n, "DT": DT}
        if role == "gelu":
            statics["APPROX"] = node.kwargs.get("approximate", "none")
        fn = ctx.kind(role, node, **statics)
        out = ctx.call(fn, *flat, node=node, rule=self)
        ctx.define(node, out.reshape(_tensor_type(shp, w)))
        rec = {"op": role, **statics}
        if scalar_rec is not None:
            rec["scalar_operand"] = scalar_rec
        ctx.record(self, node, rec, emitted=fn.id)


class LayerNormRefRule(OpRule):
    """`aten.layer_norm(x, [N], g, b, eps)` / `native_layer_norm` over the last dim with affine weight and bias ->
    one `LayerNormRef_v1{N,EPS,DT}` call per row (two-pass mean/biased-variance, reference roundings)."""

    modes = frozenset({"reference"})
    info = RuleInfo("LayerNorm[reference]", 1, "op-binding", applies_to="aten.layer_norm / native_layer_norm over the last dim with g and b",
                    preconditions=("x, g, b are value collections of one dtype in {f32, bf16}", "normalized_shape == [x.shape[-1]]",
                                   "g and b are present ([N]); elementwise_affine=False needs constant rows (not bound)",
                                   "one kind call per row"),
                    evidence_domain="numerics are LayerNormRef_v1's — reference-defined")

    def targets(self):
        return {aten.layer_norm.default, aten.native_layer_norm.default}

    def _operands(self, node):
        x, ns = node.args[0], node.args[1]
        g = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        b = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
        eps = node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)
        return x, tuple(ns), g, b, float(eps)

    def applies(self, node, ctx):
        x, ns, g, b, eps = self._operands(node)
        if g is None or b is None:
            return "numerics_unregistered: layer_norm without affine g/b (constant rows are not a bound rule yet)"
        dts = set()
        for nm, a in (("x", x), ("g", g), ("b", b)):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
            dt = _dt_of(ctx, a)
            if dt is None:
                return f"numerics_unregistered: {nm} is {ctx.dtype_of(a)}"
            dts.add(dt)
        if len(dts) != 1:
            return f"numerics_unregistered: mixed dtypes {sorted(dts)}"
        xs = shape(ctx, x)
        if not xs or ns != (xs[-1],):
            return f"normalized_shape {ns} is not the last dim of x {xs}"
        if shape(ctx, g) != (xs[-1],) or shape(ctx, b) != (xs[-1],):
            return f"g {shape(ctx, g)} / b {shape(ctx, b)} are not [N={xs[-1]}]"
        return True

    def emit(self, node, ctx):
        xn, ns, gn, bn, eps = self._operands(node)
        x, g, b = ctx.coll(xn, "x"), ctx.coll(gn, "g"), ctx.coll(bn, "b")
        DT = _dt_of(ctx, xn)
        rows, lead = rows_of(x)
        N = ns[0]
        fn, value = norm_kind(ctx, "layer_norm", node, N, eps, DT=DT)
        outs = [ctx.call(fn, r, g, b, *value, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        y = from_rows(outs, lead)
        if node.target is aten.native_layer_norm.default:
            # (y, mean, rstd): mean/rstd are internal to the kind; only y is a value of the program
            ctx.define(node, [y, None, None])
        else:
            ctx.define(node, y)
        ctx.bind_weight_role(gn, self, "layer_norm.weight", N=N)
        ctx.bind_weight_role(bn, self, "layer_norm.bias", N=N)
        ctx.record(self, node, {"N": N, "EPS": eps, "DT": DT, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class RMSNormRefRule(OpRule):
    """HF-style eager RMSNorm is a *decomposed* pattern (pow/mean/rsqrt/mul) — not bound here (see `_DECOMPOSED_HINTS`).
    This rule binds only an explicit `aten.rms_norm(x, [N], w, eps)` op (torch >= 2.4) -> `RMSNormRef_v1{N,EPS,DT}` per row."""

    modes = frozenset({"reference"})
    info = RuleInfo("RMSNorm[reference]", 1, "op-binding", applies_to="aten.rms_norm over the last dim with weight",
                    preconditions=("x, w are value collections of one dtype in {f32, bf16}", "normalized_shape == [x.shape[-1]]", "eps given"),
                    evidence_domain="numerics are RMSNormRef_v1's — reference-defined")

    def targets(self):
        t = getattr(aten, "rms_norm", None)
        return {t.default} if t is not None else set()

    def applies(self, node, ctx):
        x, ns = node.args[0], tuple(node.args[1])
        w = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        eps = node.args[3] if len(node.args) > 3 else node.kwargs.get("eps")
        if w is None:
            return "numerics_unregistered: rms_norm without weight"
        if eps is None:
            return "numerics_unregistered: rms_norm with default eps (torch picks finfo.eps; make it explicit)"
        for nm, a in (("x", x), ("w", w)):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
        if _dt_of(ctx, x) is None or _dt_of(ctx, x) != _dt_of(ctx, w):
            return f"numerics_unregistered: dtypes {ctx.dtype_of(x)} / {ctx.dtype_of(w)}"
        xs = shape(ctx, x)
        if not xs or ns != (xs[-1],) or shape(ctx, w) != (xs[-1],):
            return f"normalized_shape {ns} / w {shape(ctx, w)} vs x {xs}"
        return True

    def emit(self, node, ctx):
        xn, ns = node.args[0], tuple(node.args[1])
        wn = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        eps = float(node.args[3] if len(node.args) > 3 else node.kwargs.get("eps"))
        x, w = ctx.coll(xn), ctx.coll(wn)
        DT = _dt_of(ctx, xn)
        rows, lead = rows_of(x)
        fn, value = norm_kind(ctx, "rmsnorm", node, ns[0], eps, DT=DT)
        outs = [ctx.call(fn, r, w, *value, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(wn, self, "rmsnorm.weight", N=ns[0])
        ctx.record(self, node, {"N": ns[0], "EPS": eps, "DT": DT, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class CastRefRule(OpRule):
    """`x.to(dtype)` between bf16 and f32 in reference mode: exact widening / RNE narrowing kinds. v2 (G-RH-2): also the
    `to.device(x, device, dtype)` / `to.dtype_layout(x, dtype=…, device=…)` spellings — the device is not semantics."""
    modes = frozenset({"reference"})
    info = RuleInfo("CastF32Bf16", 2, "op-binding", applies_to="aten.to.dtype / to.device / to.dtype_layout / _to_copy between bfloat16 and float32",
                    preconditions=("operand is a value collection", "dtypes are (bf16 -> f32) or (f32 -> bf16)", "device / layout are ignored"),
                    evidence_domain="numerics: Bf16ToF32 (exact) / F32ToBf16Rn (RNE), both exact-model-tested primitives")

    def targets(self):
        return {aten.to.dtype, aten._to_copy.default, aten.to.device, aten.to.dtype_layout}

    def applies(self, node, ctx):
        if not isinstance(ctx.val(node.args[0]), Coll):
            return "operand is not a value collection"
        src, dst = ctx.dtype_of(node.args[0]), ctx.dtype_of(node)
        if (src, dst) in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
            return True
        return f"numerics_unregistered: cast {src} -> {dst}"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        src = ctx.dtype_of(node.args[0])
        shp, w = shape_of(x.type)
        n = math.prod(shp) if shp else 1
        role = "bf16_to_f32" if src == torch.bfloat16 else "f32_to_bf16"
        fn = ctx.kind(role, node, N=n)
        out = ctx.call(fn, x.reshape(Array(n, Value(w))), node=node, rule=self)
        ctx.define(node, out.reshape(_tensor_type(shp, 32 if role == "bf16_to_f32" else 16)))
        ctx.record(self, node, {"cast": role, "N": n}, emitted=fn.id)


class MatmulRefRule(OpRule):
    """`aten.matmul / mm / bmm (a[..., M, K] @ b[..., K, N])` in reference mode -> one `LinearRef_v1{K,N,DT,BIAS=false}` call per
    row of `a` with `b`'s [K,N] slice read transposed (a view — reference-sequence permutation, no gates). Covers GPT-2 `Conv1D`
    (`x @ W[in,out]`, W a parameter) and activation×activation products (`q @ k^T`, `p @ v`) alike: the kind does not care which
    operand is a parameter (XFER-G1). Batch dims must agree (no broadcasting of batch dims)."""

    modes = frozenset({"reference"})
    info = RuleInfo("Matmul[reference]", 1, "op-binding", applies_to="aten.matmul / mm / bmm with a[..., M, K] @ b[..., K, N] on f32 or bf16 values",
                    preconditions=("both operands are value collections of one dtype in {f32, bf16}", "a is >= 2-D, b is >= 2-D, a.shape[-1] == b.shape[-2]",
                                   "leading (batch) dims are equal (or b is 2-D and shared by every row of a)",
                                   "one LinearRef{K,N,BIAS=false} call per (batch, row): rows are independent"),
                    evidence_domain="numerics are LinearRef_v1's (sequential f32 fma chain from +0, one DT rounding) — reference-defined")

    def targets(self):
        return {aten.matmul.default, aten.mm.default, aten.bmm.default}

    def applies(self, node, ctx):
        a, b = node.args[0], node.args[1]
        for nm, x in (("a", a), ("b", b)):
            if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
                return f"{nm} is not a value collection"
        da, db = _dt_of(ctx, a), _dt_of(ctx, b)
        if da is None or da != db:
            return f"numerics_unregistered: dtypes {ctx.dtype_of(a)} @ {ctx.dtype_of(b)}"
        sa, sb = shape(ctx, a), shape(ctx, b)
        if sa is None or sb is None or len(sa) < 1 or len(sb) != 2 and len(sb) < 2:
            return f"shape: {sa} @ {sb}"
        if len(sb) < 2:
            return f"shape: {sa} @ {sb} (a 1-D right operand — a dot product — is not bound; only [.., K, N] right operands)"
        if sa[-1] != sb[-2]:
            return f"shape: {sa} @ {sb} (contraction dims differ)"
        if len(sa) >= 2 and len(sb) > 2 and tuple(sa[:-2]) != tuple(sb[:-2]):
            return f"batch dims differ {sa[:-2]} vs {sb[:-2]} (batch broadcasting is not a bound rule)"
        if len(sa) == 1 and len(sb) > 2:
            return f"shape: 1-D left operand with a batched right operand {sb}"
        return True

    def emit(self, node, ctx):
        an, bn = node.args[0], node.args[1]
        a, b = ctx.coll(an, "a"), ctx.coll(bn, "b")
        DT = _dt_of(ctx, an)
        sa, w = shape_of(a.type)
        sb = shape_of(b.type)[0]
        vec = len(sa) == 1  # x[K] @ W[K,N]: one row (XFER-G10)
        if vec:
            a = a.reshape(Array(1, Array(sa[0], Value(w))))
            sa = (1, sa[0])
        M, K, N = sa[-2], sa[-1], sb[-1]
        batch = tuple(sa[:-2])
        nb = math.prod(batch) if batch else 1
        a_flat = a.reshape(Array(nb, Array(M, Array(K, Value(w)))))
        if len(sb) > 2:
            b_flat = b.reshape(Array(nb, Array(K, Array(N, Value(w)))))
            bT = [_permute(b_flat[i], (1, 0), "matmul rhs") for i in range(nb)]
        else:
            bT = [_permute(b, (1, 0), "matmul rhs")] * nb
        fn = ctx.kind("linear", node, K=K, N=N, DT=DT, BIAS=False)
        outs = []
        for i in range(nb):
            for m in range(M):
                outs.append(ctx.call(fn, a_flat[i][m], bT[i], node=node, rule=self, name=f"{ctx.hint_name(node)}[{i},{m}]"))
        from verity.ml.circuits.refs import array_of
        ctx.define(node, array_of(outs).reshape(_tensor_type(((N,) if vec else batch + (M, N)), w)))
        wl = ctx.weight_of(bn)
        if wl is not None:
            ctx.bind_weight_role(bn, self, "matmul.weight", K=K, N=N)
        ctx.record(self, node, {"dtype": DT, "batch": list(batch), "M": M, "K": K, "N": N, "rhs_is_parameter": wl.name if wl else None,
                                "rhs_transposed_view": True}, emitted=f"{fn.id} x{nb * M}")


class BroadcastBiasAddRefRule(OpRule):
    """`aten.add.Tensor(x[..., N], b[N])` (a bias broadcast over the leading dims — GPT-2 `Conv1D`'s `+ b` after `matmul`) -> one
    `AddRef_v1{N,DT}` call per row. Same-shape adds are `Elementwise[reference]`."""

    modes = frozenset({"reference"})
    info = RuleInfo("BroadcastBiasAdd[reference]", 1, "op-binding", applies_to="aten.add.Tensor(x[..., N], b[N]) with b 1-D, alpha == 1, one dtype",
                    preconditions=("both operands are value collections of one dtype in {f32, bf16}", "b is 1-D of length x.shape[-1]; x is >= 2-D",
                                   "alpha == 1", "one AddRef call per row of x"),
                    evidence_domain="numerics are AddRef_v1's — reference-defined")

    def targets(self):
        return {aten.add.Tensor}

    def applies(self, node, ctx):
        x, b = node.args[0], node.args[1]
        for nm, v in (("x", x), ("b", b)):
            if not isinstance(v, torch.fx.Node) or not isinstance(ctx.val(v), Coll):
                return f"{nm} is not a value collection"
        if _dt_of(ctx, x) is None or _dt_of(ctx, x) != _dt_of(ctx, b):
            return f"numerics_unregistered: dtypes {ctx.dtype_of(x)} + {ctx.dtype_of(b)}"
        sx, sb = shape(ctx, x), shape(ctx, b)
        if sx is None or sb is None or len(sb) != 1 or len(sx) < 2 or sx[-1] != sb[0]:
            return f"not a [.., N] + [N] bias broadcast ({sx} + {sb})"
        if node.kwargs.get("alpha", 1) != 1:
            return "alpha != 1"
        return True

    def emit(self, node, ctx):
        xn, bn = node.args[0], node.args[1]
        x, b = ctx.coll(xn), ctx.coll(bn)
        DT = _dt_of(ctx, xn)
        rows, lead = rows_of(x)
        N = shape_of(b.type)[0][0]
        fn = ctx.kind("add", node, N=N, DT=DT)
        outs = [ctx.call(fn, r, b, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        if ctx.weight_of(bn) is not None:
            ctx.bind_weight_role(bn, self, "bias", N=N)
        ctx.record(self, node, {"dtype": DT, "N": N, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class EmbeddingRefRule(OpRule):
    """`aten.embedding(table[V,H], ids)` with ids a *value* (a prescribed input or a computed token) -> `EmbeddingRef_v1{V,H,DT}` per
    token; int64 ids are narrowed first with `I64ToI32_v1` (range-checked, XFER-G3 / R4). Static ids (`wpe[arange(T)]`) stay
    `StaticGather_v1`.

    v2 (G-RH-14): `padding_idx` may be any Python int or None. ATen's forward `embedding(weight, indices, padding_idx, …)` is
    `weight.index_select(0, indices)` whatever `padding_idx` is — the argument is read only by `embedding_backward` (it zeroes the
    gradient of that row) and by `nn.Embedding.__init__` (it zero-initialises the row, a fact about the weights, not the op). So the
    forward IS the plain gather and the kind is the same `EmbeddingRef_v1{V,H,DT}`; the value is recorded. `scale_grad_by_freq` /
    `sparse` stay refused as before (also gradient-only, but not observed on any class; not widened without a case)."""

    modes = frozenset({"reference"})
    info = RuleInfo("Embedding[reference]", 2, "op-binding", applies_to="aten.embedding(table[V,H] f32|bf16, ids int32|int64 value, padding_idx=any int|None)",
                    preconditions=("table is a value collection [V, H] of dtype in {f32, bf16}", "ids is a value collection (not static structure) of int32 or int64 leaves",
                                   "padding_idx is a Python int or None (forward-irrelevant: ATen's embedding forward is the row gather for every "
                                   "padding_idx; only embedding_backward reads it) — recorded; scale_grad_by_freq / sparse are defaults",
                                   "one EmbeddingRef call per token (+ one I64ToI32 per int64 token)"),
                    evidence_domain="numerics are EmbeddingRef_v1's (row gather) and I64ToI32_v1's — reference-defined")

    def targets(self):
        return {aten.embedding.default}

    def applies(self, node, ctx):
        table, ids = node.args[0], node.args[1]
        for nm, v in (("table", table), ("ids", ids)):
            if not isinstance(v, torch.fx.Node) or not isinstance(ctx.val(v), Coll):
                return f"{nm} is not a value collection" + (" (static ids are StaticGather's)" if nm == "ids" else "")
        if _dt_of(ctx, table) is None:
            return f"numerics_unregistered: table dtype {ctx.dtype_of(table)}"
        if ctx.dtype_of(ids) not in (torch.int32, torch.int64):
            return f"ids dtype {ctx.dtype_of(ids)} is not int32/int64"
        st = shape(ctx, table)
        if st is None or len(st) != 2:
            return f"table is not 2-D: {st}"
        pad = self._padding_idx(node)
        if pad is not None and (isinstance(pad, bool) or not isinstance(pad, int)):
            return f"padding_idx={pad!r} is not a Python int or None"
        if any(node.args[3:]) or any(v for k, v in node.kwargs.items() if k != "padding_idx"):
            return "scale_grad_by_freq / sparse are set"
        return True

    @staticmethod
    def _padding_idx(node):
        pad = node.args[2] if len(node.args) > 2 else node.kwargs.get("padding_idx", -1)
        return None if pad in (-1, None) else pad

    def emit(self, node, ctx):
        tn, idn = node.args[0], node.args[1]
        table, ids = ctx.coll(tn), ctx.coll(idn)
        DT = _dt_of(ctx, tn)
        V, H = shape_of(table.type)[0]
        shp, w = shape_of(ids.type)
        n = math.prod(shp) if shp else 1
        flat = ids.reshape(Array(n, Value(w)))
        fn = ctx.kind("embedding", node, V=V, H=H, DT=DT)
        narrow = ctx.kind("i64_to_i32", node) if w == 64 else None
        outs = []
        for i in range(n):
            tok = flat[i]
            if narrow is not None:
                tok = ctx.call(narrow, tok, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]/i32")
            outs.append(ctx.call(fn, tok, table, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]"))
        from verity.ml.circuits.refs import array_of
        ctx.define(node, array_of(outs).reshape(_tensor_type(tuple(shp) + (H,), 32 if DT == "f32" else 16)))
        ctx.bind_weight_role(tn, self, "embedding.table", V=V, H=H)
        pad = self._padding_idx(node)
        ctx.record(self, node, {"dtype": DT, "V": V, "H": H, "tokens": n, "ids_width": w, "narrowed": narrow is not None,
                                "padding_idx": pad, "padding_idx_forward_effect": "none (row gather; padding_idx is read by embedding_backward only)"
                                if pad is not None else "n/a"},
                   emitted=f"{fn.id} x{n}" + (f" + I64ToI32_v1 x{n}" if narrow else ""))


class SdpaRefRule(OpRule):
    """`F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=0, is_causal, scale, enable_gqa)` with `q [.., NH, T, D]`,
    `k, v [.., KVH, T, D]` of one dtype -> `EagerAttentionRef_v1{T=i+1,NH,KVH,D,DT,SCALE_OP="mul"}` per query row i: causal
    masking = the row reads keys 0..i only (exact, NUMERIC's note), the scale `D^-0.5` applied to the scores. The *reference*
    arithmetic is the kind's (f32 sequential dot, one DT rounding per stated point); torch's math SDPA scales q and k by
    sqrt(scale) before the matmul — agreement with torch CPU is measured, not assumed, as for every reference kind.

    v2 binds the three spellings transformers 5.x `sdpa_attention_forward` uses (G-RH-8/-9/-10), each only when it is the SAME
    computation as v1's `is_causal=True, scale=None, KVH=NH`:
      * the causal structure as an explicit **static bool mask** (`is_causal=False`, `attn_mask` a `StaticIndex` of dtype bool whose
        every `[T, T]` slice equals `tril(ones)` — allowed iff key <= query; torch: `is_causal=True` IS `attn_mask=tril(ones(L, S))`
        for L == S, so the two calls are one function); any other mask (a value, a lifted constant, a float mask, a non-causal
        pattern, T != S) is refused;
      * an explicit `scale` equal, as f32, to `D^-0.5` — the constant the kind applies (`RN32(D^-0.5)`); any other value is refused;
      * `enable_gqa=True` with `NH % KVH == 0`: torch defines it as `k, v = k.repeat_interleave(NH // KVH, dim=-3)` — query head h
        reads KV head `h // (NH // KVH)` — which is exactly `EagerAttentionRef{NH, KVH}`'s head mapping (its docstring: HF `repeat_kv`),
        so the kind is called with the un-repeated `[T][KVH][D]` caches; the per-head arithmetic (`EagerAttentionHeadRef`) is
        unchanged. Head counts that do not divide are refused.
    Dropout, a non-causal call, a mask together with `is_causal=True` are not this rule."""
    modes = frozenset({"reference"})
    info = RuleInfo("Sdpa[reference]", 2, "op-binding",
                    applies_to="aten.scaled_dot_product_attention(q, k, v, attn_mask=None|static causal bool, 0.0, is_causal=True|False, "
                               "scale=None|D^-0.5, enable_gqa=False|True)",
                    preconditions=("q [.., NH, T, D], k and v [.., KVH, T, D] are value collections of one dtype (bf16 / f32)",
                                   "dropout_p == 0",
                                   "causal structure: is_causal=True with attn_mask None, OR is_causal=False with attn_mask a StaticIndex bool "
                                   "tensor whose every [T, T] slice (leading dims 1 or batch) equals tril(ones(T, T)) — allowed iff key <= query",
                                   "scale is None or np.float32(scale) == np.float32(D ** -0.5)",
                                   "enable_gqa False with KVH == NH, or enable_gqa True with NH % KVH == 0 (torch: k, v repeat_interleave'd "
                                   "NH//KVH over heads == the kind's head h -> KV head h // (NH//KVH))"),
                    evidence_domain="structural; numerics: EagerAttentionRef_v1 (reference-defined)")

    def targets(self):
        return {aten.scaled_dot_product_attention.default}

    def _args(self, node):
        mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
        drop = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        scale = node.args[6] if len(node.args) > 6 else node.kwargs.get("scale")
        gqa = node.args[7] if len(node.args) > 7 else node.kwargs.get("enable_gqa", False)
        return mask, drop, causal, scale, gqa

    def _causal_mask(self, node, ctx, mask, T: int, S: int, lead: tuple) -> bool | str:
        """The explicit mask denotes the causal structure iff it is static, bool, [.., T, T] and every slice is tril(ones)."""
        if not isinstance(mask, torch.fx.Node):
            return f"attn_mask is a Python object {type(mask).__name__}"
        mv = ctx.val(mask)
        if not isinstance(mv, StaticIndex):
            if isinstance(mv, Coll):
                return ("an explicit attn_mask that is a value (an input or computed), not static structure — only a statically known "
                        "bool causal mask is bound (the causal structure is the kind's)")
            return f"an explicit attn_mask that is {type(mv).__name__}, not static structure (only a static bool causal mask is bound)"
        mt = mv.tensor
        if mt.dtype is not torch.bool:
            return f"a static attn_mask of dtype {mt.dtype} (only a bool causal mask is bound; additive masks are not this rule)"
        if T != S:
            return f"q length {T} != kv length {S} with an explicit mask (is_causal's top-left alignment only coincides for T == S)"
        if mt.dim() < 2 or tuple(mt.shape[-2:]) != (T, T):
            return f"attn_mask shape {tuple(mt.shape)} is not [.., T={T}, T]"
        for d in mt.shape[:-2]:
            if int(d) != 1 and int(d) not in lead:
                return f"attn_mask leading dims {tuple(mt.shape[:-2])} do not broadcast over q's {lead}"
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        m2 = mt.reshape(-1, T, T)
        if not all(bool(torch.equal(m2[b], causal)) for b in range(m2.shape[0])):
            return "the static attn_mask is not the causal pattern (allowed iff key <= query); a different mask is a different function"
        return True

    def applies(self, node, ctx):
        q, k, v = node.args[:3]
        for nm, a in (("q", q), ("k", k), ("v", v)):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{nm} is not a value collection"
        dts = {ctx.dtype_of(a) for a in (q, k, v)}
        if len(dts) != 1 or next(iter(dts)) not in (torch.bfloat16, torch.float32):
            return f"numerics_unregistered: dtypes {dts} (one of bf16 / f32 for all three)"
        mask, drop, causal, scale, gqa = self._args(node)
        if drop:
            return f"dropout_p={drop} (a fixed computation has no dropout)"
        sq, sk, sv = shape(ctx, q), shape(ctx, k), shape(ctx, v)
        if sq is None or sk is None or sv is None or len(sq) < 3 or len(sk) != len(sq) or sk != sv:
            return f"shapes q{sq} k{sk} v{sv} (pattern: q [.., NH, T, D], k == v [.., KVH, T, D])"
        NH, T, D = sq[-3], sq[-2], sq[-1]
        KVH, S = sk[-3], sk[-2]
        if tuple(sq[:-3]) != tuple(sk[:-3]) or sk[-1] != D:
            return f"shapes q{sq} k{sk} (batch dims / head dim differ)"
        if gqa:
            if gqa is not True:
                return f"enable_gqa={gqa!r} is not a bool literal"
            if KVH == 0 or NH % KVH != 0:
                return f"enable_gqa with NH={NH} not a multiple of KVH={KVH} (torch would reject it; no head mapping)"
        elif KVH != NH:
            return f"KVH={KVH} != NH={NH} without enable_gqa (repeat the KV heads — HF repeat_kv — or pass enable_gqa=True)"
        if mask is not None:
            if causal is True:
                return "attn_mask together with is_causal=True (torch rejects the combination)"
            ok = self._causal_mask(node, ctx, mask, T, S, tuple(sq[:-3]))
            if ok is not True:
                return ok
        elif causal is not True:
            return "not is_causal=True (a non-causal attention is a different function)"
        elif T != S:
            return f"is_causal=True with q length {T} != kv length {S} (a decode step over a cache is not this rule)"
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                return f"scale={scale!r} is not a Python number"
            if np.float32(scale) != np.float32(float(D) ** -0.5):
                return f"scale={scale} is not D^-0.5 = {float(D) ** -0.5} (D={D}) as f32 — a different scale is a different function"
        return True

    def emit(self, node, ctx):
        q, k, v = node.args[:3]
        mask, _, causal, scale, gqa = self._args(node)
        qc, kc, vc = ctx.coll(q, "q"), ctx.coll(k, "k"), ctx.coll(v, "v")
        shp, w = shape_of(qc.type)
        kshp = shape_of(kc.type)[0]
        DT = "f32" if w == 32 else "bf16"
        lead, NH, T, D = tuple(shp[:-3]), shp[-3], shp[-2], shp[-1]
        KVH = kshp[-3]
        nb = math.prod(lead) if lead else 1
        qb = qc.reshape(Array(nb, Array(NH, Array(T, Array(D, Value(w))))))
        kb = kc.reshape(Array(nb, Array(KVH, Array(T, Array(D, Value(w))))))
        vb = vc.reshape(Array(nb, Array(KVH, Array(T, Array(D, Value(w))))))
        outs_b = []
        for b in range(nb):
            q_t = _permute(qb[b], (1, 0, 2), "q rows")  # [T][NH][D]
            k_t = _permute(kb[b], (1, 0, 2), "k cache")  # [T][KVH][D]
            v_t = _permute(vb[b], (1, 0, 2), "v cache")
            rows = []
            for i in range(T):
                fn = ctx.kind("eager_attention", node, T=i + 1, NH=NH, KVH=KVH, D=D, DT=DT, SCALE_OP="mul")
                rows.append(ctx.call(fn, q_t[i], k_t[0:i + 1], v_t[0:i + 1], node=node, rule=self, name=f"{ctx.hint_name(node)}[t={i}]"))
            outs_b.append(_permute(array_of(rows), (1, 0, 2), "attention out"))  # [NH][T][D]
        out = array_of(outs_b).reshape(_tensor_type(tuple(shp), w))
        ctx.define(node, out)
        ctx.record(self, node, {"dtype": DT, "NH": NH, "KVH": KVH, "T": T, "D": D, "batches": nb,
                                "causal": "is_causal=True" if mask is None else "static bool attn_mask verified == tril(ones) (is_causal=False)",
                                "scale": "D^-0.5 (default)" if scale is None else f"{scale} == D^-0.5 as f32",
                                "enable_gqa": bool(gqa),
                                "torch_arithmetic_note": "torch math SDPA scales q,k by sqrt(scale) before the dot; the kind scales the scores"},
                   emitted=f"EagerAttentionRef_v1{{T=1..{T},NH={NH},KVH={KVH},D={D},DT={DT},SCALE_OP=mul}} x{T * nb}")


class SoftmaxRefRule(OpRule):
    """`aten._softmax(x, dim=-1, half_to_float=False)` / `aten.softmax.int` over the last dim -> `SoftmaxRef_v1{N,DT}` per row."""

    modes = frozenset({"reference"})
    info = RuleInfo("Softmax[reference]", 1, "op-binding", applies_to="aten._softmax / softmax.int over the last dim, no dtype change",
                    preconditions=("x is a value collection of dtype in {f32, bf16}", "dim is the last axis", "half_to_float is False / dtype is None",
                                   "one SoftmaxRef call per row"),
                    evidence_domain="numerics are SoftmaxRef_v1's — reference-defined")

    def targets(self):
        return {aten._softmax.default, aten.softmax.int}

    def _dim(self, node):
        return node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return "x is not a value collection"
        if _dt_of(ctx, x) is None:
            return f"numerics_unregistered: dtype {ctx.dtype_of(x)}"
        sx = shape(ctx, x)
        d = self._dim(node)
        if not sx or d not in (-1, len(sx) - 1):
            return f"softmax over dim {d} of {sx} is not the last axis (not a bound rule)"
        if node.target is aten._softmax.default and len(node.args) > 2 and node.args[2]:
            return "numerics_unregistered: half_to_float"
        if node.target is aten.softmax.int and (len(node.args) > 2 and node.args[2] is not None or node.kwargs.get("dtype") is not None):
            return "numerics_unregistered: softmax with a dtype change"
        return True

    def emit(self, node, ctx):
        xn = node.args[0]
        x = ctx.coll(xn)
        DT = _dt_of(ctx, xn)
        rows, lead = rows_of(x)
        N = shape_of(rows[0].type)[0][0]
        fn = ctx.kind("softmax", node, N=N, DT=DT)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.record(self, node, {"dtype": DT, "N": N, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


class ArgmaxRefRule(OpRule):
    """`aten.argmax(logits[.., V], dim=-1|None, keepdim=False)` -> `ArgmaxRef_v1{V,DT}` per row (lowest index among maxima); the
    torch int64 result is carried as the kind's I32 token."""

    modes = frozenset({"reference"})
    info = RuleInfo("Argmax[reference]", 1, "op-binding", applies_to="aten.argmax over the last axis of f32|bf16 logits, keepdim False",
                    preconditions=("logits is a value collection of dtype in {f32, bf16}", "reduction over the last axis (or the whole 1-D tensor)",
                                   "keepdim is False", "the int64 result is represented as I32 (values < V)"),
                    evidence_domain="numerics are ArgmaxRef_v1's — reference-defined")

    def targets(self):
        return {aten.argmax.default}

    def applies(self, node, ctx):
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return "logits is not a value collection"
        if _dt_of(ctx, x) is None:
            return f"numerics_unregistered: dtype {ctx.dtype_of(x)}"
        sx = shape(ctx, x)
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        if keep:
            return "keepdim=True"
        if dim is None and len(sx or ()) != 1:
            return "argmax over a flattened >1-D tensor (not a bound rule)"
        if dim is not None and dim not in (-1, len(sx) - 1):
            return f"argmax over dim {dim} of {sx} is not the last axis"
        return True

    def emit(self, node, ctx):
        xn = node.args[0]
        x = ctx.coll(xn)
        DT = _dt_of(ctx, xn)
        rows, lead = rows_of(x)
        V = shape_of(rows[0].type)[0][0]
        fn = ctx.kind("argmax", node, V=V, DT=DT)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        from verity.ml.circuits.refs import array_of
        out = array_of(outs) if lead else outs[0]
        ctx.define(node, out.reshape(_tensor_type(tuple(lead), 32)) if lead else out)
        ctx.record(self, node, {"dtype": DT, "V": V, "rows": len(rows), "result": "I32 token (torch int64)"}, emitted=f"{fn.id} x{len(rows)}")


class IntCatNarrowRule(OpRule):
    """`aten.cat / stack` of integer token collections of mixed leaf widths (an int64 prompt Input concatenated with the I32
    tokens `ArgmaxRef` selects — torch calls both int64) -> every 64-bit leaf narrowed with `I64ToI32_v1`, then the plain
    concatenation of I32 leaves. Same-width cats are `CatStack_v1` (no gates)."""

    modes = frozenset({"reference"})
    info = RuleInfo("IntCatNarrow[reference]", 1, "op-binding", applies_to="aten.cat / aten.stack of int64|int32 token collections with mixed 64/32-bit leaves",
                    preconditions=("every part is a value collection of 32- or 64-bit integer leaves and at least one differs", "the torch result dtype is int64",
                                   "one I64ToI32 call per 64-bit leaf (range-checked by the kind)"),
                    evidence_domain="numerics are I64ToI32_v1's — reference-defined; the concatenation itself is structural")

    def targets(self):
        return {aten.cat.default, aten.stack.default}

    def applies(self, node, ctx):
        parts = node.args[0]
        if not isinstance(parts, (list, tuple)) or not parts:
            return "no parts"
        widths = set()
        for p in parts:
            if not isinstance(p, torch.fx.Node) or not isinstance(ctx.val(p), Coll):
                return "a part is not a value collection"
            if ctx.dtype_of(p) not in (torch.int32, torch.int64):
                return f"part dtype {ctx.dtype_of(p)} is not an integer token type"
            widths.add(shape_of(ctx.val(p).type)[1])
        if widths != {32, 64}:
            return f"leaf widths {sorted(widths)} are not mixed 32/64 (a same-width cat is a view)"
        if ctx.dtype_of(node) is not torch.int64:
            return f"result dtype {ctx.dtype_of(node)} is not int64"
        return True

    def emit(self, node, ctx):
        from verity.ml.circuits.refs import array_of, concat
        parts = [ctx.coll(p) for p in node.args[0]]
        narrow = ctx.kind("i64_to_i32", node)
        outs, n64 = [], 0
        for p in parts:
            shp, w = shape_of(p.type)
            if w == 64:
                flat = p.reshape(Array(math.prod(shp) if shp else 1, Value(64)))
                toks = [ctx.call(narrow, flat[i], node=node, rule=self, name=f"{ctx.hint_name(node)}[{n64 + i}]/i32") for i in range(flat.type.n)]
                n64 += len(toks)
                outs.append(array_of(toks).reshape(_tensor_type(tuple(shp), 32)))
            else:
                outs.append(p)
        if node.target is aten.stack.default:
            outs = [o.reshape(_tensor_type((1,) + tuple(shape_of(o.type)[0]), 32)) for o in outs]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        rank = len(shape_of(outs[0].type)[0])
        d = dim + rank if dim < 0 else dim
        if d != 0:
            refuse("unsupported_view", node, ctx.module_path(node), f"narrowing cat along dim {d} (only the leading axis is bound)")
        shp0 = shape_of(outs[0].type)[0]
        total = sum(shape_of(o.type)[0][0] for o in outs)
        ctx.define(node, Coll(_tensor_type((total,) + tuple(shp0[1:]), 32), concat([o.refs for o in outs])))
        ctx.record(self, node, {"parts": len(parts), "narrowed_leaves": n64}, emitted=f"I64ToI32_v1 x{n64}")


from verity_vllm.program.frontend.rules.patterns import PATTERN_RULES  # noqa: E402  (pattern rules are tried before the single-op rules of the same target)

REFERENCE_RULES = [*PATTERN_RULES, LinearRefRule(), MatmulRefRule(), ElementwiseRefRule(), BroadcastBiasAddRefRule(), LayerNormRefRule(),
                   RMSNormRefRule(), EmbeddingRefRule(), SdpaRefRule(), SoftmaxRefRule(), ArgmaxRefRule(), IntCatNarrowRule(), CastRefRule()]

__all__ = ["REFERENCE_RULES", "LinearRule", "LinearRefRule"]
