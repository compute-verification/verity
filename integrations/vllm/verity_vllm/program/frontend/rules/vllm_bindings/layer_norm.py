"""`nn.LayerNorm` with weight and bias (GPT-NeoX): `aten.layer_norm` on bf16 rows, the kernel torch launches for it."""

from __future__ import annotations

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, norm_kind, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _row_value, aten
from verity.ml.definitions.registry import layer_norm as _LN


class LayerNormAtenRule(OpRule):
    """`aten.layer_norm(x bf16 [.., N], [N], w bf16 [N], b bf16 [N], eps)` -> `layer_norm_aten{N,EPS}` (LayerNormAten_v1) per row, `w`
    and `b` shared: torch 2.13's `vectorized_layer_norm_kernel<bf16, float>`, the path taken for N a multiple of 4 on 8-byte aligned
    rows (`LayerNormKernelImplInternal`; every contiguous bf16 activation and parameter is).  vLLM's batch-invariant mode overrides
    no `aten::layer_norm`, so this is the kernel it launches on every architecture: its 128-thread Welford tree is fixed by the source
    (`num_threads() = 4 * C10_WARP_SIZE`), not by the SM.  An `eps` the export omits is `F.layer_norm`'s default 1e-5."""
    modes = frozenset({"profile"})
    info = RuleInfo("LayerNormAten[aten]", 1, "op-binding",
                    applies_to="aten.layer_norm.default(x bf16 [.., N], normalized_shape=[N], weight bf16 [N], bias bf16 [N], eps)",
                    preconditions=("x is a bf16 value collection of rank >= 1", "normalized_shape == [x.shape[-1]]",
                                   "weight and bias are both present, bf16 [N]", f"N % {_LN.VEC} == 0 (the vectorized kernel)",
                                   "eps is a static float (absent: 1e-5)"),
                    evidence_domain="structural; numerics: LayerNormAten_v1 (quarantine; torch 2.13 vectorized_layer_norm_kernel)")

    def targets(self):
        return {aten.layer_norm.default}

    @staticmethod
    def _operands(node):
        a = list(node.args) + [None] * (5 - len(node.args))
        kw = node.kwargs or {}
        eps = kw.get("eps", a[4])
        return a[0], a[1], kw.get("weight", a[2]), kw.get("bias", a[3]), 1e-5 if eps is None else eps

    def applies(self, node, ctx):
        x, ns, w, b, eps = self._operands(node)
        r = _row_value(ctx, x, "x")
        if r:
            return r
        if w is None or b is None:
            return "numerics_unregistered: LayerNormAten_v1 is the affine layer norm with both weight and bias"
        for nm, t in (("weight", w), ("bias", b)):
            r = _row_value(ctx, t, nm)
            if r:
                return r
            if not isinstance(ctx.val(t), Coll):
                return f"{nm} is not a value collection"
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            return f"eps {eps!r} is not a static number"
        xs = shape(ctx, x)
        N = xs[-1]
        if not isinstance(ns, (list, tuple)) or tuple(ns) != (N,):
            return f"normalized_shape {ns} is not the last dim of x {xs}"
        if shape(ctx, w) != (N,) or shape(ctx, b) != (N,):
            return f"weight {shape(ctx, w)} / bias {shape(ctx, b)} are not [N={N}]"
        if N % _LN.VEC:
            return f"numerics_unregistered: N={N} is not a multiple of {_LN.VEC}: torch takes the unvectorized RowwiseMoments path"
        return True

    def emit(self, node, ctx):
        xn, _ns, wn, bn, eps = self._operands(node)
        rows, lead = rows_of(ctx.coll(xn))
        wc, bc = ctx.coll(wn), ctx.coll(bn)
        N, eps = rows[0].type.n, float(eps)
        fn, value = norm_kind(ctx, "layer_norm_aten", node, N, eps)
        outs = [ctx.call(fn, r, wc, bc, *value, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, from_rows(outs, lead))
        ctx.bind_weight_role(wn, self, "layer_norm.weight", N=N, EPS=eps)
        ctx.bind_weight_role(bn, self, "layer_norm.bias", N=N, EPS=eps)
        ctx.record(self, node, {"N": N, "EPS": eps, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


__all__ = ["LayerNormAtenRule"]
