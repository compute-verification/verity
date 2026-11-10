"""[GLM-5.3] MLA's latent absorptions as the H200 serves them (vLLM d9105ea80 `deepseek_v32/attention.py`), around the sparse
attention `vllm_dsa` binds: `bmm(q_nope^T, W_UK_T)` and `bmm(attn^T, W_UV)`, `batched_gemm{B,K,N,DOT}` per token, as the bmm alone
(`MlaAbsorbBmmRule`) or with the q-nope/value chain's transposes around it, bound before the generic views
(`MlaAbsorbTransposePatternRule`)."""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import torch

from verity.circuits.refs import array_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.patterns import Consumed
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_dsa import BF16, _missing, _rows, _value
from verity_vllm.program.frontend.torch_frontend import _permute

aten = torch.ops.aten


class MlaAbsorbBmmRule(OpRule):
    """`aten.bmm(x [B, M, K] bf16, W [B, K, N] bf16 parameter)`, MLA's latent absorptions (`W_UK_T`, `W_UV`): per token m,
    `batched_gemm{B,K,N,DOT}(x[:, m, :], W^T)` over all B heads; the node is [B, M, N]."""
    modes = frozenset({"profile"})
    info = RuleInfo("MlaAbsorbBmm[vllm-dsa]", 1, "op-binding", applies_to="aten.bmm(x [B, M, K], W [B, K, N]) with W a parameter",
                    preconditions=("x a bf16 value [B, M, K]; W a bf16 parameter [B, K, N] (W_UK_T or W_UV)",
                                   "DOT the target's k16 step (frontend.targets.tensor_core_dot)"),
                    evidence_domain="structural; numerics: batched_gemm (cuBLAS's batched bf16 GEMM)")

    def targets(self):
        return {aten.bmm.default}

    def applies(self, node, ctx):
        x, w = node.args[0], node.args[1]
        if ctx.weight_of(w) is None:
            return "the right operand is not a parameter"
        if (r := _missing(ctx, ["batched_gemm"])):
            return r
        if (r := _value(ctx, x, BF16, (None, None, None), "x")):
            return r
        B, _, K = shape(ctx, x)
        if (r := _value(ctx, w, BF16, (B, K, None), "W")):
            return r
        from verity_vllm.program.frontend import targets as TG
        try:
            TG.tensor_core_dot(active_target_profile())
        except TG.UnregisteredTarget as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets as TG
        x, w = node.args[0], node.args[1]
        B, M, K = shape(ctx, x)
        N = shape(ctx, w)[2]
        DOT = TG.tensor_core_dot(active_target_profile())
        fn = ctx.kind("batched_gemm", node, B=B, K=K, N=N, DOT=DOT)
        where = ctx.module_path(node)
        xa = _permute(_rows(ctx, x, 16), (1, 0, 2), where)
        wt = _permute(_rows(ctx, w, 16), (0, 2, 1), where)
        h = ctx.hint_name(node)
        outs = [ctx.call(fn, xa[m], wt, node=node, rule=self, name=f"{h}[{m}]") for m in range(M)]
        ctx.define(node, _permute(array_of(outs), (1, 0, 2), where))
        wl = ctx.weight_of(w)
        ctx.bind_weight_role(w, self, "mla.absorb", B=B, K=K, N=N)
        ctx.record(self, node, {"B": B, "M": M, "K": K, "N": N, "DOT": DOT.name, "weight": wl.name}, emitted=f"{fn.id} x{M}")


_ASSERTS = {aten._assert_tensor_metadata.default}


def _single_data_user(node):
    users = [u for u in node.users if u.target not in _ASSERTS]
    return users[0] if len(users) == 1 else None


def _transpose_01(node, ctx) -> bool:
    if not isinstance(node, torch.fx.Node) or node.target is not aten.transpose.int or len(shape(ctx, node) or ()) != 3:
        return False
    d0, d1 = int(node.args[1]), int(node.args[2])
    d0 = d0 + 3 if d0 < 0 else d0
    d1 = d1 + 3 if d1 < 0 else d1
    return {d0, d1} == {0, 1}


class MlaAbsorbTransposePatternRule(OpRule):
    """The source spelling ``bmm(x.transpose(0,1), W).transpose(0,1)`` of an MLA absorb.

    The first transpose can span millions of leaves over the composite reference sequence made by splitting q into
    q-nope/q-pe.  It has no compact generic view.  This rule recognises the whole single-use chain and emits the same
    per-token ``batched_gemm`` calls directly from untransposed x, so the large intermediate never enters the Program.
    """

    modes = frozenset({"profile"})
    info = RuleInfo(
        "MlaAbsorbTranspose[vllm-dsa]",
        1,
        "op-binding",
        applies_to="x [M,B,K].transpose(0,1) -> aten.bmm(., W [B,K,N] parameter) -> transpose(0,1)",
        preconditions=(
            "both transposes swap axes 0 and 1 of rank-3 bf16 tensors",
            "the first transpose and bmm each have exactly one non-assert reader, so no absorbed intermediate escapes",
            "W is a bf16 parameter [B,K,N], batched_gemm is registered, and the target's k16 DOT is registered",
        ),
        evidence_domain="structural; numerics: batched_gemm (cuBLAS's batched bf16 GEMM)",
    )

    def targets(self):
        return {aten.transpose.int}

    def _match(self, node, ctx):
        if not _transpose_01(node, ctx):
            return "the anchor is not a rank-3 transpose of axes 0 and 1"
        x = node.args[0]
        if (r := _value(ctx, x, BF16, (None, None, None), "x")):
            return r
        bmm = _single_data_user(node)
        if bmm is None or bmm.target is not aten.bmm.default or bmm.args[0] is not node:
            return "the transpose is not read only as aten.bmm's left operand"
        out = _single_data_user(bmm)
        if out is None or not _transpose_01(out, ctx) or out.args[0] is not bmm:
            return "the bmm is not read only by a trailing transpose of axes 0 and 1"
        w = bmm.args[1]
        if ctx.weight_of(w) is None:
            return "the right operand is not a parameter"
        M, B, K = shape(ctx, x)
        if (r := _value(ctx, w, BF16, (B, K, None), "W")):
            return r
        N = shape(ctx, w)[2]
        if shape(ctx, bmm) != (B, M, N) or shape(ctx, out) != (M, B, N):
            return f"the bmm / trailing transpose shapes are {shape(ctx, bmm)} / {shape(ctx, out)}, not {(B, M, N)} / {(M, B, N)}"
        if (r := _missing(ctx, ["batched_gemm"])):
            return r
        from verity_vllm.program.frontend import targets as TG

        try:
            TG.tensor_core_dot(active_target_profile())
        except TG.UnregisteredTarget as e:
            return f"numerics_unregistered: {e}"
        return {"x": x, "w": w, "bmm": bmm, "out": out, "M": M, "B": B, "K": K, "N": N}

    def applies(self, node, ctx):
        m = self._match(node, ctx)
        return True if isinstance(m, dict) else f"not the MLA absorb transpose pattern ({m})"

    def emit(self, node, ctx):
        from verity_vllm.program.frontend import targets as TG

        m = self._match(node, ctx)
        assert isinstance(m, dict)
        DOT = TG.tensor_core_dot(active_target_profile())
        fn = ctx.kind("batched_gemm", node, B=m["B"], K=m["K"], N=m["N"], DOT=DOT)
        x = _rows(ctx, m["x"], 16)
        wt = _permute(_rows(ctx, m["w"], 16), (0, 2, 1), ctx.module_path(node))
        h = ctx.hint_name(m["bmm"])
        outs = [ctx.call(fn, x[i], wt, node=m["out"], rule=self, name=f"{h}[{i}]") for i in range(m["M"])]
        for n in (node, m["bmm"]):
            ctx.env[n] = Consumed(self.rule_id, m["out"])
            for u in n.users:
                if u.target in _ASSERTS:
                    ctx.env[u] = None
        ctx.define(m["out"], array_of(outs))
        wl = ctx.weight_of(m["w"])
        ctx.bind_weight_role(m["w"], self, "mla.absorb", B=m["B"], K=m["K"], N=m["N"])
        ctx.record(
            self,
            node,
            {"B": m["B"], "M": m["M"], "K": m["K"], "N": m["N"], "DOT": DOT.name, "weight": wl.name,
             "chain": [node.name, m["bmm"].name, m["out"].name]},
            emitted=f"{fn.id} x{m['M']}",
        )


VLLM_DSA_ABSORB_RULES = [MlaAbsorbBmmRule()]
VLLM_DSA_PATTERN_RULES = [MlaAbsorbTransposePatternRule()]

__all__ = ["MlaAbsorbBmmRule", "MlaAbsorbTransposePatternRule", "VLLM_DSA_ABSORB_RULES", "VLLM_DSA_PATTERN_RULES"]
