"""Fused add + RMSNorm under the batch-invariant dispatch, derived."""

from __future__ import annotations

from verity_vllm.program.frontend.rules.base import RuleInfo
from verity_vllm.program.frontend.rules.profile import FusedAddRmsNormRule as _FusedAddRmsNormRule
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED, _fused_norm_dispatch_ok


# -- fused add + rms norm: the batch-invariant dispatch, derived ------------------------------------------------------------


class FusedAddRmsNormBIRule(_FusedAddRmsNormRule):
    """`FusedAddRmsNorm[vllm]` with the kernel dispatch DERIVED: binds `RMSNormFusedCuda_v2` (vLLM's generic fused_add_rms_norm_kernel,
    the one launched under batch invariance) only when the export shows the batch-invariant path (`observe_export_facts`); otherwise
    refuses — the width-8 kernel (`RMSNormFusedCuda_v1`) is superseded and not in the profile's vocabulary."""
    info = RuleInfo("FusedAddRmsNorm[vllm-bi]", 1, "in-place-binding", applies_to="_C.fused_add_rms_norm(Tensor! x, Tensor! residual, weight, eps)",
                    preconditions=("the export's fused_add_rms_norm dispatch was OBSERVED as batch-invariant (op reached via `rms_norm_batch_invariant`; "
                                   "BI Triton overrides installed and launched in the same graph) — derived, never assumed",)
                    + tuple(_FusedAddRmsNormRule.info.preconditions),
                    evidence_domain=_FusedAddRmsNormRule.info.evidence_domain)

    def applies(self, node, ctx):
        r = _fused_norm_dispatch_ok()
        if r:
            return r
        return super().applies(node, ctx)

    def emit(self, node, ctx):
        super().emit(node, ctx)
        # the parent recorded this application; add the derived dispatch to that record (one application per node)
        rec = ctx.applications[-1]
        if rec.get("node") == node.name:
            rec["rule"], rec["version"] = self.rule_id, self.info.version
            rec["preconditions_checked"] = {**rec.get("preconditions_checked", {}),
                                            "fused_add_rms_norm_dispatch": _OBSERVED["export_facts"].get("fused_add_rms_norm_dispatch"),
                                            "dispatch_evidence": _OBSERVED["export_facts"].get("evidence")}
