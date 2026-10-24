import Definitions.NetworkCertifier.Guarantees.SampledAudit

/-! `NetTiming.SecurityProofs.SampledAuditAcceptedStated : NetTiming.Guarantees.SampledAuditAcceptedStated`, by
reflexivity: the guarantee is there for its record, which holds `SampledAuditAccepted`'s text. -/

/-- **The sampled audit's statement is recorded** (`NetTiming.Guarantees.SampledAuditAcceptedStated`). -/
theorem NetTiming.SecurityProofs.SampledAuditAcceptedStated : NetTiming.Guarantees.SampledAuditAcceptedStated := rfl
