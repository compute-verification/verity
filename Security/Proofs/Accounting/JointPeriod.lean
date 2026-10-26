import Definitions.Accounting.Guarantees.JointPeriod
import Proofs.Core.UnionBound

/-!
# Proof of the joint guarantee over a period and over a horizon of periods

Each audit's per-period guarantee (`PeriodRun.Sound`) bounds its failure over its own coordinate of the coins, at every
value of the others, so it bounds it over all of them (`Pouw.pr_le_of_update`), and a union bound over the five
audits sums the errors (`Pouw.union_bound_update`). Over a horizon the union runs over every period's audits
(`Pouw.union_bound_horizon`).
-/

namespace Accounting.SecurityProofs

/-- **The joint guarantee over a period** (`Accounting.Guarantees.JointPeriod`). -/
theorem JointPeriod : Accounting.Guarantees.JointPeriod := by
  intro ι α I _ _ _ Ω _ _ d C r c p δ h
  exact Pouw.union_bound_update r.own (fun a ξ => r.accepts a ξ ∧ ¬ r.Holds c p a ξ) δ h

/-- **The joint guarantee over a horizon of periods** (`Accounting.Guarantees.JointHorizon`). -/
theorem JointHorizon : Accounting.Guarantees.JointHorizon := by
  intro ι α I _ _ _ Ω _ _ d C m r c p δ h
  exact Pouw.union_bound_horizon (fun j a => (r j).own a)
    (fun j a ξ => (r j).accepts a ξ ∧ ¬ (r j).Holds (c j) p a ξ) δ h

end Accounting.SecurityProofs
