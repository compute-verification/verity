import Definitions.Accounting.Guarantees.JointBlocksAbstract
import Proofs.Core.UnionBound

/-!
# Proof of the joint guarantee over a horizon of blocks with extra events

The extra events' union is at most their bounds summed (`Pouw.Proofs.pr_exists_le_sum`), and
`Pouw.union_bound_horizon_with` adds it to the union over every block's audits.
-/

namespace Accounting.SecurityProofs

/-- **The joint guarantee over a horizon of blocks, with extra events**
(`Accounting.Guarantees.JointHorizonBlocksAbstract`). -/
theorem JointHorizonBlocksAbstract : Accounting.Guarantees.JointHorizonBlocksAbstract := by
  intro ι α I _ _ _ Ω _ _ d C B W r c p δ F ε h hF
  have hU : Pouw.pr (fun ξ => ∃ w, F w ξ) ≤ ∑ w, ε w :=
    (Pouw.pr_mono fun _ ⟨w, hw⟩ => ⟨w, Finset.mem_univ w, hw⟩).trans <|
      (Pouw.Proofs.pr_exists_le_sum Finset.univ F).trans (Finset.sum_le_sum fun w _ => hF w)
  exact Pouw.union_bound_horizon_with _ hU (fun b a => (r b).own a)
    (fun b a ξ => (r b).accepts a ξ ∧ ¬ (r b).Holds (c b) p a ξ) δ h

end Accounting.SecurityProofs
