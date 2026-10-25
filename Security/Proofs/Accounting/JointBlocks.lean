import Definitions.Accounting.Guarantees.JointBlocks
import Proofs.Core.UnionBound

/-!
# Proof of the joint guarantee over a horizon of server-time blocks and the wipes between them

The blocks' audits are a horizon of periods (`Pouw.union_bound_horizon_with`, each audit's guarantee over its own
coordinate). The extra event is some wipe's survivor bound or some block's memory term failing, whose probability is at
most `∑_b ε_wipe b + ∑_b δ_PoUS b` by a union bound over the blocks (`Pouw.Proofs.pr_exists_le_sum`).
-/

namespace Accounting.SecurityProofs

/-- Some event of a finite family holds with probability at most the sum of their bounds. -/
theorem pr_exists_univ_le {β J : Type} [Fintype β] [Fintype J] (E : J → β → Prop) (η : J → ℝ)
    (h : ∀ j, Pouw.pr (E j) ≤ η j) : Pouw.pr (fun x => ∃ j, E j x) ≤ ∑ j, η j :=
  (Pouw.pr_mono fun _ ⟨j, hj⟩ => ⟨j, Finset.mem_univ j, hj⟩).trans <|
    (Pouw.Proofs.pr_exists_le_sum Finset.univ E).trans (Finset.sum_le_sum fun j _ => h j)

/-- **The joint guarantee over a horizon of server-time blocks** (`Accounting.Guarantees.JointHorizonBlocks`). -/
theorem JointHorizonBlocks : Accounting.Guarantees.JointHorizonBlocks := by
  intro ι α I _ _ _ _ Ω _ _ d C B s c p f δ εWipe δPous hS hW hM
  have hF := (Pouw.pr_or_le _ _).trans (add_le_add (pr_exists_univ_le _ εWipe hW) (pr_exists_univ_le _ δPous hM))
  refine (Pouw.pr_mono fun ξ h => ?_).trans
    ((Pouw.union_bound_horizon_with _ hF (fun b a => (s.block b).own a)
      (fun b a ξ => (s.block b).accepts a ξ ∧ ¬ (s.block b).Holds (c b) p a ξ) δ hS).trans_eq (by ring))
  rcases h with h | h | h
  · exact Or.inr h
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)

end Accounting.SecurityProofs
