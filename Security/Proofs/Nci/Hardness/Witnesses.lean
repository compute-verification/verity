import Definitions.Nci.Guarantees.Hardness.Witnesses
import Proofs.Nci.Hardness
import Proofs.Nci.Ldl.Witnesses

/-!
# Priced reads: the honest and nontrivial lemmas' proofs

The one-product instance as a counted operation (`Nci.Ldl.Honest.O`): the product's gate `{1}` is the one block, so a
set of at most one operation gate computes the product whole exactly when it is `{1}`, which reads one unregistered
input bit. So the product is `Priced` at `β = 1` per activation occurrence and not at `β = 2` (`not_priced_two`), and
each nontrivial lemma's changed instance fails that premise, or `ActivationRate`, and meets the others.
-/

namespace Nci.Ldl.Honest

open Verity Nci.Hardness

theorem ops_empty : O.ops ∅ = ∅ := by
  decide

theorem ops_block : O.ops {1} = {()} := by
  decide

theorem whole_block : O.whole {1} = {()} := by
  decide

theorem width_block : d.D.width (d.D.into {1} \ (d.θ ∪ ∅)) = 1 := by
  decide

theorem priced_one : Assumptions.Priced d O (fun _ => 1) 0 1 (fun _ => ∅) := by
  intro S hS _
  rw [show d.D.computed = {1} from computed_eq, Finset.subset_singleton_iff] at hS
  rcases hS with rfl | rfl
  · rw [ops_empty]
    simp
  · rw [ops_block, width_block]
    simp

theorem not_priced_two : ¬ Assumptions.Priced d O (fun _ => 2) 0 1 (fun _ => ∅) := fun h => by
  have h1 := h {1} (by rw [show d.D.computed = {1} from computed_eq]) (by decide)
  rw [ops_block, width_block] at h1
  norm_num at h1

theorem fanOut_O : O.FanOut 1 := fun v => by
  cases v
  decide

theorem wholeIn : O.WholeIn P := fun p _ => ⟨0, by cases p; decide⟩

end Nci.Ldl.Honest

namespace Nci.Hardness.SecurityProofs

open Verity Nci.Ldl Nci.Ldl.Honest

theorem FanOutBlockCap_honest : Guarantees.FanOutBlockCap_honest :=
  ⟨fanOut_O, priced_one⟩

theorem FanOutBlockCap_nontrivial : Guarantees.FanOutBlockCap_nontrivial := by
  unfold Guarantees.FanOutBlockCap_nontrivial
  rw [whole_block]
  norm_num

theorem FanOutBoundPriced_honest : Guarantees.FanOutBoundPriced_honest :=
  ⟨ckPartition, wholeIn, fanOut_O, priced_one⟩

theorem FanOutBoundPriced_nontrivial : Guarantees.FanOutBoundPriced_nontrivial := by
  unfold Guarantees.FanOutBoundPriced_nontrivial
  rw [show d.D.size d.D.computed = 1 from size_computed, show M.J.card = 1 from rfl]
  norm_num

theorem PricedOfActivationRate_honest : Guarantees.PricedOfActivationRate_honest :=
  ⟨fun p _ => by cases p; decide, consistent _, evaluatesBlocks, activationRate _⟩

theorem PricedOfActivationRate_nontrivial : Guarantees.PricedOfActivationRate_nontrivial := by
  unfold Guarantees.PricedOfActivationRate_nontrivial
  rw [novelReads_block, show R.P.gates 0 = {1} from gates_eq 0, ops_block, Nat.clog_one_right]
  norm_num

end Nci.Hardness.SecurityProofs
