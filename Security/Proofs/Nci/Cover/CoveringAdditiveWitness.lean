import Proofs.Nci.Cover.CoveringAdditive
import Proofs.Nci.CoveringTwoCostsWitness

/-!
# The additive covering bound is not vacuous

`CoveringBoundTwoCosts`'s witness (`Proofs.Nci.CoveringTwoCostsWitness`) meets every hypothesis of
`Nci.Cover.Guarantees.CoveringBoundAdditive` at `c_w = 1/2`, `c_a = 1`, `D = 5`, `k = 2`, `s = 3`, with a chained
multiplication block (`coveringBoundAdditive_nonvacuous`). The weight gates a block completes a product from are among
its inputs and outside the token computation, which holds only computed gates, so it is conditionally parameter-hard at
`c_w = 1/2` at every `s` (`tc_conditional`).
-/

namespace Nci.Cover.SecurityProofs.CoveringAdditive

open Finset Nci.SecurityProofs.CoveringTwoCosts

/-- Conditionally parameter-hard at `c_w = 1/2`, at every `s`: the weight gates are inputs, outside `θ = ∅` and outside
the token computation. -/
theorem tc_conditional (s : ℕ) : Assumptions.ConditionalParameterHardAt tcDecl tcBlocks s (1 / 2) := by
  classical
  intro S _
  have hw : ∀ i : Fin 6, tcWeight i ∈ tcCircuit.reads (tcProd i) ∧ tcWeight i ∈ Nci.Covering.sources tcCircuit ∧
      tcWeight i ∉ univ.biUnion tcBlocks.T := by decide
  have hfib : ∀ g : Fin 14, (univ.filter fun i : Fin 6 => tcWeight i = g).card ≤ 2 := by decide
  have himg : (tcBlocks.weightsCompleted S).image tcWeight ⊆ S.ins \ (tcDecl.θ ∪ univ.biUnion tcBlocks.T) := by
    intro g hg
    obtain ⟨i, hi, rfl⟩ := mem_image.1 hg
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    have hm := (mem_filter.1 hp).2 (mem_singleton_self _)
    obtain ⟨hgs, hgi⟩ := mem_sdiff.1 hm
    obtain ⟨hr, hsrc, hT⟩ := hw p.1
    refine mem_sdiff.2 ⟨S.sources_ins _ (S.closed _ hgs hgi hr) hsrc, ?_⟩
    rw [mem_union, not_or]
    exact ⟨notMem_empty _, hT⟩
  have hle : (tcBlocks.weightsCompleted S).card ≤ 2 * ((tcBlocks.weightsCompleted S).image tcWeight).card :=
    card_le_mul_card_image _ 2 fun g _ =>
      (card_le_card (filter_subset_filter _ (subset_univ _))).trans (hfib g)
  have h2 : ((tcBlocks.weightsCompleted S).card : ℝ) ≤
      2 * (S.ins \ (tcDecl.θ ∪ univ.biUnion tcBlocks.T)).card := by
    exact_mod_cast hle.trans (Nat.mul_le_mul_left 2 (card_le_card himg))
  linarith

/-- **The additive covering bound is not vacuous**: one declaration, its multiplication blocks and a covering satisfy
every hypothesis of `CoveringBoundAdditive` at `c_w = 1/2`, `c_a = 1`, with a chained multiplication block. -/
theorem coveringBoundAdditive_nonvacuous :
    ∃ (d : Verity.RegisteredCircuit) (B : Nci.Covering.MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ)
      (I : Fin n → Nci.Covering.Subcircuit d.D),
      (k : ℝ) < cw * D ∧ 0 < ca ∧ Assumptions.ConditionalParameterHardAt d B s cw ∧ Nci.Covering.Covers d s k I ∧
      (Nci.Assumptions.chainedBlocksAt d B D ca).Nonempty :=
  ⟨tcDecl, tcBlocks, 3, 2, 5, 1 / 2, 1, 10, tcCover, by norm_num, by norm_num, tc_conditional 3, tc_covers,
    ⟨(5, 0), tc_mem⟩⟩

end Nci.Cover.SecurityProofs.CoveringAdditive
