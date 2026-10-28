import Definitions.Nci.Guarantees.CoverChained
import Proofs.Nci.CoveringTwoCostsWitness

/-!
# `CoverTrainingChained` is not vacuous

One reference circuit, one declaration that contains it and one division, where every hypothesis of
`Nci.Cover.Guarantees.CoverTrainingChained` holds and `𝓜*` is nonempty (`coverTrainingChained_nonvacuous`). The circuit
is the two-cost covering bound's witness (`Proofs.Nci.CoveringTwoCostsWitness`, `tcDecl`), whose gates are one bit
wide, so its bits (`bitDecl tcDecl`) are its gates (`bt`) with nothing registered, and its multiplication blocks
(`tcBlocks`) carry over as `bBlocks`: parameter-hard at `c_w = 1/2` at every `s`, with `p₅` chained at
`(D, c_a) = (5, 1)`. The declaration is the circuit itself (`selfContainment`), divided into one IU per computed gate
(`onePerGate`): each has one gate and reads two bits, so it complies at `Γ = 1`, `α = 2`, and `α Γ = 2 < c_w D = 5/2`.
The block size is `s = w₀ Γ + α Γ + width(θ) = 3`.
-/

namespace Nci.Cover.SecurityProofs.ChainedWitness

open Verity Finset Nci.Covering Nci.SecurityProofs.CoveringTwoCosts

theorem bitN : (bitDecl tcDecl).D.N = 14 := by decide

/-- Gate `g` of the witness as its one bit. -/
def bt (g : Fin 14) : Fin (bitDecl tcDecl).D.N := Fin.cast bitN.symm g

theorem bt_injective : Function.Injective bt := Fin.cast_injective bitN.symm

theorem bitθ : (bitDecl tcDecl).θ = ∅ := by decide

/-- The two-cost witness's multiplication blocks (`tcBlocks`), on its bits. -/
abbrev bBlocks : Nci.Covering.MulBlocks (bitDecl tcDecl) where
  nw := 6
  nt := 1
  P := univ
  m := fun p => {bt (tcProd p.1)}
  o := fun p => bt (tcProd p.1)
  T := fun _ => ({4, 5, 6, 7, 8, 9, 10, 11, 12, 13} : Finset (Fin 14)).map ⟨bt, bt_injective⟩
  m_disjoint := by decide
  m_computed := by decide
  o_mem := by decide
  reach := fun p _ g hg => by
    rw [mem_singleton] at hg
    subst hg
    exact .refl (mem_singleton_self _)
  T_disjoint := fun j j' h => absurd (Subsingleton.elim j j') h
  T_θ := fun _ => by rw [bitθ]; exact disjoint_empty_right _

theorem b_parameterHard (s : ℕ) : Assumptions.ParameterHardAt (bitDecl tcDecl) bBlocks s (1 / 2) := by
  classical
  intro S _
  have hw : ∀ i : Fin 6, bt (tcWeight i) ∈ (bitDecl tcDecl).D.reads (bt (tcProd i)) ∧
      bt (tcWeight i) ∈ sources (bitDecl tcDecl).D := by decide
  have hfib : ∀ g : Fin (bitDecl tcDecl).D.N, (univ.filter fun i : Fin 6 => bt (tcWeight i) = g).card ≤ 2 := by
    decide
  have himg : (bBlocks.weightsCompleted S).image (bt ∘ tcWeight) ⊆ S.ins := by
    intro g hg
    obtain ⟨i, hi, rfl⟩ := mem_image.1 hg
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    have hm := (mem_filter.1 hp).2 (mem_singleton_self _)
    obtain ⟨hgs, hgi⟩ := mem_sdiff.1 hm
    exact S.sources_ins _ (S.closed _ hgs hgi (hw p.1).1) (hw p.1).2
  have hle : (bBlocks.weightsCompleted S).card ≤ 2 * ((bBlocks.weightsCompleted S).image (bt ∘ tcWeight)).card :=
    card_le_mul_card_image _ 2 fun g _ =>
      (card_le_card (filter_subset_filter _ (subset_univ _))).trans (hfib g)
  have h2 : ((bBlocks.weightsCompleted S).card : ℝ) ≤ 2 * S.ins.card := by
    exact_mod_cast hle.trans (Nat.mul_le_mul_left 2 (card_le_card himg))
  rw [bitθ, sdiff_empty]
  linarith

theorem b_path (i : Fin 6) (hi : i ∈ ({0, 1, 2, 3, 4} : Finset (Fin 6))) :
    FeedsWithin (bitDecl tcDecl).D (bBlocks.T 0) (bBlocks.o (i, 0)) (bt 12) := by
  simp only [mem_insert, mem_singleton] at hi
  rcases hi with rfl | rfl | rfl | rfl | rfl
  · exact .step (h := bt 9) (by decide) (by decide) (.step (h := bt 10) (by decide) (by decide)
      (.step (h := bt 11) (by decide) (by decide) (.step (h := bt 12) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := bt 9) (by decide) (by decide) (.step (h := bt 10) (by decide) (by decide)
      (.step (h := bt 11) (by decide) (by decide) (.step (h := bt 12) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := bt 10) (by decide) (by decide)
      (.step (h := bt 11) (by decide) (by decide) (.step (h := bt 12) (by decide) (by decide) (.refl (by decide))))
  · exact .step (h := bt 11) (by decide) (by decide) (.step (h := bt 12) (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := bt 12) (by decide) (by decide) (.refl (by decide))

theorem b_chained : Assumptions.ActivationChainedAt (bitDecl tcDecl) bBlocks 5 1 (5, 0) := by
  refine ⟨mem_univ _, {bt 12}, {0, 1, 2, 3, 4}, by decide, by decide, by decide,
    fun i hi => ⟨mem_univ _, by revert i; decide, bt 12, mem_singleton_self _, b_path i hi⟩, ?_⟩
  intro Q hQ X _ hcut
  rcases X.eq_empty_or_nonempty with rfl | hX
  · have hWQ : 0 < (({0, 1, 2, 3, 4} : Finset (Fin 6)) \ Q).card := by
      have := le_card_sdiff Q ({0, 1, 2, 3, 4} : Finset (Fin 6))
      have h5 : ({0, 1, 2, 3, 4} : Finset (Fin 6)).card = 5 := by decide
      omega
    obtain ⟨i, hi⟩ := card_pos.1 hWQ
    rcases hcut i hi with ⟨x, hx⟩ | hno
    · simp at hx
    · exact absurd (by simpa using b_path i (mem_sdiff.1 hi).1) (hno (bt 12) (mem_singleton_self _))
  · exact_mod_cast card_pos.2 hX

theorem b_mem : ((5, 0) : Fin 6 × Fin 1) ∈ Assumptions.chainedBlocksAt (bitDecl tcDecl) bBlocks 5 1 := by
  classical
  simp only [Assumptions.chainedBlocksAt, mem_filter]
  exact ⟨mem_univ _, b_chained⟩

/-- The witness contains itself. -/
def selfContainment : Containment tcDecl.D tcDecl.D where
  emb := id
  emb_inj := Function.injective_id
  emb_w _ := rfl
  isInput_emb _ := rfl
  reads_emb g _ := by ext; simp
  op_emb _ _ _ := rfl
  outputs_emb _ h := h

/-- One IU per computed gate. -/
def onePerGate : Partition tcDecl.D 10 where
  unit g := if h : g.val < 4 then none else some ⟨g.val - 4, by have : g.val < 14 := g.isLt; omega⟩
  input_iff g := by revert g; decide

theorem complies : Complies 1 2 tcDecl := by
  refine ⟨10, onePerGate, fun u => ?_⟩
  have h : ∀ u, tcDecl.D.size (onePerGate.gates u) = 1 ∧ novelReads tcDecl (onePerGate.gates u) = 2 := by decide
  rw [(h u).1, (h u).2]
  norm_num

/-- **`CoverTrainingChained` is not vacuous, at `c_w ≠ c_a`**: the two-cost covering bound's witness
(`Proofs.Nci.CoveringTwoCostsWitness`), read bit by bit (`bitDecl`), contained in itself and divided into one IU per
gate, meets every hypothesis of `Nci.Cover.Guarantees.CoverTrainingChained` at `Γ = 1`, `α = 2`, `c_w = 1/2`,
`c_a = 1`, `D = 5` and `s = 3`, with a chained multiplication block. -/
theorem coverTrainingChained_nonvacuous :
    ∃ (Γ : ℕ) (α : ℝ) (c d : Declaration) (M : Containment c.D d.D) (B : Nci.Covering.MulBlocks (bitDecl c)) (s D w₀ : ℕ)
      (cw ca : ℝ),
      0 ≤ α ∧ Complies Γ α d ∧ Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M) ∧
      Assumptions.ParameterHardAt (bitDecl c) B s cw ∧ α * Γ < cw * D ∧ 0 < ca ∧ (∀ g, c.D.w g ≤ w₀) ∧
      w₀ * Γ + α * Γ + c.D.width c.θ ≤ s ∧ cw ≠ ca ∧ (Assumptions.chainedBlocksAt (bitDecl c) B D ca).Nonempty :=
  ⟨1, 2, tcDecl, tcDecl, selfContainment, bBlocks, 3, 5, 1, 1 / 2, 1, by norm_num, complies,
    fun _ _ h => absurd h (notMem_empty _), b_parameterHard 3, by norm_num, by norm_num, fun _ => le_rfl,
    by norm_num [Circuit.width], by norm_num, ⟨(5, 0), b_mem⟩⟩

end Nci.Cover.SecurityProofs.ChainedWitness
