module

public import Proofs.Slowdown.CoveringTwoCosts
public import Proofs.Slowdown.CoveringWitness
public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Tactic.NormNum

/-! The two-cost covering bound is not vacuous at `c_w ≠ c_a`. -/

@[expose] public section

namespace Slowdown.SecurityProofs.CoveringTwoCosts

open Verity Finset Slowdown.Covering

def tcReads : Fin 14 → Finset (Fin 14) :=
  ![∅, ∅, ∅, ∅, {0, 3}, {0, 3}, {1, 3}, {1, 3}, {2, 3}, {4, 5}, {9, 6}, {10, 7}, {11, 8}, {2, 12}]

abbrev tcCircuit : Circuit where
  N := 14
  w := fun _ => 1
  isInput := fun g => decide (g.val < 4)
  reads := tcReads
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {13}

abbrev tcDecl : RegisteredCircuit := .ofCircuit tcCircuit ∅ (by simp)

def tcProd : Fin 6 → Fin 14 := ![4, 5, 6, 7, 8, 13]

def tcWeight : Fin 6 → Fin 14 := ![0, 0, 1, 1, 2, 2]

abbrev tcBlocks : MulBlocks tcDecl where
  nw := 6
  nt := 1
  P := univ
  m := fun p => {tcProd p.1}
  o := fun p => tcProd p.1
  T := fun _ => {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
  m_disjoint := by decide
  m_computed := by decide
  o_mem := by decide
  reach := fun p _ g hg => by
    rw [mem_singleton] at hg
    subst hg
    exact .refl (mem_singleton_self _)
  T_disjoint := fun j j' h => absurd (Subsingleton.elim j j') h
  T_θ := fun _ => disjoint_empty_right _

def tcSub (gs is : Finset (Fin 14))
    (h : is ⊆ gs ∧ (∀ g ∈ gs, g ∉ is → tcCircuit.reads g ⊆ gs) ∧ ∀ g ∈ gs, g ∈ sources tcCircuit → g ∈ is) :
    Subcircuit tcDecl.D :=
  ⟨gs, is, h.1, h.2.1, h.2.2⟩

def tcCover : Fin 10 → Subcircuit tcDecl.D :=
  ![tcSub {0, 3, 4} {0, 3} (by decide), tcSub {0, 3, 5} {0, 3} (by decide), tcSub {1, 3, 6} {1, 3} (by decide),
    tcSub {1, 3, 7} {1, 3} (by decide), tcSub {2, 3, 8} {2, 3} (by decide), tcSub {4, 5, 9} {4, 5} (by decide),
    tcSub {9, 6, 10} {9, 6} (by decide), tcSub {10, 7, 11} {10, 7} (by decide),
    tcSub {11, 8, 12} {11, 8} (by decide), tcSub {2, 12, 13} {2, 12} (by decide)]

theorem tc_covers : Covers tcDecl 3 2 tcCover := by
  refine ⟨?_, ?_⟩
  · unfold InferenceBlock
    decide
  · unfold Subcircuit.inner
    decide

theorem tc_parameterHard (s : ℕ) : Assumptions.ParameterHardAt tcDecl tcBlocks s (1 / 2) := by
  classical
  intro S _
  have hw : ∀ i : Fin 6, tcWeight i ∈ tcCircuit.reads (tcProd i) ∧ tcWeight i ∈ sources tcCircuit := by decide
  have hfib : ∀ g : Fin 14, (univ.filter fun i : Fin 6 => tcWeight i = g).card ≤ 2 := by decide
  have himg : (tcBlocks.weightsCompleted S).image tcWeight ⊆ S.ins := by
    intro g hg
    obtain ⟨i, hi, rfl⟩ := mem_image.1 hg
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    have hm := (mem_filter.1 hp).2 (mem_singleton_self _)
    obtain ⟨hgs, hgi⟩ := mem_sdiff.1 hm
    exact S.sources_ins _ (S.closed _ hgs hgi (hw p.1).1) (hw p.1).2
  have hle : (tcBlocks.weightsCompleted S).card ≤ 2 * ((tcBlocks.weightsCompleted S).image tcWeight).card :=
    card_le_mul_card_image _ 2 fun g _ =>
      (card_le_card (filter_subset_filter _ (subset_univ _))).trans (hfib g)
  have h2 : ((tcBlocks.weightsCompleted S).card : ℝ) ≤ 2 * S.ins.card := by
    exact_mod_cast hle.trans (Nat.mul_le_mul_left 2 (card_le_card himg))
  have hθ : S.ins \ tcDecl.θ = S.ins := sdiff_empty
  rw [hθ]
  linarith

theorem tc_path (i : Fin 6) (hi : i ∈ ({0, 1, 2, 3, 4} : Finset (Fin 6))) :
    FeedsWithin tcDecl.D (tcBlocks.T 0) (tcBlocks.o (i, 0)) 12 := by
  simp only [mem_insert, mem_singleton] at hi
  rcases hi with rfl | rfl | rfl | rfl | rfl
  · exact .step (h := 9) (by decide) (by decide) (.step (h := 10) (by decide) (by decide)
      (.step (h := 11) (by decide) (by decide) (.step (h := 12) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := 9) (by decide) (by decide) (.step (h := 10) (by decide) (by decide)
      (.step (h := 11) (by decide) (by decide) (.step (h := 12) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := 10) (by decide) (by decide)
      (.step (h := 11) (by decide) (by decide) (.step (h := 12) (by decide) (by decide) (.refl (by decide))))
  · exact .step (h := 11) (by decide) (by decide) (.step (h := 12) (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 12) (by decide) (by decide) (.refl (by decide))

theorem tc_chained : Assumptions.ActivationChainedAt tcDecl tcBlocks 5 1 (5, 0) := by
  refine ⟨mem_univ _, {12}, {0, 1, 2, 3, 4}, by decide, by decide, by decide,
    fun i hi => ⟨mem_univ _, by revert i; decide, 12, mem_singleton_self _, tc_path i hi⟩, ?_⟩
  intro Q hQ X _ hcut
  rcases X.eq_empty_or_nonempty with rfl | hX
  · have hWQ : 0 < (({0, 1, 2, 3, 4} : Finset (Fin 6)) \ Q).card := by
      have := le_card_sdiff Q ({0, 1, 2, 3, 4} : Finset (Fin 6))
      have h5 : ({0, 1, 2, 3, 4} : Finset (Fin 6)).card = 5 := by decide
      omega
    obtain ⟨i, hi⟩ := card_pos.1 hWQ
    rcases hcut i hi with ⟨x, hx⟩ | hno
    · simp at hx
    · exact absurd (by simpa using tc_path i (mem_sdiff.1 hi).1) (hno 12 (mem_singleton_self _))
  · exact_mod_cast card_pos.2 hX

theorem tc_mem : ((5, 0) : Fin 6 × Fin 1) ∈ Assumptions.chainedBlocksAt tcDecl tcBlocks 5 1 := by
  classical
  simp only [Assumptions.chainedBlocksAt, mem_filter]
  exact ⟨mem_univ _, tc_chained⟩

/-- The witness is at `c_w = 1/2`, `c_a = 1`, `D = 5`, `k = 2` and `s = 3`, with a chained multiplication block. -/
theorem coveringBoundTwoCosts_nonvacuous :
    ∃ (d : RegisteredCircuit) (B : MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
      cw ≠ ca ∧ (k : ℝ) < cw * D ∧ 0 < ca ∧ Assumptions.ParameterHardAt d B s cw ∧ Covers d s k I ∧
      (Assumptions.chainedBlocksAt d B D ca).Nonempty :=
  ⟨tcDecl, tcBlocks, 3, 2, 5, 1 / 2, 1, 10, tcCover, by norm_num, by norm_num, by norm_num, tc_parameterHard 3,
    tc_covers, ⟨(5, 0), tc_mem⟩⟩

open Slowdown.SecurityProofs.Covering in
/-- At `c_a = -1`, `CoveringBound`'s witness meets the other hypotheses and the count's bound is below every count,
so `c_a > 0` is needed. -/
theorem coveringBoundTwoCosts_needs_ca_pos :
    ∃ (d : RegisteredCircuit) (B : MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
      (k : ℝ) < cw * D ∧ Assumptions.ParameterHardAt d B s cw ∧ Covers d s k I ∧
      ¬ ((Assumptions.chainedBlocksAt d B D ca).card : ℝ) ≤ n * (k / cw * (k / ca) + k) := by
  have hhard : Assumptions.ParameterHardAt exDecl exBlocks 3 1 := by
    simpa only [one_mul] using (parameterHard_iff exBlocks 3 1 1).1 ex_parameterHard
  refine ⟨exDecl, exBlocks, 3, 2, 3, 1, -1, 6, exCover, by norm_num, hhard, ex_covers, ?_⟩
  have h0 := (Nat.cast_nonneg (Assumptions.chainedBlocksAt exDecl exBlocks 3 (-1)).card : (0 : ℝ) ≤ _)
  norm_num
  linarith

end Slowdown.SecurityProofs.CoveringTwoCosts
