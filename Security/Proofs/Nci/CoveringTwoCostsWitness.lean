import Proofs.Nci.CoveringTwoCosts
import Proofs.Nci.CoveringWitness
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.NormNum

/-!
# The two-cost covering bound is not vacuous, at `c_w ≠ c_a`

One circuit where every hypothesis of `Nci.Guarantees.CoveringBoundTwoCosts` holds with `c_w = 1/2 < c_a = 1` and `𝓜*` is
nonempty (`coveringBoundTwoCosts_nonvacuous`), at `D = 5`, `k = 2` and `s = 3`: the regime two costs are for, weights
cheaper than activations. Six weights share three weight gates `u₀ u₁ u₂` (gates 0–2), two weights a gate, and a token
input `x` (gate 3); nothing is registered (`θ = ∅`). The products `pᵢ = uᵢ/₂ · x` of weights 0–4 (gates 4–8) feed the
activation `a = (((p₀ + p₁) + p₂) + p₃) + p₄` (gates 9–12), and `p₅ = u₂ · a` (gate 13) is chained through `a` by the five
weights 0–4. A block that completes a product holds its weight gate as an input, and each gate serves two weights, so the
circuit is parameter-hard at `c_w = 1/2` at every `s`; `a` is one gate, so every cut has at least `c_a = 1`. Ten blocks
of three gates, each with two inputs, cover it.

`c_a > 0` can't be dropped (`coveringBoundTwoCosts_needs_ca_pos`): at `c_a = -1`, on `CoveringBound`'s witness
(`Proofs.Nci.CoveringWitness`), the count's bound is negative.
-/

namespace Nci.SecurityProofs.CoveringTwoCosts

open Verity Finset Nci.Covering

/-- The gates each gate reads. -/
def tcReads : Fin 14 → Finset (Fin 14) :=
  ![∅, ∅, ∅, ∅, {0, 3}, {0, 3}, {1, 3}, {1, 3}, {2, 3}, {4, 5}, {9, 6}, {10, 7}, {11, 8}, {2, 12}]

/-- The witness circuit, one bit a gate. -/
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

/-- The witness declaration: nothing registered. -/
abbrev tcDecl : Registered := .ofCircuit tcCircuit ∅ (by simp)

/-- The product gate of weight `i`. -/
def tcProd : Fin 6 → Fin 14 := ![4, 5, 6, 7, 8, 13]

/-- The weight gate of weight `i`: two weights a gate. -/
def tcWeight : Fin 6 → Fin 14 := ![0, 0, 1, 1, 2, 2]

/-- One multiplication block per weight, at one token, whose computation is every computed gate. -/
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

/-- A subcircuit of the witness, its conditions decided. -/
def tcSub (gs is : Finset (Fin 14))
    (h : is ⊆ gs ∧ (∀ g ∈ gs, g ∉ is → tcCircuit.reads g ⊆ gs) ∧ ∀ g ∈ gs, g ∈ sources tcCircuit → g ∈ is) :
    Subcircuit tcDecl.D :=
  ⟨gs, is, h.1, h.2.1, h.2.2⟩

/-- Ten blocks of three gates, two of them inputs: the five products, the four additions, and the last product. -/
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

/-- Parameter-hard at `c_w = 1/2`, at every `s`: the weights a block completes have their gates among its inputs, at most
two weights a gate. -/
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

/-- `p₅` is chained at `(D, c_a) = (5, 1)`, through `a` by the weights 0–4. -/
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

/-- **The two-cost covering bound is not vacuous at `c_w ≠ c_a`**: one declaration, its multiplication blocks and a
covering satisfy every hypothesis of `CoveringBoundTwoCosts` at `c_w = 1/2`, `c_a = 1`, with a chained multiplication
block. -/
theorem coveringBoundTwoCosts_nonvacuous :
    ∃ (d : Registered) (B : MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
      cw ≠ ca ∧ (k : ℝ) < cw * D ∧ 0 < ca ∧ Assumptions.ParameterHardAt d B s cw ∧ Covers d s k I ∧
      (Assumptions.chainedBlocksAt d B D ca).Nonempty :=
  ⟨tcDecl, tcBlocks, 3, 2, 5, 1 / 2, 1, 10, tcCover, by norm_num, by norm_num, by norm_num, tc_parameterHard 3,
    tc_covers, ⟨(5, 0), tc_mem⟩⟩

open Nci.SecurityProofs.Covering in
/-- **`c_a > 0` is needed**: at `c_a = -1`, `CoveringBound`'s witness meets the other hypotheses (`c_w = 1`, `D = 3`,
`k = 2`, `s = 3`, six blocks), and the count's bound `n · ((k / c_w)(k / c_a) + k) = -12` is below every count. -/
theorem coveringBoundTwoCosts_needs_ca_pos :
    ∃ (d : Registered) (B : MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
      (k : ℝ) < cw * D ∧ Assumptions.ParameterHardAt d B s cw ∧ Covers d s k I ∧
      ¬ ((Assumptions.chainedBlocksAt d B D ca).card : ℝ) ≤ n * (k / cw * (k / ca) + k) := by
  have hhard : Assumptions.ParameterHardAt exDecl exBlocks 3 1 := by
    simpa only [one_mul] using (parameterHard_iff exBlocks 3 1 1).1 ex_parameterHard
  refine ⟨exDecl, exBlocks, 3, 2, 3, 1, -1, 6, exCover, by norm_num, hhard, ex_covers, ?_⟩
  have h0 := (Nat.cast_nonneg (Assumptions.chainedBlocksAt exDecl exBlocks 3 (-1)).card : (0 : ℝ) ≤ _)
  norm_num
  linarith

end Nci.SecurityProofs.CoveringTwoCosts
