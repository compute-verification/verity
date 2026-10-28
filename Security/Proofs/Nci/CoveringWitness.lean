import Definitions.Nci.Guarantees.Covering
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.NormNum

/-!
# The covering bound is not vacuous

One circuit where every hypothesis of `Nci.Guarantees.CoveringBound` holds and `𝓜*` is nonempty
(`coveringBound_nonvacuous`), at `c = r · b = 1`, `D = 3`, `k = 2` and `s = 3`. Weights `w₀ … w₃` (gates 0–3) and a token
input `x` (gate 4) are unregistered inputs (`θ = ∅`). The products `pᵢ = wᵢ · x` (gates 5–7) feed the activation
`a = (p₀ + p₁) + p₂` (gates 8, 9), and `p₃ = w₃ · a` (gate 10) is chained through `a` by the weights 0, 1 and 2. A block
that completes a product holds its weight as an input, so the circuit is parameter-hard at every `s`, and six blocks of
three gates, each with two inputs, cover it.
-/

namespace Nci.SecurityProofs.Covering

open Verity Finset Nci.Covering

/-- The gates each gate reads. -/
def exReads : Fin 11 → Finset (Fin 11) :=
  ![∅, ∅, ∅, ∅, ∅, {0, 4}, {1, 4}, {2, 4}, {5, 6}, {8, 7}, {3, 9}]

/-- The witness circuit, one bit a gate. -/
abbrev exCircuit : Circuit where
  N := 11
  w := fun _ => 1
  isInput := fun g => decide (g.val < 5)
  reads := exReads
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {10}

/-- The witness declaration: nothing registered. -/
abbrev exDecl : Registered := .ofCircuit exCircuit ∅ (by simp)

/-- The product gate of weight `i`. -/
def exProd : Fin 4 → Fin 11 := ![5, 6, 7, 10]

/-- One multiplication block per weight, at one token, whose computation is every computed gate. -/
abbrev exBlocks : MulBlocks exDecl where
  nw := 4
  nt := 1
  P := univ
  m := fun p => {exProd p.1}
  o := fun p => exProd p.1
  T := fun _ => {5, 6, 7, 8, 9, 10}
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
def exSub (gs is : Finset (Fin 11))
    (h : is ⊆ gs ∧ (∀ g ∈ gs, g ∉ is → exCircuit.reads g ⊆ gs) ∧ ∀ g ∈ gs, g ∈ sources exCircuit → g ∈ is) :
    Subcircuit exDecl.D :=
  ⟨gs, is, h.1, h.2.1, h.2.2⟩

/-- Six blocks of three gates, two of them inputs: the products, the two additions, and the last product. -/
def exCover : Fin 6 → Subcircuit exDecl.D :=
  ![exSub {0, 4, 5} {0, 4} (by decide), exSub {1, 4, 6} {1, 4} (by decide), exSub {2, 4, 7} {2, 4} (by decide),
    exSub {5, 6, 8} {5, 6} (by decide), exSub {7, 8, 9} {7, 8} (by decide), exSub {3, 9, 10} {3, 9} (by decide)]

theorem ex_covers : Covers exDecl 3 2 exCover := by
  refine ⟨?_, ?_⟩
  · unfold InferenceBlock
    decide
  · unfold Subcircuit.inner
    decide

/-- The input gate of weight `i`. -/
def exWeight : Fin 4 → Fin 11 := ![0, 1, 2, 3]

theorem ex_parameterHard : Assumptions.ParameterHard exDecl exBlocks 3 1 1 := by
  classical
  intro S _
  have hw : ∀ i : Fin 4, exWeight i ∈ exCircuit.reads (exProd i) ∧ exWeight i ∈ sources exCircuit := by decide
  have hinj : Function.Injective exWeight := by decide
  have hle : (exBlocks.weightsCompleted S).card ≤ S.ins.card := by
    refine card_le_card_of_injOn exWeight (fun i hi => ?_) hinj.injOn
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    have hm := (mem_filter.1 hp).2 (mem_singleton_self _)
    obtain ⟨hgs, hgi⟩ := mem_sdiff.1 hm
    exact S.sources_ins _ (S.closed _ hgs hgi (hw p.1).1) (hw p.1).2
  have hθ : S.ins \ exDecl.θ = S.ins := sdiff_empty
  rw [hθ, one_mul, one_mul]
  exact_mod_cast hle

theorem ex_path (i : Fin 4) (hi : i ∈ ({0, 1, 2} : Finset (Fin 4))) :
    FeedsWithin exDecl.D (exBlocks.T 0) (exBlocks.o (i, 0)) 9 := by
  simp only [mem_insert, mem_singleton] at hi
  rcases hi with rfl | rfl | rfl
  · exact .step (h := 8) (by decide) (by decide) (.step (h := 9) (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 8) (by decide) (by decide) (.step (h := 9) (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 9) (by decide) (by decide) (.refl (by decide))

theorem ex_chained : Assumptions.ActivationChained exDecl exBlocks 3 1 1 (3, 0) := by
  refine ⟨mem_univ _, {9}, {0, 1, 2}, by decide, by decide, by decide,
    fun i hi => ⟨mem_univ _, by revert i; decide, 9, mem_singleton_self _, ex_path i hi⟩, ?_⟩
  intro Q hQ X _ hcut
  rcases X.eq_empty_or_nonempty with rfl | hX
  · have hWQ : 0 < (({0, 1, 2} : Finset (Fin 4)) \ Q).card := by
      have := le_card_sdiff Q ({0, 1, 2} : Finset (Fin 4))
      have h3 : ({0, 1, 2} : Finset (Fin 4)).card = 3 := by decide
      omega
    obtain ⟨i, hi⟩ := card_pos.1 hWQ
    rcases hcut i hi with ⟨x, hx⟩ | hno
    · simp at hx
    · exact absurd (by simpa using ex_path i (mem_sdiff.1 hi).1) (hno 9 (mem_singleton_self _))
  · rw [one_mul]
    exact_mod_cast card_pos.2 hX

theorem ex_mem : ((3, 0) : Fin 4 × Fin 1) ∈ Assumptions.chainedBlocks exDecl exBlocks 3 1 1 := by
  classical
  simp only [Assumptions.chainedBlocks, mem_filter]
  exact ⟨mem_univ _, ex_chained⟩

/-- **The covering bound is not vacuous**: one declaration, its multiplication blocks and a covering satisfy every
hypothesis of `CoveringBound`, with a chained multiplication block. -/
theorem coveringBound_nonvacuous :
    ∃ (d : Registered) (B : MulBlocks d) (s k D : ℕ) (r b : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
      (k : ℝ) < r * b * D ∧ Assumptions.ParameterHard d B s r b ∧ Covers d s k I ∧
      (Assumptions.chainedBlocks d B D r b).Nonempty :=
  ⟨exDecl, exBlocks, 3, 2, 3, 1, 1, 6, exCover, by norm_num, ex_parameterHard, ex_covers,
    ⟨(3, 0), ex_mem⟩⟩

end Nci.SecurityProofs.Covering
