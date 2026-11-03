import Proofs.Nci.Products.Cap
import Definitions.Nci.Guarantees.Products.Witnesses
import Mathlib.Tactic.NormNum

/-!
# Products of two fresh operands: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Products/Witnesses.lean`, at `Single` and `Pair`
(`Security/Definitions/Nci/Products/Instances.lean`). Their gates, reads and cones are small finite sets, which
`decide` computes. Every inference block of `Single` is empty or gate 2, and of `Pair` one of the four sets of gates 4
and 5 (`gates_cases`), so widths as passed is a few counts at each: with the operands fresh a block reads every operand
bit its MACs read, and with them registered none beyond `θ`. Each bound is compared with its price in `ℝ` or `ℝ≥0∞`.
The `example`s at the end apply each guarantee to its lemmas, so the lemmas are the premises as the guarantee states
them.
-/

namespace Nci.Products.SecurityProofs

open Finset
open Verity (Declaration)
open Nci.Formalism Nci.Products Nci.Products.Single
open scoped ENNReal

namespace Single

/-- The product's one MAC, gate 2, is connected, whatever is registered: a set holding some of it holds all of it. -/
theorem connected (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) (i : Fin 1) (l : Fin 1) (j : Fin 1)
    (X : Finset (Fin C.N)) (hX : X ⊆ (product θ hθ).mac i l j) (hne : X.Nonempty)
    (hX' : X ≠ (product θ hθ).mac i l j) :
    ∃ h ∈ (product θ hθ).mac i l j, ∃ g ∈ (product θ hθ).mac i l j, h ∈ C.reads g ∧ (h ∈ X ↔ g ∉ X) := by
  rcases subset_singleton_iff.1 hX with rfl | rfl
  · exact absurd hne not_nonempty_empty
  · exact absurd rfl hX'

/-- **The product is by the standard algorithm** with its operands fresh: its one MAC is gate 2, which reads both
operands' bits, and the cones `{0}` and `{1}` are disjoint from each other and from it. -/
theorem algorithm_fresh : freshProduct.Algorithm :=
  ⟨by decide, by decide, by decide, connected _ _, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- **The product is by the standard algorithm** with its operands registered, as with them fresh. -/
theorem algorithm_registered : registeredProduct.Algorithm :=
  ⟨by decide, by decide, by decide, connected _ _, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- Every block of computed gates is empty or gate 2. -/
theorem gates_cases {I : Finset (Fin C.N)} (hI : I ⊆ C.computed) : I = ∅ ∨ I = {2} :=
  subset_singleton_iff.1 (hI.trans (by decide : C.computed ⊆ {2}))

/-- **Widths as passed hold with the operands fresh**: gate 2's block reads both operands' bits beyond `θ`, and holds
`C`'s final bit. -/
theorem passed_fresh : freshProduct.Passed 1 2 1 1 1 where
  operandA I hI := by
    rcases gates_cases hI.1 with rfl | rfl <;> decide
  operandB I hI := by
    rcases gates_cases hI.1 with rfl | rfl <;> decide
  partialSum I hI := by
    rcases gates_cases hI.1 with rfl | rfl <;> decide

/-- **Widths as passed fail with the operands registered**, at any `k`: gate 2's block reads both operands free, none
of `A`'s cone beyond `θ`. -/
theorem not_passed_registered (k : ℕ) : ¬ registeredProduct.Passed 1 k 1 1 1 := fun h => by
  have h0 : (registered.unregistered {2}).card = 0 := by decide
  have hI : InferenceBlock registered 1 k {2} :=
    ⟨(by decide : ({2} : Finset (Fin C.N)) ⊆ C.computed), by decide, by rw [h0]; exact Nat.zero_le k⟩
  exact absurd (h.operandA {2} hI 0 0 (by decide)) (by decide)

theorem boolean : C.Boolean := ⟨fun _ => rfl, by decide⟩

theorem isCK_fresh : P.IsCK fresh 1 2 := ⟨boolean, by decide, by decide⟩

theorem isCK_registered (k : ℕ) : P.IsCK registered 1 k :=
  ⟨boolean, by decide, fun u => by
    rw [show registered.unregistered (P.gates u) = ∅ by revert u; decide, card_empty]
    exact Nat.zero_le k⟩

theorem gates_zero : P.gates 0 = {2} := by decide

theorem unsplit (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) : (product θ hθ).Unsplit P :=
  fun _ _ _ => ⟨0, by rw [gates_zero]⟩

theorem outBits_eq : registeredProduct.outBits = 1 := by decide

theorem computed_card : registered.D.computed.card = 1 := by decide

theorem capAt_zero : capAt 0 1 1 1 (1 / 2) = 0 := by
  unfold capAt
  norm_num

theorem not_cap :
    ¬ ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * capAt 0 1 1 1 (1 / 2) + 1 / 2 * registeredProduct.outBits := by
  rw [capAt_zero, outBits_eq]
  norm_num

theorem not_wordCap :
    ¬ ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * ((1 : ℕ) * √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)))) +
      √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) * registeredProduct.outBits := by
  rw [outBits_eq]
  norm_num

theorem not_bound_le : ¬ registeredProduct.bound 1 0 1 1 1 (1 / 2) ≤ partitionSlowdown registered 1 1 := by
  have hpos : ENNReal.ofReal (((1 * 1 * 1 : ℕ) : ℝ) - 1 / 2 * ((1 : ℕ) : ℝ)) / ((1 : ℕ) : ℝ≥0∞) ≠ 0 := by
    norm_num [ENNReal.ofReal_eq_zero]
  rw [Standard.bound, partitionSlowdown, capAt_zero, outBits_eq, computed_card, ENNReal.ofReal_zero, ENNReal.zero_div,
    ENNReal.div_zero hpos]
  norm_num

theorem not_partitionBound : ¬ PartitionBound registered 1 0 (registeredProduct.bound 1 0 1 1 1 (1 / 2)) :=
  fun h => not_bound_le (h 1 P (isCK_registered 0))

end Single

namespace Pair

/-- Each MAC of `Pair` is one gate, so connected, whatever is registered. -/
theorem connected (θ : Finset (Fin Pair.C.N)) (hθ : ∀ g ∈ θ, Pair.C.isInput g = true) (i : Fin 1) (l : Fin 2)
    (j : Fin 1) (X : Finset (Fin Pair.C.N)) (hX : X ⊆ (Pair.product θ hθ).mac i l j) (hne : X.Nonempty)
    (hX' : X ≠ (Pair.product θ hθ).mac i l j) :
    ∃ h ∈ (Pair.product θ hθ).mac i l j, ∃ g ∈ (Pair.product θ hθ).mac i l j, h ∈ Pair.C.reads g ∧
      (h ∈ X ↔ g ∉ X) := by
  obtain ⟨a, ha⟩ : ∃ a, (Pair.product θ hθ).mac i l j = {a} := by
    fin_cases l
    · exact ⟨4, rfl⟩
    · exact ⟨5, rfl⟩
  rw [ha] at hX hX'
  rcases subset_singleton_iff.1 hX with rfl | rfl
  · exact absurd hne not_nonempty_empty
  · exact absurd rfl hX'

/-- **The product is by the standard algorithm** with its operands fresh: its MACs are gates 4 and 5, each reading
its operands' bits, and gate 5 reads gate 4. -/
theorem algorithm_fresh : Pair.freshProduct.Algorithm :=
  ⟨by decide, by decide, by decide, connected _ _, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- **The product is by the standard algorithm** with its operands registered, as with them fresh. -/
theorem algorithm_registered : Pair.registeredProduct.Algorithm :=
  ⟨by decide, by decide, by decide, connected _ _, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- Every block of computed gates is one of four. -/
theorem gates_cases {I : Finset (Fin Pair.C.N)} (hI : I ⊆ Pair.C.computed) :
    I = ∅ ∨ I = {4} ∨ I = {5} ∨ I = {4, 5} := by
  have h : I ∈ Pair.C.computed.powerset := mem_powerset.2 hI
  rw [show Pair.C.computed.powerset = {∅, {4}, {5}, {4, 5}} by decide] at h
  simpa using h

/-- **Widths as passed hold with the operands fresh**: a block that holds gate 4 reads `a₀` and `b₀` and hands its
partial sum to gate 5, and one that holds gate 5 reads `a₁` and `b₁` and holds `C`'s final bit. -/
theorem passed_fresh : Pair.freshProduct.Passed 1 3 1 1 1 where
  operandA I hI := by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide
  operandB I hI := by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide
  partialSum I hI := by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide

/-- **Widths as passed fail with the operands registered**, at any `k`: gate 4's block reads `a₀` free. -/
theorem not_passed_registered (k : ℕ) : ¬ Pair.registeredProduct.Passed 1 k 1 1 1 := fun h => by
  have h0 : (Pair.registered.unregistered {4}).card = 0 := by decide
  have hI : InferenceBlock Pair.registered 1 k {4} :=
    ⟨(by decide : ({4} : Finset (Fin Pair.C.N)) ⊆ Pair.C.computed), by decide, by rw [h0]; exact Nat.zero_le k⟩
  exact absurd (h.operandA {4} hI 0 0 (by decide)) (by decide)

/-- **The partial sum runs along the chain**: gate 5 reads gate 4, `b_C = 1` bit. -/
theorem chained_fresh : Pair.freshProduct.Chained 1 := by
  unfold Standard.Chained
  decide

theorem chained_registered : Pair.registeredProduct.Chained 1 := by
  unfold Standard.Chained
  decide

theorem boolean : Pair.C.Boolean := ⟨fun _ => rfl, by decide⟩

theorem isCK_fresh : Pair.P.IsCK Pair.fresh 1 3 := ⟨boolean, by decide, by decide⟩

theorem isCK_registered : Pair.P.IsCK Pair.registered 1 1 := ⟨boolean, by decide, by decide⟩

theorem unsplit_fresh : Pair.freshProduct.Unsplit Pair.P := by
  unfold Standard.Unsplit
  decide

theorem unsplit_registered : Pair.registeredProduct.Unsplit Pair.P := by
  unfold Standard.Unsplit
  decide

theorem not_chainCap :
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (2 : ℕ) * (((1 : ℕ) : ℝ) / 3 * √(((1 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) +
        ((1 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ)) := by
  have h3 : √(((1 : ℕ) : ℝ) / 3) ≤ 1 := by
    rw [Real.sqrt_le_left zero_le_one]
    norm_num
  have h2 : √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ)) ≤ 3 / 2 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  have e1 : √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) = 1 := by norm_num
  have e2 : √(((1 : ℕ) : ℝ) * (1 : ℕ)) = 1 := by norm_num
  rw [e1, e2]
  intro h
  push_cast at h h2 h3
  linarith

end Pair

theorem Cap_honest : Guarantees.Cap_honest :=
  ⟨Single.algorithm_fresh, Single.passed_fresh, Single.isCK_fresh, Nat.one_pos, by norm_num⟩

theorem Cap_nontrivial : Guarantees.Cap_nontrivial :=
  ⟨Single.algorithm_registered, Single.not_passed_registered 0, Single.isCK_registered 0, Nat.one_pos, by norm_num,
    Single.not_cap⟩

theorem WordCap_honest : Guarantees.WordCap_honest :=
  ⟨Single.algorithm_fresh, Single.passed_fresh, Single.isCK_fresh, Single.unsplit _ _, by norm_num, Nat.one_pos⟩

theorem WordCap_nontrivial : Guarantees.WordCap_nontrivial :=
  ⟨Single.algorithm_registered, Single.not_passed_registered 1, Single.isCK_registered 1, Single.unsplit _ _,
    Nat.one_pos, Single.not_wordCap⟩

theorem ChainCap_honest : Guarantees.ChainCap_honest :=
  ⟨Pair.algorithm_fresh, Pair.passed_fresh, Pair.chained_fresh, Pair.isCK_fresh, Pair.unsplit_fresh, Nat.one_pos⟩

theorem ChainCap_nontrivial : Guarantees.ChainCap_nontrivial :=
  ⟨Pair.algorithm_registered, Pair.not_passed_registered 1, Pair.chained_registered, Pair.isCK_registered,
    Pair.unsplit_registered, Nat.one_pos, Pair.not_chainCap⟩

theorem Bound_honest : Guarantees.Bound_honest :=
  ⟨Single.algorithm_fresh, Single.passed_fresh, Single.isCK_fresh, Nat.one_pos, by norm_num⟩

theorem Bound_nontrivial : Guarantees.Bound_nontrivial :=
  ⟨Single.algorithm_registered, Single.not_passed_registered 0, Nat.one_pos, by norm_num,
    Single.not_partitionBound⟩

example : ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * capAt 2 1 1 1 (1 / 2) + 1 / 2 * freshProduct.outBits :=
  let h := Cap_honest
  Cap _ _ _ _ _ 1 2 1 1 1 1 P (1 / 2) h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.1 h.2.2.2.1 h.2.2.2.2

example (h : registeredProduct.Passed 1 0 1 1 1) : False :=
  let n := Cap_nontrivial
  n.2.2.2.2.2 (Cap _ _ _ _ _ 1 0 1 1 1 1 P (1 / 2) n.1 h n.2.2.1 n.2.2.2.1 n.2.2.2.1 n.2.2.2.1 n.2.2.2.2.1)

example : ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * ((2 : ℕ) * √((2 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)))) +
    √((2 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) * freshProduct.outBits :=
  let h := WordCap_honest
  WordCap _ _ _ _ _ 1 2 1 1 1 1 P h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2 h.2.2.2.2.2 h.2.2.2.2.2

example (h : registeredProduct.Passed 1 1 1 1 1) : False :=
  let n := WordCap_nontrivial
  n.2.2.2.2.2 (WordCap _ _ _ _ _ 1 1 1 1 1 1 P n.1 h n.2.2.1 n.2.2.2.1 n.2.2.2.2.1 n.2.2.2.2.1 n.2.2.2.2.1
    n.2.2.2.2.1)

example : ((1 * 2 * 1 : ℕ) : ℝ) ≤
    (2 : ℕ) * (((3 : ℕ) : ℝ) / 3 * √(((3 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) +
      ((3 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ)) :=
  let h := ChainCap_honest
  ChainCap _ _ _ _ _ 1 3 1 1 1 2 Pair.P h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2 h.2.2.2.2.2 h.2.2.2.2.2

example (h : Pair.registeredProduct.Passed 1 1 1 1 1) : False :=
  let n := ChainCap_nontrivial
  n.2.2.2.2.2.2 (ChainCap _ _ _ _ _ 1 1 1 1 1 2 Pair.P n.1 h n.2.2.1 n.2.2.2.1 n.2.2.2.2.1 n.2.2.2.2.2.1
    n.2.2.2.2.2.1 n.2.2.2.2.2.1)

example : freshProduct.bound 1 2 1 1 1 (1 / 2) ≤ partitionSlowdown fresh 1 1 :=
  let h := Bound_honest
  Bound _ _ _ _ _ 1 2 1 1 1 (1 / 2) h.1 h.2.1 h.2.2.2.1 h.2.2.2.1 h.2.2.2.1 h.2.2.2.2 1 P h.2.2.1

example (h : registeredProduct.Passed 1 0 1 1 1) : False :=
  let n := Bound_nontrivial
  n.2.2.2.2 (Bound _ _ _ _ _ 1 0 1 1 1 (1 / 2) n.1 h n.2.2.1 n.2.2.1 n.2.2.1 n.2.2.2.1)

end Nci.Products.SecurityProofs
