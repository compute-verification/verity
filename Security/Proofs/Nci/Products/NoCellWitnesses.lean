import Proofs.Nci.Products.NoCell
import Proofs.Nci.Products.Witnesses
import Definitions.Nci.Guarantees.Products.NoCellWitnesses
import Mathlib.Tactic.NormNum

/-!
# Products of two fresh operands without the word level: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Products/NoCellWitnesses.lean`, at `Pair`
(`Security/Definitions/Nci/Products/Instances.lean`) and its one block `P1`
(`Security/Definitions/Nci/Products/NoCellInstances.lean`). The partial sums are received with the start gate 4 at any
`(c, k)`: a block holding gate 4 holds the start, and one holding gate 5 without it reads gate 4, a gate of the entry
beyond `θ` (`gates_cases`, then `decide`). Each bound is compared with its price in `ℝ` or `ℝ≥0∞`. The `example`s at the
end apply each guarantee to its lemmas, so the lemmas are the premises as the guarantee states them.
-/

namespace Nci.Products.SecurityProofs

open Finset
open Nci.Formalism Nci.Products
open scoped ENNReal

namespace Pair

/-- **Partial sums received with the operands fresh**, at any `(c, k)`: the start is gate 4; a block holding gate 4
holds it, and one holding gate 5 alone reads gate 4. -/
theorem received_fresh (c k : ℕ) : Pair.freshProduct.Received c k 1 :=
  ⟨fun _ _ => {4}, fun _ _ => (card_singleton _).le, fun I hI => by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide⟩

/-- **Partial sums received with the operands registered**, as with them fresh: gate 4 is computed, so outside `θ`. -/
theorem received_registered (c k : ℕ) : Pair.registeredProduct.Received c k 1 :=
  ⟨fun _ _ => {4}, fun _ _ => (card_singleton _).le, fun I hI => by
    rcases gates_cases hI.1 with rfl | rfl | rfl | rfl <;> decide⟩

/-- **Widths as passed fail with the operands registered**, at any `c ≥ 1` and `k`: gate 4's block reads `a₀` free. -/
theorem not_passed_registered_of_one_le {c : ℕ} (hc : 1 ≤ c) (k : ℕ) :
    ¬ Pair.registeredProduct.Passed c k 1 1 1 := fun h => by
  have h0 : (Pair.registered.unregistered {4}).card = 0 := by decide
  have hI : InferenceBlock Pair.registered c k {4} :=
    ⟨(by decide : ({4} : Finset (Fin Pair.C.N)) ⊆ Pair.C.computed), by rw [card_singleton]; exact hc,
      by rw [h0]; exact Nat.zero_le k⟩
  exact absurd (h.operandA {4} hI 0 0 (by decide)) (by decide)

theorem isCK_P1 : Pair.P1.IsCK Pair.registered 2 1 := ⟨boolean, by decide, by decide⟩

theorem computed_card : Pair.registered.D.computed.card = 2 := by decide

theorem not_noCellCap :
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (1 : ℕ) * (((1 : ℕ) : ℝ) / 3 * √(((1 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (1 : ℕ)) +
        ((1 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((1 : ℕ) : ℝ) * (1 * 1 : ℕ)) := by
  have h3 : √(((1 : ℕ) : ℝ) / 3) ≤ 1 := by
    rw [Real.sqrt_le_left zero_le_one]
    norm_num
  have e1 : √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) = 1 := by norm_num
  have e2 : √(((1 : ℕ) : ℝ) * (1 : ℕ)) = 1 := by norm_num
  rw [e1, e2]
  intro h
  push_cast at h h3
  linarith

/-- Two products against a block of `Γ < 3/2` and `B = 1/2` need more than one block. -/
theorem one_lt_chainBlocks {Γ : ℝ} (hΓ0 : 0 ≤ Γ) (hΓ : Γ < 3 / 2) : 1 < chainBlocks 2 Γ (1 / 2) := by
  unfold chainBlocks
  have hR0 : 0 ≤ √((1 / 2 : ℝ) ^ 2 + 4 * Γ * 2) := Real.sqrt_nonneg _
  have hR : √((1 / 2 : ℝ) ^ 2 + 4 * Γ * 2) < 7 / 2 := by
    rw [Real.sqrt_lt' (by norm_num)]
    nlinarith
  have h : 1 < 2 * 2 / (√((1 / 2 : ℝ) ^ 2 + 4 * Γ * 2) + 1 / 2) := by
    rw [lt_div_iff₀ (by positivity)]
    linarith
  exact one_lt_pow₀ h two_ne_zero

theorem not_noCellBound_le : ¬ noCellBound Pair.registered 1 2 1 2 1 1 1 1 ≤ partitionSlowdown Pair.registered 1 2 := by
  have h3 : √(((1 : ℕ) : ℝ) / 3) ≤ 1 := by
    rw [Real.sqrt_le_left zero_le_one]
    norm_num
  have h30 : 0 ≤ √(((1 : ℕ) : ℝ) / 3) := Real.sqrt_nonneg _
  have eN : ((1 * 2 * 1 : ℕ) : ℝ) = 2 := by norm_num
  have e1 : √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) = 1 := by norm_num
  have eB : ((1 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √((1 * 1 : ℕ) : ℝ) = 1 / 2 := by norm_num
  have hcb := one_lt_chainBlocks (Γ := ((1 : ℕ) : ℝ) / 3 * √(((1 : ℕ) : ℝ) / 3) / 1 + (1 : ℕ))
    (by push_cast; positivity) (by push_cast at h3 ⊢; linarith)
  unfold noCellBound partitionSlowdown
  rw [computed_card, eN, e1, eB, ENNReal.mul_div_cancel_right (by norm_num) (by norm_num),
    ENNReal.div_self (by norm_num) (by norm_num), not_le, ENNReal.one_lt_ofReal]
  exact hcb

theorem not_partitionBound :
    ¬ PartitionBound Pair.registered 2 1 (noCellBound Pair.registered 1 2 1 2 1 1 1 1) :=
  fun h => not_noCellBound_le (h 1 Pair.P1 isCK_P1)

end Pair

theorem NoCellCap_honest : Guarantees.NoCellCap_honest :=
  ⟨Pair.algorithm_fresh, Pair.passed_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos⟩

theorem NoCellCap_nontrivial : Guarantees.NoCellCap_nontrivial :=
  ⟨Pair.algorithm_registered, Pair.not_passed_registered_of_one_le (by norm_num) 1, Pair.received_registered 2 1,
    Pair.isCK_P1, Nat.one_pos, Pair.not_noCellCap⟩

theorem NoCellBound_honest : Guarantees.NoCellBound_honest :=
  ⟨Pair.algorithm_fresh, Pair.passed_fresh, Pair.received_fresh 1 3, Pair.isCK_fresh, Nat.one_pos⟩

theorem NoCellBound_nontrivial : Guarantees.NoCellBound_nontrivial :=
  ⟨Pair.algorithm_registered, Pair.not_passed_registered_of_one_le (by norm_num) 1, Pair.received_registered 2 1,
    Nat.one_pos, Pair.not_partitionBound⟩

example : ((1 * 2 * 1 : ℕ) : ℝ) ≤
    (2 : ℕ) * (((3 : ℕ) : ℝ) / 3 * √(((3 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (3 : ℕ)) +
      ((3 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ)) :=
  let h := NoCellCap_honest
  NoCellCap _ _ _ _ _ 1 3 1 1 1 2 Pair.P h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2 h.2.2.2.2 h.2.2.2.2

example (h : Pair.registeredProduct.Passed 2 1 1 1 1) : False :=
  let n := NoCellCap_nontrivial
  n.2.2.2.2.2 (NoCellCap _ _ _ _ _ 2 1 1 1 1 1 Pair.P1 n.1 h n.2.2.1 n.2.2.2.1 n.2.2.2.2.1 n.2.2.2.2.1
    n.2.2.2.2.1)

example : noCellBound Pair.fresh 1 2 1 1 3 1 1 1 ≤ partitionSlowdown Pair.fresh 2 1 :=
  let h := NoCellBound_honest
  NoCellBound _ _ _ _ _ 1 3 1 1 1 h.1 h.2.1 h.2.2.1 h.2.2.2.2 h.2.2.2.2 h.2.2.2.2 2 Pair.P h.2.2.2.1

example (h : Pair.registeredProduct.Passed 2 1 1 1 1) : False :=
  let n := NoCellBound_nontrivial
  n.2.2.2.2 (NoCellBound _ _ _ _ _ 2 1 1 1 1 n.1 h n.2.2.1 n.2.2.2.1 n.2.2.2.1 n.2.2.2.1)

end Nci.Products.SecurityProofs
