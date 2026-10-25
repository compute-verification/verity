import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# A1, the compression lemma: `PousProofs.a1_compression_card` and `a1_compression_pr`, as drafted

`enc` is injective on `E` into the bit lists of length at most `L`, of which there are `Σ_{m ≤ L} 2^m < 2^(L+1)`.
The probability form divides by `|α|`. This is the converse of `exists_code` in `thm1ii` and `thm1-family`
(a code from a cardinality bound); neither implies the other, so there is nothing to reuse here.
-/

namespace PousProofs

open Pous
open scoped ENNReal

/-- A1. Distinct descriptions of at most `L` bits: fewer than `2^(L+1)` outcomes. -/
theorem a1_compression_card {α : Type} [Fintype α] (E : α → Prop) [DecidablePred E] (L : ℕ)
    (enc : α → List Bool) (hlen : ∀ x, E x → (enc x).length ≤ L)
    (hinj : ∀ x y, E x → E y → enc x = enc y → x = y) :
    (Finset.univ.filter E).card < 2 ^ (L + 1) := by
  classical
  -- the bit lists of length at most `L`
  let T : Finset (List Bool) := (Finset.range (L + 1)).biUnion fun m =>
    (Finset.univ : Finset (Fin m → Bool)).image List.ofFn
  have hmaps : Set.MapsTo enc (Finset.univ.filter E : Set α) (T : Set (List Bool)) := by
    intro x hx
    rw [Finset.mem_coe, Finset.mem_filter] at hx
    rw [Finset.mem_coe, Finset.mem_biUnion]
    refine ⟨(enc x).length, Finset.mem_range.2 (Nat.lt_succ_of_le (hlen x hx.2)), ?_⟩
    rw [Finset.mem_image]
    exact ⟨(enc x).get, Finset.mem_univ _, List.ofFn_get _⟩
  have h1 : (Finset.univ.filter E).card ≤ T.card :=
    Finset.card_le_card_of_injOn enc hmaps fun x hx y hy hxy =>
      hinj x y (Finset.mem_filter.1 hx).2 (Finset.mem_filter.1 hy).2 hxy
  have h2 : T.card ≤ ∑ m ∈ Finset.range (L + 1), 2 ^ m := by
    refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun m _ => Finset.card_image_le.trans ?_)
    simp
  have h3 : ∀ n, ∑ m ∈ Finset.range n, 2 ^ m < 2 ^ n := by
    intro n
    induction n with
    | zero => simp
    | succ n ih => rw [Finset.sum_range_succ, pow_succ]; omega
  exact h1.trans_lt (h2.trans_lt (h3 _))

/-- A1, probability form. -/
theorem a1_compression_pr {α : Type} [Fintype α] (E : α → Prop) (L : ℕ) (enc : α → List Bool)
    (hlen : ∀ x, E x → (enc x).length ≤ L) (hinj : ∀ x y, E x → E y → enc x = enc y → x = y) :
    pr E ≤ 2 ^ (L + 1) / Fintype.card α := by
  classical
  rw [pr_def]
  gcongr
  exact_mod_cast (a1_compression_card E L enc hlen hinj).le

end PousProofs
