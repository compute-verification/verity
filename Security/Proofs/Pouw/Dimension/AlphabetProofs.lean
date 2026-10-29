import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Int.Interval
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Products of two drawn digits: how often a value fixed before the draw is hit (proofs)

A noise whose entry `(i, j)` is the product `X i · W j` of a row digit and a column digit, both drawn after the data, is
left unchanged at that entry by a data lane `−v` fixed before the draw exactly when `X i · W j = v`. Its `p*` is the
largest share of the digit pairs whose product is one value.

* **Per entry** (`entry_card`): with rows and columns drawn independently and uniformly, the draws on which entry
  `(i, j)` hits `v` are a `|fiber v|/(|A||B|)` share of all draws, however the rows and columns share their digits.
* **Over the matrix** (`sum_hits_le`, `sum_cost_ge`): if no value is the product of more than `P` pairs, the expected
  number of entries hitting their targets is at most `p*·r·s` with `p* = P/(|A||B|)`, so a cost that pays `c` for each
  other entry averages at least `(1 − p*)·c·r·s`. That premise (`hcost`) is what a floor has to supply; nothing here
  proves it.
* **`uw6`** (`uw6_fiber_le`, `uw6_fiber_120`): over `±1..±32` on both factors no value is the product of more than 20
  of the 4,096 pairs, and 120 is the product of exactly 20, so `p* = 5/1024`. The count reduces to the divisors of
  `|v|` in `[1, 32]` with a cofactor in `[1, 32]` (`fiber_card_le_divs`), at most 10 for every `|v|` (`divs_card_le`,
  by `decide` up to 1,024; none past it).

The targets are fixed before the draw. A target chosen after it is a branch on the coins, which the adaptive bounds
treat (`AdaptiveProofs`).
-/

namespace Pouw.Dimension.Alphabet

open Finset

/-- The signed alphabet `±1, …, ±b`. -/
def signed (b : ℕ) : Finset ℤ := (Icc (-(b : ℤ)) b).erase 0

/-- The pairs of `A × B` whose product is `v`. -/
def fiber (A B : Finset ℤ) (v : ℤ) : Finset (ℤ × ℤ) := (A ×ˢ B).filter fun p => p.1 * p.2 = v

/-- The magnitudes `a ∈ [1, b]` that divide `n` with a cofactor in `[1, b]`. -/
def divs (b n : ℕ) : Finset ℕ := (range (b + 1)).filter fun a => 0 < a ∧ a ∣ n ∧ 0 < n / a ∧ n / a ≤ b

theorem mem_signed {b : ℕ} {x : ℤ} : x ∈ signed b ↔ x ≠ 0 ∧ x.natAbs ≤ b := by
  simp only [signed, mem_erase, mem_Icc]
  omega

theorem card_signed (b : ℕ) : (signed b).card = 2 * b := by
  rw [signed, card_erase_of_mem (by simp), Int.card_Icc]
  omega

/-- Over `±1..±b` a product's value fixes the first factor up to sign among the divisors of `|v|`, and then the second
factor. -/
theorem fiber_card_le_divs (b : ℕ) (v : ℤ) : (fiber (signed b) (signed b) v).card ≤ 2 * (divs b v.natAbs).card := by
  have hmaps : ∀ p ∈ fiber (signed b) (signed b) v,
      (p.1.natAbs, decide (0 < p.1)) ∈ divs b v.natAbs ×ˢ (univ : Finset Bool) := by
    rintro ⟨x, w⟩ hp
    simp only [fiber, mem_filter, mem_product] at hp
    obtain ⟨⟨hx, hw⟩, hv⟩ := hp
    rw [mem_signed] at hx hw
    have hxa : 0 < x.natAbs := Int.natAbs_pos.mpr hx.1
    have hvn : v.natAbs = x.natAbs * w.natAbs := by rw [← hv, Int.natAbs_mul]
    have hq : v.natAbs / x.natAbs = w.natAbs := by rw [hvn, Nat.mul_div_cancel_left _ hxa]
    simp only [mem_product, mem_univ, and_true, divs, mem_filter, mem_range]
    refine ⟨by omega, hxa, ⟨w.natAbs, hvn⟩, ?_, ?_⟩
    · rw [hq]; exact Int.natAbs_pos.mpr hw.1
    · rw [hq]; exact hw.2
  have hinj : Set.InjOn (fun p : ℤ × ℤ => (p.1.natAbs, decide (0 < p.1)))
      (fiber (signed b) (signed b) v : Set (ℤ × ℤ)) := by
    rintro ⟨x, w⟩ hp ⟨x', w'⟩ hp' he
    simp only [fiber, coe_filter, mem_product, Set.mem_ofPred_eq] at hp hp'
    simp only [Prod.mk.injEq, decide_eq_decide] at he
    have hx0 := (mem_signed.mp hp.1.1).1
    have hxx : x = x' := by omega
    subst hxx
    exact Prod.ext rfl (mul_left_cancel₀ hx0 (hp.2.trans hp'.2.symm))
  calc (fiber (signed b) (signed b) v).card ≤ (divs b v.natAbs ×ˢ (univ : Finset Bool)).card :=
        card_le_card_of_injOn _ (fun p hp => hmaps p hp) hinj
    _ = 2 * (divs b v.natAbs).card := by rw [card_product, card_univ, Fintype.card_bool, mul_comm]

set_option maxRecDepth 100000 in
theorem divs_small : ∀ n < 1025, (divs 32 n).card ≤ 10 := by decide

theorem divs_card_le (n : ℕ) : (divs 32 n).card ≤ 10 := by
  by_cases hn : n < 1025
  · exact divs_small n hn
  · rw [card_eq_zero.mpr]
    · exact Nat.zero_le _
    · refine eq_empty_of_forall_notMem fun a ha => ?_
      simp only [divs, mem_filter, mem_range] at ha
      obtain ⟨ha32, -, hdvd, -, hq⟩ := ha
      have : n ≤ 32 * 32 := by
        calc n = a * (n / a) := (Nat.mul_div_cancel' hdvd).symm
          _ ≤ 32 * 32 := Nat.mul_le_mul (by omega) hq
      omega

/-- **`uw6`'s `p*`**: with both factors drawn from `±1..±32`, no value is the product of more than 20 of the 4,096
pairs, so `Pr[x·w = v] ≤ 20/4096 = 5/1024` for every `v`. -/
theorem uw6_fiber_le (v : ℤ) : (fiber (signed 32) (signed 32) v).card ≤ 20 :=
  (fiber_card_le_divs 32 v).trans (by have := divs_card_le v.natAbs; omega)

/-- The bound is attained: 120 is the product of 20 pairs (`±(4,30)`, `±(5,24)`, …, `±(30,4)`). -/
theorem uw6_fiber_120 : 20 ≤ (fiber (signed 32) (signed 32) 120).card := by
  let L : Finset (ℤ × ℤ) := {(4, 30), (5, 24), (6, 20), (8, 15), (10, 12), (12, 10), (15, 8), (20, 6), (24, 5), (30, 4),
    (-4, -30), (-5, -24), (-6, -20), (-8, -15), (-10, -12), (-12, -10), (-15, -8), (-20, -6), (-24, -5), (-30, -4)}
  have hL : L.card = 20 := by decide
  rw [← hL]
  refine card_le_card fun p hp => ?_
  simp only [fiber, mem_filter, mem_product, mem_signed]
  simp only [L, mem_insert, mem_singleton] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl <;> decide

variable {r s : ℕ}

/-- Row digits `X : Fin r → A` and column digits `W : Fin s → B`, drawn independently and uniformly. -/
abbrev draws (A B : Finset ℤ) (r s : ℕ) : Finset ((Fin r → ℤ) × (Fin s → ℤ)) :=
  Fintype.piFinset (fun _ => A) ×ˢ Fintype.piFinset (fun _ => B)

/-- The entries whose product `X i · W j` equals their target. -/
def hits (v : Fin r → Fin s → ℤ) (D : (Fin r → ℤ) × (Fin s → ℤ)) : ℕ :=
  (univ.filter fun e : Fin r × Fin s => D.1 e.1 * D.2 e.2 = v e.1 e.2).card

/-- **One entry**: the draws on which `X i · W j = v` number `|fiber v|·|A|^(r−1)·|B|^(s−1)`, so the entry hits `v`
with probability `|fiber v|/(|A||B|)` whatever the other rows and columns draw. -/
theorem entry_card (A B : Finset ℤ) (i : Fin r) (j : Fin s) (v : ℤ) :
    ((draws A B r s).filter fun D => D.1 i * D.2 j = v).card =
      (fiber A B v).card * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
  have hmaps : ∀ D ∈ (draws A B r s).filter (fun D => D.1 i * D.2 j = v), (D.1 i, D.2 j) ∈ fiber A B v := by
    intro D hD
    simp only [draws, mem_filter, mem_product, Fintype.mem_piFinset] at hD
    simp only [fiber, mem_filter, mem_product]
    exact ⟨⟨hD.1.1 i, hD.1.2 j⟩, hD.2⟩
  rw [card_eq_sum_card_fiberwise hmaps, ← Nat.cast_id ((fiber A B v).card), ← nsmul_eq_mul, ← sum_const]
  refine sum_congr rfl fun p hp => ?_
  simp only [fiber, mem_filter, mem_product] at hp
  have : ((draws A B r s).filter fun D => D.1 i * D.2 j = v).filter (fun D => (D.1 i, D.2 j) = p) =
      ((Fintype.piFinset fun _ => A).filter fun X : Fin r → ℤ => X i = p.1) ×ˢ
        ((Fintype.piFinset fun _ => B).filter fun W : Fin s → ℤ => W j = p.2) := by
    ext ⟨X, W⟩
    simp only [draws, mem_filter, mem_product, Prod.ext_iff]
    constructor
    · rintro ⟨⟨⟨hX, hW⟩, -⟩, h1, h2⟩
      exact ⟨⟨hX, h1⟩, hW, h2⟩
    · rintro ⟨⟨hX, h1⟩, hW, h2⟩
      exact ⟨⟨⟨hX, hW⟩, by rw [h1, h2]; exact hp.2⟩, h1, h2⟩
  rw [this, card_product, Fintype.card_filter_piFinset_const_eq_of_mem _ _ hp.1.1,
    Fintype.card_filter_piFinset_const_eq_of_mem _ _ hp.1.2, Fintype.card_fin, Fintype.card_fin]

/-- **Entries left unchanged, in expectation.** If no value is the product of more than `P` pairs of `A × B`, then for
targets `v` fixed before the draw the hits summed over the draws are at most `P·r·s·|A|^(r−1)·|B|^(s−1)`: on average a
`p* = P/(|A||B|)` share of the `r·s` entries, however the rows and columns share their digits. -/
theorem sum_hits_le (A B : Finset ℤ) (P : ℕ) (hP : ∀ v, (fiber A B v).card ≤ P) (v : Fin r → Fin s → ℤ) :
    ∑ D ∈ draws A B r s, hits v D ≤ P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
  simp only [hits, card_filter]
  rw [sum_comm]
  calc ∑ e : Fin r × Fin s, ∑ D ∈ draws A B r s, (if D.1 e.1 * D.2 e.2 = v e.1 e.2 then 1 else 0)
      = ∑ e : Fin r × Fin s, (fiber A B (v e.1 e.2)).card * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
        refine sum_congr rfl fun e _ => ?_
        rw [← card_filter, entry_card]
    _ ≤ ∑ _e : Fin r × Fin s, P * (A.card ^ (r - 1) * B.card ^ (s - 1)) :=
        sum_le_sum fun e _ => Nat.mul_le_mul_right _ (hP _)
    _ = P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
        ring

/-- Summed over the draws, a cost that pays `c` for each entry it doesn't leave unchanged plus `c` per hit covers
`c·r·s` a draw. -/
theorem sum_cost_hits (A B : Finset ℤ) (v : Fin r → Fin s → ℤ) (c : ℕ) (cost : (Fin r → ℤ) × (Fin s → ℤ) → ℕ)
    (hcost : ∀ D ∈ draws A B r s, c * (r * s) ≤ cost D + c * hits v D) :
    c * (r * s) * (draws A B r s).card ≤ ∑ D ∈ draws A B r s, cost D + c * ∑ D ∈ draws A B r s, hits v D := by
  calc c * (r * s) * (draws A B r s).card = ∑ _D ∈ draws A B r s, c * (r * s) := by
        rw [sum_const, nsmul_eq_mul, Nat.cast_id, mul_comm]
    _ ≤ ∑ D ∈ draws A B r s, (cost D + c * hits v D) := sum_le_sum hcost
    _ = _ := by rw [sum_add_distrib, mul_sum]

/-- **`p*` in the bound.** If every draw's cost pays `c` for each of the `r·s` entries it doesn't leave unchanged, the
cost summed over the draws is at least `c·r·s·|draws|` less `c·P·r·s·|A|^(r−1)·|B|^(s−1)`: in expectation `(1 − p*)`
of the full `c·r·s`. -/
theorem sum_cost_ge (A B : Finset ℤ) (P : ℕ) (hP : ∀ v, (fiber A B v).card ≤ P) (v : Fin r → Fin s → ℤ) (c : ℕ)
    (cost : (Fin r → ℤ) × (Fin s → ℤ) → ℕ) (hcost : ∀ D ∈ draws A B r s, c * (r * s) ≤ cost D + c * hits v D) :
    c * (r * s) * (draws A B r s).card ≤
      ∑ D ∈ draws A B r s, cost D + c * (P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1))) :=
  (sum_cost_hits A B v c cost hcost).trans
    (Nat.add_le_add_left (Nat.mul_le_mul_left _ (sum_hits_le A B P hP v)) _)

theorem card_draws (b : ℕ) : (draws (signed b) (signed b) r s).card = (2 * b) ^ r * (2 * b) ^ s := by
  rw [card_product, Fintype.card_piFinset, Fintype.card_piFinset, prod_const, prod_const, card_univ, card_univ,
    Fintype.card_fin, Fintype.card_fin, card_signed]

/-- **`uw6`**: with row and column digits from `±1..±32` and targets fixed before the draw, at most `5/1024` (0.49%)
of the entries hit their target on average. -/
theorem uw6_sum_hits_le (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) :
    1024 * ∑ D ∈ draws (signed 32) (signed 32) r s, hits v D ≤ 5 * (r * s) * (draws (signed 32) (signed 32) r s).card := by
  have h := sum_hits_le (r := r) (s := s) (signed 32) (signed 32) 20 uw6_fiber_le v
  rw [card_signed] at h
  rw [card_draws]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at h
  calc 1024 * ∑ D ∈ draws (signed 32) (signed 32) (r' + 1) (s' + 1), hits v D
      ≤ 1024 * (20 * ((r' + 1) * (s' + 1)) * ((2 * 32) ^ r' * (2 * 32) ^ s')) := Nat.mul_le_mul_left _ h
    _ = 5 * ((r' + 1) * (s' + 1)) * ((2 * 32) ^ (r' + 1) * (2 * 32) ^ (s' + 1)) := by ring

/-- **`uw6` with `p*` in the bound**: a cost that pays `c` for every entry that doesn't hit its target averages at
least `(1 − 5/1024)·c·r·s` over the draws. -/
theorem uw6_sum_cost_ge (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) (c : ℕ)
    (cost : (Fin r → ℤ) × (Fin s → ℤ) → ℕ)
    (hcost : ∀ D ∈ draws (signed 32) (signed 32) r s, c * (r * s) ≤ cost D + c * hits v D) :
    1019 * (c * (r * s) * (draws (signed 32) (signed 32) r s).card) ≤
      1024 * ∑ D ∈ draws (signed 32) (signed 32) r s, cost D := by
  have hsum := sum_cost_hits _ _ v c cost hcost
  have hh := Nat.mul_le_mul_left c (uw6_sum_hits_le hr hs v)
  nlinarith

end Pouw.Dimension.Alphabet
