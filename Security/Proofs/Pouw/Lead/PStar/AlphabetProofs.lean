import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Int.Interval
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Two drawn digits combined per entry: how often a value fixed before the draw is hit (proofs)

A noise whose entry `(i, j)` combines a row digit and a column digit drawn after the data, `op (X i) (W j)`, is left
unchanged at that entry by a data lane `−v` fixed before the draw exactly when `op (X i) (W j) = v`. Its `p*` is the
largest share of the digit pairs that `op` sends to one value.

* **Per entry** (`entry_card`): with rows and columns drawn independently and uniformly, the draws on which entry
  `(i, j)` hits `v` are a `|fiber v|/(|A||B|)` share of all draws, however the rows and columns share their digits.
* **Over the matrix** (`sum_hits_le`, `sum_cost_ge`): if no value has more than `P` preimages, the expected number of
  entries hitting their targets is at most `p*·r·s` with `p* = P/(|A||B|)`, so a cost that pays `c` for each other
  entry averages at least `(1 − p*)·c·r·s`. That premise (`hcost`) is what a floor has to supply; nothing here proves
  it.
* **`uw6`, the product** (`uw6_fiber_le`, `uw6_fiber_120`, `uw6_sum_hits_le`): over `±1..±32` on both factors no value
  is the product of more than 20 of the 4,096 pairs, and 120 is the product of exactly 20, so `p* = 5/1024`. The count
  reduces to the divisors of `|v|` in `[1, 32]` with a cofactor in `[1, 32]` (`fiber_card_le_divs`), at most 10 for
  every `|v|` (`divs_card_le`, by `decide` up to 1,024; none past it).
* **The coin lane, `1 + b + s`** (`lane_fiber_le`, `lane_sum_hits_le`): with a row bit `b ∈ {0, 1}` and an odd column
  coin `s`, parity fixes `b` and then `s`, so each value has at most one preimage and `p* = 1/(2|B|)`: `2^−(e+1)` for
  coins odd in `±(2^e − 1)`.
* **A lane in a sum** (`fiber_add_card_le`; `lane_add_card_le` and `hwLane_add_card_le` per entry,
  `lane_add_sum_hits_le` and `hwLane_add_sum_hits_le` over the matrix): a lane on its own row and column digits caps the
  `p*` of any sum it enters at its own, whatever the other lanes, on other digits, add. hw's row lane `(1 + 2b)·σ` has
  at most two preimages a value (`hwLane_fiber_le`), so `p* ≤ 1/|B|`. Digits may be of any type, so a lane's digits
  and the rest's are a product.

The targets are fixed before the draw. A target chosen after it is a branch on the coins, which the adaptive bounds
treat (`Lead.CoinBranch`).
-/

namespace Pouw.Dimension.Alphabet

open Finset

/-- The signed alphabet `±1, …, ±b`. -/
def signed (b : ℕ) : Finset ℤ := (Icc (-(b : ℤ)) b).erase 0

/-- The pairs of `A × B` that `op` sends to `v`. -/
def fiber {α β : Type*} (op : α → β → ℤ) (A : Finset α) (B : Finset β) (v : ℤ) : Finset (α × β) :=
  (A ×ˢ B).filter fun p => op p.1 p.2 = v

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
theorem fiber_card_le_divs (b : ℕ) (v : ℤ) : (fiber (· * ·) (signed b) (signed b) v).card ≤ 2 * (divs b v.natAbs).card := by
  have hmaps : ∀ p ∈ fiber (· * ·) (signed b) (signed b) v,
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
      (fiber (· * ·) (signed b) (signed b) v : Set (ℤ × ℤ)) := by
    rintro ⟨x, w⟩ hp ⟨x', w'⟩ hp' he
    simp only [fiber, coe_filter, mem_product, Set.mem_ofPred_eq] at hp hp'
    simp only [Prod.mk.injEq, decide_eq_decide] at he
    have hx0 := (mem_signed.mp hp.1.1).1
    have hxx : x = x' := by omega
    subst hxx
    exact Prod.ext rfl (mul_left_cancel₀ hx0 (hp.2.trans hp'.2.symm))
  calc (fiber (· * ·) (signed b) (signed b) v).card ≤ (divs b v.natAbs ×ˢ (univ : Finset Bool)).card :=
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
theorem uw6_fiber_le (v : ℤ) : (fiber (· * ·) (signed 32) (signed 32) v).card ≤ 20 :=
  (fiber_card_le_divs 32 v).trans (by have := divs_card_le v.natAbs; omega)

/-- The bound is attained: 120 is the product of 20 pairs (`±(4,30)`, `±(5,24)`, …, `±(30,4)`). -/
theorem uw6_fiber_120 : 20 ≤ (fiber (· * ·) (signed 32) (signed 32) 120).card := by
  let L : Finset (ℤ × ℤ) := {(4, 30), (5, 24), (6, 20), (8, 15), (10, 12), (12, 10), (15, 8), (20, 6), (24, 5), (30, 4),
    (-4, -30), (-5, -24), (-6, -20), (-8, -15), (-10, -12), (-12, -10), (-15, -8), (-20, -6), (-24, -5), (-30, -4)}
  have hL : L.card = 20 := by decide
  rw [← hL]
  refine card_le_card fun p hp => ?_
  simp only [fiber, mem_filter, mem_product, mem_signed]
  simp only [L, mem_insert, mem_singleton] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl <;> decide

variable {r s : ℕ} {α β : Type*} [DecidableEq α] [DecidableEq β]

omit [DecidableEq α] [DecidableEq β] in
/-- Row digits `X : Fin r → A` and column digits `W : Fin s → B`, drawn independently and uniformly. -/
abbrev draws (A : Finset α) (B : Finset β) (r s : ℕ) : Finset ((Fin r → α) × (Fin s → β)) :=
  Fintype.piFinset (fun _ => A) ×ˢ Fintype.piFinset (fun _ => B)

omit [DecidableEq α] [DecidableEq β] in
/-- The entries whose `op (X i) (W j)` equals their target. -/
def hits (op : α → β → ℤ) (v : Fin r → Fin s → ℤ) (D : (Fin r → α) × (Fin s → β)) : ℕ :=
  (univ.filter fun e : Fin r × Fin s => op (D.1 e.1) (D.2 e.2) = v e.1 e.2).card

/-- **One entry**: the draws on which `op (X i) (W j) = v` number `|fiber v|·|A|^(r−1)·|B|^(s−1)`, so the entry hits `v`
with probability `|fiber v|/(|A||B|)` whatever the other rows and columns draw. -/
theorem entry_card (op : α → β → ℤ) (A : Finset α) (B : Finset β) (i : Fin r) (j : Fin s) (v : ℤ) :
    ((draws A B r s).filter fun D => op (D.1 i) (D.2 j) = v).card =
      (fiber op A B v).card * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
  have hmaps : ∀ D ∈ (draws A B r s).filter (fun D => op (D.1 i) (D.2 j) = v), (D.1 i, D.2 j) ∈ fiber op A B v := by
    intro D hD
    simp only [draws, mem_filter, mem_product, Fintype.mem_piFinset] at hD
    simp only [fiber, mem_filter, mem_product]
    exact ⟨⟨hD.1.1 i, hD.1.2 j⟩, hD.2⟩
  rw [card_eq_sum_card_fiberwise hmaps, ← Nat.cast_id ((fiber op A B v).card), ← nsmul_eq_mul, ← sum_const]
  refine sum_congr rfl fun p hp => ?_
  simp only [fiber, mem_filter, mem_product] at hp
  have : ((draws A B r s).filter fun D => op (D.1 i) (D.2 j) = v).filter (fun D => (D.1 i, D.2 j) = p) =
      ((Fintype.piFinset fun _ => A).filter fun X : Fin r → α => X i = p.1) ×ˢ
        ((Fintype.piFinset fun _ => B).filter fun W : Fin s → β => W j = p.2) := by
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
theorem sum_hits_le (op : α → β → ℤ) (A : Finset α) (B : Finset β) (P : ℕ) (hP : ∀ v, (fiber op A B v).card ≤ P)
    (v : Fin r → Fin s → ℤ) :
    ∑ D ∈ draws A B r s, hits op v D ≤ P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
  simp only [hits, card_filter]
  rw [sum_comm]
  calc ∑ e : Fin r × Fin s, ∑ D ∈ draws A B r s, (if op (D.1 e.1) (D.2 e.2) = v e.1 e.2 then 1 else 0)
      = ∑ e : Fin r × Fin s, (fiber op A B (v e.1 e.2)).card * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
        refine sum_congr rfl fun e _ => ?_
        rw [← card_filter, entry_card]
    _ ≤ ∑ _e : Fin r × Fin s, P * (A.card ^ (r - 1) * B.card ^ (s - 1)) :=
        sum_le_sum fun e _ => Nat.mul_le_mul_right _ (hP _)
    _ = P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1)) := by
        rw [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
        ring

omit [DecidableEq α] [DecidableEq β] in
/-- Summed over the draws, a cost that pays `c` for each entry it doesn't leave unchanged plus `c` per hit covers
`c·r·s` a draw. -/
theorem sum_cost_hits (op : α → β → ℤ) (A : Finset α) (B : Finset β) (v : Fin r → Fin s → ℤ) (c : ℕ)
    (cost : (Fin r → α) × (Fin s → β) → ℕ) (hcost : ∀ D ∈ draws A B r s, c * (r * s) ≤ cost D + c * hits op v D) :
    c * (r * s) * (draws A B r s).card ≤ ∑ D ∈ draws A B r s, cost D + c * ∑ D ∈ draws A B r s, hits op v D := by
  calc c * (r * s) * (draws A B r s).card = ∑ _D ∈ draws A B r s, c * (r * s) := by
        rw [sum_const, nsmul_eq_mul, Nat.cast_id, mul_comm]
    _ ≤ ∑ D ∈ draws A B r s, (cost D + c * hits op v D) := sum_le_sum hcost
    _ = _ := by rw [sum_add_distrib, mul_sum]

/-- **`p*` in the bound.** If every draw's cost pays `c` for each of the `r·s` entries it doesn't leave unchanged, the
cost summed over the draws is at least `c·r·s·|draws|` less `c·P·r·s·|A|^(r−1)·|B|^(s−1)`: in expectation `(1 − p*)`
of the full `c·r·s`. -/
theorem sum_cost_ge (op : α → β → ℤ) (A : Finset α) (B : Finset β) (P : ℕ) (hP : ∀ v, (fiber op A B v).card ≤ P)
    (v : Fin r → Fin s → ℤ) (c : ℕ) (cost : (Fin r → α) × (Fin s → β) → ℕ)
    (hcost : ∀ D ∈ draws A B r s, c * (r * s) ≤ cost D + c * hits op v D) :
    c * (r * s) * (draws A B r s).card ≤
      ∑ D ∈ draws A B r s, cost D + c * (P * (r * s) * (A.card ^ (r - 1) * B.card ^ (s - 1))) :=
  (sum_cost_hits op A B v c cost hcost).trans
    (Nat.add_le_add_left (Nat.mul_le_mul_left _ (sum_hits_le op A B P hP v)) _)

omit [DecidableEq α] [DecidableEq β] in
theorem card_draws (A : Finset α) (B : Finset β) : (draws A B r s).card = A.card ^ r * B.card ^ s := by
  rw [card_product, Fintype.card_piFinset, Fintype.card_piFinset, prod_const, prod_const, card_univ, card_univ,
    Fintype.card_fin, Fintype.card_fin]

/-- **`uw6`**: with row and column digits from `±1..±32` and targets fixed before the draw, at most `5/1024` (0.49%)
of the entries hit their target on average. -/
theorem uw6_sum_hits_le (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) :
    1024 * ∑ D ∈ draws (signed 32) (signed 32) r s, hits (· * ·) v D ≤
      5 * (r * s) * (draws (signed 32) (signed 32) r s).card := by
  have h := sum_hits_le (r := r) (s := s) (· * ·) (signed 32) (signed 32) 20 uw6_fiber_le v
  rw [card_signed] at h
  rw [card_draws, card_signed]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at h
  calc 1024 * ∑ D ∈ draws (signed 32) (signed 32) (r' + 1) (s' + 1), hits (· * ·) v D
      ≤ 1024 * (20 * ((r' + 1) * (s' + 1)) * ((2 * 32) ^ r' * (2 * 32) ^ s')) := Nat.mul_le_mul_left _ h
    _ = 5 * ((r' + 1) * (s' + 1)) * ((2 * 32) ^ (r' + 1) * (2 * 32) ^ (s' + 1)) := by ring

/-- **`uw6` with `p*` in the bound**: a cost that pays `c` for every entry that doesn't hit its target averages at
least `(1 − 5/1024)·c·r·s` over the draws. -/
theorem uw6_sum_cost_ge (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) (c : ℕ)
    (cost : (Fin r → ℤ) × (Fin s → ℤ) → ℕ)
    (hcost : ∀ D ∈ draws (signed 32) (signed 32) r s, c * (r * s) ≤ cost D + c * hits (· * ·) v D) :
    1019 * (c * (r * s) * (draws (signed 32) (signed 32) r s).card) ≤
      1024 * ∑ D ∈ draws (signed 32) (signed 32) r s, cost D := by
  have hsum := sum_cost_hits _ _ _ v c cost hcost
  have hh := Nat.mul_le_mul_left c (uw6_sum_hits_le hr hs v)
  nlinarith

/-- The odd coins `±1, ±3, …, ±(2^e − 1)`. -/
def oddLane (e : ℕ) : Finset ℤ := (Icc (-(2 ^ (e - 1) : ℤ)) (2 ^ (e - 1) - 1)).image fun k => 2 * k + 1

theorem odd_of_mem_oddLane {e : ℕ} {x : ℤ} (hx : x ∈ oddLane e) : Odd x := by
  obtain ⟨k, -, rfl⟩ := mem_image.mp hx
  exact ⟨k, rfl⟩

theorem card_oddLane (e : ℕ) : (oddLane (e + 1)).card = 2 ^ (e + 1) := by
  rw [oddLane, card_image_of_injective _ (fun a b h => by omega), Int.card_Icc, Nat.add_sub_cancel]
  have : (2 : ℤ) ^ e - 1 + 1 - -2 ^ e = ((2 ^ (e + 1) : ℕ) : ℤ) := by push_cast; ring
  rw [this, Int.toNat_natCast]

/-- The coin lane's increment `1 + b + s` from a row bit `b` and an odd coin `s`. -/
def lane (b x : ℤ) : ℤ := 1 + b + x

/-- **One preimage at most**: the parity of `1 + b + s` fixes the bit and then the odd coin, so over bits `{0, 1}` and
any set of odd coins each value is hit by at most one pair, a `1/(2|B|)` share. -/
theorem lane_fiber_le (B : Finset ℤ) (hB : ∀ x ∈ B, Odd x) (v : ℤ) : (fiber lane {0, 1} B v).card ≤ 1 := by
  rw [card_le_one]
  rintro ⟨b, x⟩ hp ⟨b', x'⟩ hp'
  simp only [fiber, mem_filter, mem_product, mem_insert, mem_singleton] at hp hp'
  simp only [lane] at hp hp'
  obtain ⟨⟨hb, hx⟩, hv⟩ := hp
  obtain ⟨⟨hb', hx'⟩, hv'⟩ := hp'
  obtain ⟨k, rfl⟩ := hB _ hx
  obtain ⟨k', rfl⟩ := hB _ hx'
  have hbb : b = b' := by omega
  subst hbb
  exact Prod.ext rfl (by omega)

/-- **The coin lane**: with row bits from `{0, 1}`, column coins odd in `±(2^(e+1) − 1)` and targets fixed before the
draw, at most a `2^−(e+2)` share of the entries hit their target on average (`p* = 1/32` at 4-bit coins). -/
theorem lane_sum_hits_le (e : ℕ) (hr : 0 < r) (hs : 0 < s) (v : Fin r → Fin s → ℤ) :
    2 ^ (e + 2) * ∑ D ∈ draws ({0, 1} : Finset ℤ) (oddLane (e + 1)) r s, hits lane v D ≤
      r * s * (draws ({0, 1} : Finset ℤ) (oddLane (e + 1)) r s).card := by
  have h := sum_hits_le (r := r) (s := s) lane {0, 1} (oddLane (e + 1)) 1
    (lane_fiber_le _ fun _ hx => odd_of_mem_oddLane hx) v
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  rw [card_oddLane, h2] at h
  rw [card_draws, card_oddLane, h2]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at h
  calc 2 ^ (e + 2) * ∑ D ∈ draws {0, 1} (oddLane (e + 1)) (r' + 1) (s' + 1), hits lane v D
      ≤ 2 ^ (e + 2) * (1 * ((r' + 1) * (s' + 1)) * (2 ^ r' * (2 ^ (e + 1)) ^ s')) := Nat.mul_le_mul_left _ h
    _ = (r' + 1) * (s' + 1) * (2 ^ (r' + 1) * (2 ^ (e + 1)) ^ (s' + 1)) := by ring

/-- **A lane on its own coins caps any sum it enters**: with row digits `A₁ × A'` and column digits `B₁ × B'`, the
entry `op a₁ b₁ + rest a' b'` hits `v` on at most `P·|A'|·|B'|` digit pairs when no value of `op` has more than `P`
preimages in `A₁ × B₁`. So its `p*` is at most `op`'s, `P/(|A₁||B₁|)`, whatever the other lanes add. -/
theorem fiber_add_card_le (op : ℤ → ℤ → ℤ) (rest : α → β → ℤ)
    (A₁ B₁ : Finset ℤ) (A' : Finset α) (B' : Finset β) {P : ℕ} (hP : ∀ v, (fiber op A₁ B₁ v).card ≤ P) (v : ℤ) :
    (((A₁ ×ˢ A') ×ˢ (B₁ ×ˢ B')).filter fun p => op p.1.1 p.2.1 + rest p.1.2 p.2.2 = v).card
      ≤ P * (A'.card * B'.card) := by
  set T := ((A₁ ×ˢ A') ×ˢ (B₁ ×ˢ B')).filter fun p => op p.1.1 p.2.1 + rest p.1.2 p.2.2 = v with hT
  have hmaps : ∀ p ∈ T, (p.1.2, p.2.2) ∈ A' ×ˢ B' := by
    intro p hp
    rw [hT, mem_filter] at hp
    simp only [mem_product] at hp ⊢
    exact ⟨hp.1.1.2, hp.1.2.2⟩
  rw [card_eq_sum_card_fiberwise fun p hp => hmaps p hp]
  calc ∑ y ∈ A' ×ˢ B', (T.filter fun p => (p.1.2, p.2.2) = y).card
      ≤ ∑ _y ∈ A' ×ˢ B', P := sum_le_sum fun y _ => by
        refine (card_le_card_of_injOn (fun p => (p.1.1, p.2.1)) (fun p hp => ?_) ?_).trans
          (hP (v - rest y.1 y.2))
        · rw [mem_coe, mem_filter, hT, mem_filter] at hp
          obtain ⟨⟨hpm, hpv⟩, hpy⟩ := hp
          simp only [mem_product] at hpm
          rw [mem_coe, fiber, mem_filter, mem_product]
          refine ⟨⟨hpm.1.1, hpm.2.1⟩, ?_⟩
          rw [← hpy]
          dsimp only
          linarith
        · intro p hp q hq he
          rw [mem_coe, mem_filter] at hp hq
          have h1 : (p.1.2, p.2.2) = (q.1.2, q.2.2) := hp.2.trans hq.2.symm
          have h2 : (p.1.1, p.2.1) = (q.1.1, q.2.1) := he
          obtain ⟨h1a, h1b⟩ := Prod.mk.inj h1
          obtain ⟨h2a, h2b⟩ := Prod.mk.inj h2
          exact Prod.ext (Prod.ext h2a h1a) (Prod.ext h2b h1b)
    _ = P * (A'.card * B'.card) := by rw [sum_const, card_product, nsmul_eq_mul, Nat.cast_id, mul_comm]

/-- **The coin lane in any sum**: a lane `1 + b + s` on its own row bit and odd column coin keeps `p* ≤ 1/(2|B₁|)` per
entry whatever other lanes, on other digits, add to it. -/
theorem lane_add_card_le (rest : α → β → ℤ) (B₁ : Finset ℤ)
    (hB : ∀ x ∈ B₁, Odd x) (A' : Finset α) (B' : Finset β) (v : ℤ) :
    2 * B₁.card * ((((({0, 1} : Finset ℤ) ×ˢ A') ×ˢ (B₁ ×ˢ B')).filter fun p =>
        lane p.1.1 p.2.1 + rest p.1.2 p.2.2 = v).card)
      ≤ ((({0, 1} : Finset ℤ) ×ˢ A') ×ˢ (B₁ ×ˢ B')).card := by
  have h := fiber_add_card_le lane rest {0, 1} B₁ A' B' (P := 1) (lane_fiber_le B₁ hB) v
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  rw [card_product, card_product, card_product, h2]
  calc 2 * B₁.card * _ ≤ 2 * B₁.card * (1 * (A'.card * B'.card)) := Nat.mul_le_mul_left _ h
    _ = 2 * A'.card * (B₁.card * B'.card) := by ring

/-- hw's row lane: a row bit `b` scales the column coin `σ` by `1 + 2b`. -/
def hwLane (b x : ℤ) : ℤ := (1 + 2 * b) * x

/-- **Two preimages at most**: `b = 0` leaves `σ = v` and `b = 1` leaves `3σ = v`, so over bits `{0, 1}` and any coins
each value is hit by at most two pairs, a `1/|B|` share. -/
theorem hwLane_fiber_le (B : Finset ℤ) (v : ℤ) : (fiber hwLane {0, 1} B v).card ≤ 2 := by
  have hsub : fiber hwLane {0, 1} B v ⊆ {((0 : ℤ), v), ((1 : ℤ), v / 3)} := by
    rintro ⟨b, x⟩ hp
    simp only [fiber, mem_filter, mem_product, mem_insert, mem_singleton] at hp
    simp only [hwLane] at hp
    obtain ⟨⟨hb | hb, _⟩, hv⟩ := hp
    · subst hb; simp only [mem_insert, mem_singleton, Prod.mk.injEq]; left; exact ⟨trivial, by omega⟩
    · subst hb; simp only [mem_insert, mem_singleton, Prod.mk.injEq]; right; exact ⟨trivial, by omega⟩
  exact (card_le_card hsub).trans (card_le_two)

/-- **hw's row lane in any sum**: `(1 + 2b)·σ` on its own row bit and column coin keeps `p* ≤ 1/|B₁|` per entry whatever
other lanes, on other digits, add to it. -/
theorem hwLane_add_card_le (rest : α → β → ℤ) (B₁ : Finset ℤ)
    (A' : Finset α) (B' : Finset β) (v : ℤ) :
    B₁.card * ((((({0, 1} : Finset ℤ) ×ˢ A') ×ˢ (B₁ ×ˢ B')).filter fun p =>
        hwLane p.1.1 p.2.1 + rest p.1.2 p.2.2 = v).card)
      ≤ ((({0, 1} : Finset ℤ) ×ˢ A') ×ˢ (B₁ ×ˢ B')).card := by
  have h := fiber_add_card_le hwLane rest {0, 1} B₁ A' B' (P := 2) (hwLane_fiber_le B₁) v
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  rw [card_product, card_product, card_product, h2]
  calc B₁.card * _ ≤ B₁.card * (2 * (A'.card * B'.card)) := Nat.mul_le_mul_left _ h
    _ = 2 * A'.card * (B₁.card * B'.card) := by ring

/-- **The coin lane in any sum, over the matrix**: with row digits `{0, 1} × A'`, column digits `oddLane (e + 1) × B'`
and targets fixed before the draw, entries `1 + b + s + rest a' b'` hit their target on at most a `2^−(e+2)` share of
the entries on average, whatever `rest` adds. -/
theorem lane_add_sum_hits_le (rest : α → β → ℤ) (A' : Finset α) (B' : Finset β) (e : ℕ) (hr : 0 < r) (hs : 0 < s)
    (v : Fin r → Fin s → ℤ) :
    2 ^ (e + 2) * ∑ D ∈ draws (({0, 1} : Finset ℤ) ×ˢ A') (oddLane (e + 1) ×ˢ B') r s,
        hits (fun a b => lane a.1 b.1 + rest a.2 b.2) v D ≤
      r * s * (draws (({0, 1} : Finset ℤ) ×ˢ A') (oddLane (e + 1) ×ˢ B') r s).card := by
  have hP : ∀ w, (fiber (fun a b => lane a.1 b.1 + rest a.2 b.2) (({0, 1} : Finset ℤ) ×ˢ A')
      (oddLane (e + 1) ×ˢ B') w).card ≤ A'.card * B'.card := fun w => by
    have := fiber_add_card_le lane rest {0, 1} (oddLane (e + 1)) A' B' (P := 1)
      (lane_fiber_le _ fun _ hx => odd_of_mem_oddLane hx) w
    rw [one_mul] at this
    exact this
  have h := sum_hits_le (r := r) (s := s) _ _ _ _ hP v
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  rw [card_product, card_product, card_oddLane, h2] at h
  rw [card_draws, card_product, card_product, card_oddLane, h2]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at h
  calc 2 ^ (e + 2) * ∑ D ∈ draws (({0, 1} : Finset ℤ) ×ˢ A') (oddLane (e + 1) ×ˢ B') (r' + 1) (s' + 1),
        hits (fun a b => lane a.1 b.1 + rest a.2 b.2) v D
      ≤ 2 ^ (e + 2) * (A'.card * B'.card * ((r' + 1) * (s' + 1)) *
          ((2 * A'.card) ^ r' * (2 ^ (e + 1) * B'.card) ^ s')) := Nat.mul_le_mul_left _ h
    _ = (r' + 1) * (s' + 1) * ((2 * A'.card) ^ (r' + 1) * (2 ^ (e + 1) * B'.card) ^ (s' + 1)) := by ring

/-- **hw's row lane in any sum, over the matrix**: with row digits `{0, 1} × A'` and column digits `B₁ × B'`, entries
`(1 + 2b)·σ + rest a' b'` hit targets fixed before the draw on at most a `1/|B₁|` share of the entries on average. -/
theorem hwLane_add_sum_hits_le (rest : α → β → ℤ) (B₁ : Finset ℤ) (A' : Finset α) (B' : Finset β) (hr : 0 < r)
    (hs : 0 < s) (v : Fin r → Fin s → ℤ) :
    B₁.card * ∑ D ∈ draws (({0, 1} : Finset ℤ) ×ˢ A') (B₁ ×ˢ B') r s,
        hits (fun a b => hwLane a.1 b.1 + rest a.2 b.2) v D ≤
      r * s * (draws (({0, 1} : Finset ℤ) ×ˢ A') (B₁ ×ˢ B') r s).card := by
  have hP : ∀ w, (fiber (fun a b => hwLane a.1 b.1 + rest a.2 b.2) (({0, 1} : Finset ℤ) ×ˢ A') (B₁ ×ˢ B') w).card
      ≤ 2 * (A'.card * B'.card) := fun w =>
    fiber_add_card_le hwLane rest {0, 1} B₁ A' B' (P := 2) (hwLane_fiber_le B₁) w
  have h := sum_hits_le (r := r) (s := s) _ _ _ _ hP v
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  rw [card_product, card_product, h2] at h
  rw [card_draws, card_product, card_product, h2]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  simp only [Nat.add_sub_cancel] at h
  calc B₁.card * ∑ D ∈ draws (({0, 1} : Finset ℤ) ×ˢ A') (B₁ ×ˢ B') (r' + 1) (s' + 1),
        hits (fun a b => hwLane a.1 b.1 + rest a.2 b.2) v D
      ≤ B₁.card * (2 * (A'.card * B'.card) * ((r' + 1) * (s' + 1)) *
          ((2 * A'.card) ^ r' * (B₁.card * B'.card) ^ s')) := Nat.mul_le_mul_left _ h
    _ = (r' + 1) * (s' + 1) * ((2 * A'.card) ^ (r' + 1) * (B₁.card * B'.card) ^ (s' + 1)) := by ring

end Pouw.Dimension.Alphabet
