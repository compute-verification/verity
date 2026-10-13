import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Definitions.Accounting.Memory

/-!
# The counting core of lemma M

The fibres of `π : (Fin M → Bool) → Fin q → Bool` partition the `2^M` states, so the values whose fibre has more
than `T` states number at most `2^M / T` (`card_heavy_mul_le`, Markov's inequality over `z`). At `T = 2^(M − q + k)`
that is at most `2^(q − k)` of the `2^q` values (`counting`).
-/

namespace Accounting.MemoryProofs

open Accounting.Memory

/-- **Markov over the fibres.** `T` times the number of values whose fibre exceeds `T` is at most `2^M`. -/
theorem card_heavy_mul_le {M q : ℕ} (π : (Fin M → Bool) → Fin q → Bool) (T : ℕ) :
    T * (Finset.univ.filter fun z => T < (fibre π z).card).card ≤ 2 ^ M := by
  classical
  have hsum : ∑ z, (fibre π z).card = 2 ^ M := by
    have h := Finset.card_eq_sum_card_fiberwise (s := (Finset.univ : Finset (Fin M → Bool)))
      (t := (Finset.univ : Finset (Fin q → Bool))) (f := π) (fun _ _ => Finset.mem_univ _)
    rw [Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at h
    rw [h]
    rfl
  calc T * (Finset.univ.filter fun z => T < (fibre π z).card).card
      = (Finset.univ.filter fun z => T < (fibre π z).card).card • T := by rw [smul_eq_mul, mul_comm]
    _ ≤ ∑ z ∈ Finset.univ.filter (fun z => T < (fibre π z).card), (fibre π z).card :=
        Finset.card_nsmul_le_sum _ _ _ fun z hz => (Finset.mem_filter.1 hz).2.le
    _ ≤ ∑ z, (fibre π z).card := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    _ = 2 ^ M := hsum

/-- **The threshold's arithmetic.** If `2^(M + k − q)` times `N` is at most `2^M`, then `2^k · N ≤ 2^q`. -/
theorem two_pow_mul_le {M q k N : ℕ} (h : 2 ^ (M + k - q) * N ≤ 2 ^ M) : 2 ^ k * N ≤ 2 ^ q := by
  rcases le_or_gt q (M + k) with hq | hq
  · have hpow : 2 ^ (M + k - q) * 2 ^ q = 2 ^ M * 2 ^ k := by
      rw [← pow_add, ← pow_add, Nat.sub_add_cancel hq]
    have h2 : 2 ^ M * (2 ^ k * N) ≤ 2 ^ M * 2 ^ q := by
      calc 2 ^ M * (2 ^ k * N) = 2 ^ (M + k - q) * N * 2 ^ q := by
            rw [mul_right_comm, hpow]; ring
        _ ≤ 2 ^ M * 2 ^ q := Nat.mul_le_mul_right _ h
    exact Nat.le_of_mul_le_mul_left h2 (by positivity)
  · rw [show M + k - q = 0 by omega, pow_zero, one_mul] at h
    calc 2 ^ k * N ≤ 2 ^ k * 2 ^ M := Nat.mul_le_mul_left _ h
      _ = 2 ^ (M + k) := by rw [← pow_add, add_comm]
      _ ≤ 2 ^ q := Nat.pow_le_pow_right (by norm_num) hq.le

/-- **The counting core.** At most `2^(q − k)` of the `2^q` values leave more than `2^(M − q + k)` states. -/
theorem counting (M q k : ℕ) (π : (Fin M → Bool) → Fin q → Bool) :
    2 ^ k * (Finset.univ.filter fun z => 2 ^ (M + k - q) < (fibre π z).card).card ≤ 2 ^ q :=
  two_pow_mul_le (card_heavy_mul_le π _)

/-- **Markov over the fibres with advice.** Each state reads at most `2^a` values under all advice, so `T` times the
number of values whose fibre exceeds `T` is at most `2^(M + a)`. -/
theorem card_heavy_advice_mul_le {M a q : ℕ} (π : (Fin M → Bool) → (Fin a → Bool) → Fin q → Bool) (T : ℕ) :
    T * (Finset.univ.filter fun z => T < (fibreAdvice π z).card).card ≤ 2 ^ (M + a) := by
  classical
  have hsum : ∑ z, (fibreAdvice π z).card ≤ 2 ^ (M + a) := by
    calc ∑ z, (fibreAdvice π z).card
        = ∑ z : Fin q → Bool, ∑ σ : Fin M → Bool, if ∃ α, π σ α = z then 1 else 0 := by
          simp only [fibreAdvice, Finset.card_filter]
      _ = ∑ σ : Fin M → Bool, ∑ z : Fin q → Bool, if ∃ α, π σ α = z then 1 else 0 := Finset.sum_comm
      _ = ∑ σ : Fin M → Bool, (Finset.univ.image (π σ)).card := by
          refine Finset.sum_congr rfl fun σ _ => ?_
          rw [← Finset.card_filter]
          congr 1
          ext z
          simp
      _ ≤ ∑ _σ : Fin M → Bool, 2 ^ a := Finset.sum_le_sum fun σ _ =>
          Finset.card_image_le.trans (by simp)
      _ = 2 ^ (M + a) := by simp [Finset.sum_const, pow_add]
  calc T * (Finset.univ.filter fun z => T < (fibreAdvice π z).card).card
      = (Finset.univ.filter fun z => T < (fibreAdvice π z).card).card • T := by rw [smul_eq_mul, mul_comm]
    _ ≤ ∑ z ∈ Finset.univ.filter (fun z => T < (fibreAdvice π z).card), (fibreAdvice π z).card :=
        Finset.card_nsmul_le_sum _ _ _ fun z hz => (Finset.mem_filter.1 hz).2.le
    _ ≤ ∑ z, (fibreAdvice π z).card := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    _ ≤ 2 ^ (M + a) := hsum

/-- **The counting core with advice.** At most `2^(q − k)` of the `2^q` values leave more than `2^(M + a − q + k)`
states that read them with some advice. -/
theorem countingAdvice (M a q k : ℕ) (π : (Fin M → Bool) → (Fin a → Bool) → Fin q → Bool) :
    2 ^ k * (Finset.univ.filter fun z => 2 ^ (M + a + k - q) < (fibreAdvice π z).card).card ≤ 2 ^ q :=
  two_pow_mul_le (card_heavy_advice_mul_le π _)

end Accounting.MemoryProofs
