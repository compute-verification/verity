import Definitions.Pous.P2.Cost

set_option exponentiation.threshold 20000

namespace Scratch

open Pous

/-- The ring model's primitive space has a factor `p`, so its size divides no power of two when `p` is odd and `B ≥ 1`. -/
theorem ringRO_card_not_dvd_two_pow {B w p : ℕ} [NeZero p] (hB : 0 < B) (hp : Odd p) (hp1 : 1 < p) (N : ℕ) :
    ¬ Fintype.card (ringRO B w p).Ω ∣ 2 ^ N := by
  intro h
  have hcard : Fintype.card (ringRO B w p).Ω =
      Fintype.card (OddLabeling p) * (Fintype.card (Bits w) * p) ^ B := by
    have e : (ringRO B w p).Ω = (OddLabeling p × (Fin B → Bits w × Fin p)) := rfl
    rw [Fintype.card_congr (Equiv.cast e)]
    simp [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  have hpd : p ∣ Fintype.card (ringRO B w p).Ω := by
    rw [hcard]
    exact Dvd.dvd.mul_left (dvd_pow (dvd_mul_left p _) hB.ne') _
  obtain ⟨m, -, hm⟩ := (Nat.dvd_prime_pow Nat.prime_two).1 (hpd.trans h)
  rcases m with _ | m
  · simp at hm; omega
  · rw [hm] at hp
    exact Nat.not_even_iff_odd.mpr hp (Nat.even_pow.mpr ⟨even_two, m.succ_ne_zero⟩)

/-- **No coupling of the ring model's coins into a game's coins.** If a product of coordinates is in bijection with a
coin space of `2 ^ N` points (the causal game's releases, `Fin n → Bool` each), no coordinate is in bijection with the
ring model's primitive space times anything. -/
theorem no_ring_coordinate {B w p : ℕ} [NeZero p] (hB : 0 < B) (hp : Odd p) (hp1 : 1 < p)
    {X : Type} [Fintype X] {N : ℕ} (hX : Fintype.card X = 2 ^ N)
    {I : Type} [Fintype I] [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)] (e : ((i : I) → Ω i) ≃ X)
    (j : I) {Y : Type} [Fintype Y] (ej : Ω j ≃ (ringRO B w p).Ω × Y) : False := by
  apply ringRO_card_not_dvd_two_pow (w := w) hB hp hp1 N
  have h1 : Fintype.card (Ω j) ∣ Fintype.card ((i : I) → Ω i) := by
    rw [Fintype.card_pi]
    exact Finset.dvd_prod_of_mem (fun i => Fintype.card (Ω i)) (Finset.mem_univ j)
  have h2 : Fintype.card (ringRO B w p).Ω ∣ Fintype.card (Ω j) := by
    rw [Fintype.card_congr ej, Fintype.card_prod]
    exact dvd_mul_right _ _
  rw [← hX, ← Fintype.card_congr e]
  exact h2.trans h1

/-- At PoUS's deployed ring, `p = 2^16448 − 21065` is odd and above one. -/
theorem deployed_p_odd : Odd (2 ^ 16448 - 21065 : ℕ) ∧ 1 < (2 ^ 16448 - 21065 : ℕ) := by
  have h : (21066 : ℕ) < 2 ^ 16448 :=
    calc (21066 : ℕ) < 2 ^ 15 := by norm_num
      _ ≤ 2 ^ 16448 := Nat.pow_le_pow_right (by norm_num) (by norm_num)
  refine ⟨?_, by omega⟩
  rw [Nat.odd_sub (by omega)]
  simp [Nat.odd_iff, Nat.even_iff, Nat.pow_mod]

end Scratch

#print axioms Scratch.no_ring_coordinate
#print axioms Scratch.deployed_p_odd
