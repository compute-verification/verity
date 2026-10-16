import Mathlib

/-!
# Johnson's bound for a set family

Sets `A i ⊆ Fin N` (`i ∈ S`), each of at least `t` elements and pairwise meeting in at most `a`, number at most `N`
once `(a + 2)·N < t²` and `t ≤ N` (`johnson_card_le`). For the decoder, `A i` is where a candidate prefix of rows agrees
with the opened columns: two distinct prefixes differ in a row, two polynomials of degree below `k` agree at at most
`k − 1` distinct points, and the regime `(k + 1)·N < t²` is this one at `a = k − 1`.

The proof counts incidences: with `c x` the number of sets holding `x`, `Σ c x = Σ |A i| =: Z` and
`Σ c x² = Σ_{i,j} |A i ∩ A j| ≤ Z + n²a` at `n = |S|`, so Cauchy–Schwarz gives `Z² ≤ N·(Z + n²a)`. With `Z ≥ n·t` and
`2nt > N` this is `n·t² ≤ N·n·a + N·t`, which the regime contradicts unless `n ≤ N`; and `2nt ≤ N` gives `n ≤ N` at
once.
-/

namespace Claims.GS

/-- **Johnson's bound**: at most `N` sets of at least `t` elements of `Fin N`, pairwise meeting in at most `a`, when
`(a + 2)·N < t²` and `t ≤ N`. -/
theorem johnson_card_le {ι : Type} [DecidableEq ι] (S : Finset ι) {N t a : ℕ} (A : ι → Finset (Fin N))
    (hA : ∀ i ∈ S, t ≤ (A i).card) (hI : ∀ i ∈ S, ∀ j ∈ S, i ≠ j → (A i ∩ A j).card ≤ a)
    (hreg : (a + 2) * N < t * t) (htN : t ≤ N) : S.card ≤ N := by
  classical
  set n := S.card
  let c : Fin N → ℕ := fun x => ∑ i ∈ S, if x ∈ A i then 1 else 0
  have hsum : ∑ x, c x = ∑ i ∈ S, (A i).card := by
    simp only [c]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_boole, Finset.filter_mem_eq_inter, Finset.univ_inter]
    simp
  have hsq : ∑ x, c x ^ 2 = ∑ i ∈ S, ∑ j ∈ S, (A i ∩ A j).card := by
    simp only [c, sq, Finset.sum_mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    have e : ∀ x, ((if x ∈ A i then 1 else 0) * (if x ∈ A j then 1 else 0) : ℕ) =
        if x ∈ A i ∩ A j then 1 else 0 := by
      intro x
      by_cases hi : x ∈ A i <;> by_cases hj : x ∈ A j <;> simp [hi, hj]
    rw [Finset.sum_congr rfl fun x _ => e x, Finset.sum_boole, Finset.filter_mem_eq_inter, Finset.univ_inter]
    simp
  have hsq_le : ∑ x, c x ^ 2 ≤ ∑ i ∈ S, (A i).card + n * (n * a) := by
    have e : n * (n * a) = ∑ _i ∈ S, n * a := by rw [Finset.sum_const, smul_eq_mul]
    rw [hsq, e, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun i hi => ?_
    rw [← Finset.add_sum_erase S _ hi, Finset.inter_self]
    refine Nat.add_le_add_left ?_ _
    calc ∑ j ∈ S.erase i, (A i ∩ A j).card ≤ ∑ _j ∈ S.erase i, a :=
          Finset.sum_le_sum fun j hj => hI i hi j (Finset.mem_of_mem_erase hj) (Finset.ne_of_mem_erase hj).symm
      _ = (S.erase i).card * a := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ n * a := Nat.mul_le_mul_right _ (Finset.card_erase_le)
  have hCS : (∑ x, c x) ^ 2 ≤ N * ∑ x, c x ^ 2 := by
    have := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset (Fin N))) (f := c)
    simpa using this
  set Z := ∑ i ∈ S, (A i).card
  have hZ : n * t ≤ Z := by
    have := Finset.card_nsmul_le_sum S (fun i => (A i).card) t hA
    simpa [n] using this
  rw [hsum] at hCS
  have h1 : Z * Z ≤ N * Z + N * (n * (n * a)) := by nlinarith
  have ht1 : 1 ≤ t := by
    rcases Nat.eq_zero_or_pos t with h0 | h0
    · simp [h0] at hreg
    · exact h0
  by_cases hsmall : 2 * (n * t) ≤ N
  · nlinarith
  · push Not at hsmall
    have hn : 0 < n := by
      rcases Nat.eq_zero_or_pos n with h0 | h0
      · simp [h0] at hsmall
      · exact h0
    zify at *
    have hprod : (0 : ℤ) ≤ ((Z : ℤ) - n * t) * ((Z : ℤ) + n * t - N) := mul_nonneg (by linarith) (by linarith)
    have h2 : ((n : ℤ) * t) * (n * t) ≤ N * (n * (n * a)) + N * (n * t) := by nlinarith
    have h3 : (n : ℤ) * (t * t) ≤ N * (n * a) + N * t := by
      have : (n : ℤ) * (n * (t * t)) ≤ n * (N * (n * a) + N * t) := by nlinarith
      exact le_of_mul_le_mul_left this (by exact_mod_cast hn)
    have h4 : ((a : ℤ) + 2) * N + 1 ≤ t * t := by exact_mod_cast hreg
    nlinarith

end Claims.GS
