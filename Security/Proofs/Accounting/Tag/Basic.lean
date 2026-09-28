import Mathlib.Algebra.Polynomial.Roots
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Data.List.GetD
import Definitions.Accounting.Tag.PolyHash

/-!
# Shared lemmas for the tag proofs

`p = 2^31 − 1` is prime (Lucas–Lehmer, Mathlib's `norm_num` extension), so `ZMod P` is a field; probabilities under a
uniform draw are counts; and a nonzero sparse polynomial of degree at most `d` vanishes at at most `d` points (the root
count `PousDigest.card_zeros_le` makes for PoUS's digest, restated here so this area doesn't import PoUS's proofs).
-/

namespace Accounting.TagProofs

open Accounting.Tag Polynomial
open scoped ENNReal

/-- `p = 2^31 − 1` is prime: it is the Mersenne number `M_31`, which passes the Lucas–Lehmer test. -/
theorem prime_P : P.Prime := by
  have h := lucas_lehmer_sufficiency 31 (by norm_num) (by norm_num)
  simpa [mersenne] using h

instance fact_prime_P : Fact P.Prime := ⟨prime_P⟩

/-- `prob` under any distribution on a finite type: the sum of the point masses where the event holds. -/
theorem prob_eq_sum {α : Type} [Fintype α] (μ : PMF α) (E : α → Prop) [DecidablePred E] :
    prob μ E = ∑ x, if E x then μ x else 0 := by
  unfold prob
  rw [PMF.toOuterMeasure_apply_fintype]
  refine Finset.sum_congr rfl fun x _ => ?_
  by_cases h : E x <;> simp [Set.indicator, h]

/-- A distribution's point masses on a finite type sum to `1`. -/
theorem sum_pmf {α : Type} [Fintype α] (μ : PMF α) : ∑ x, μ x = 1 := by
  rw [← tsum_fintype (L := SummationFilter.unconditional _)]
  exact μ.tsum_coe

/-- An event of at most `n` points has probability at most `n` times the largest point mass. -/
theorem prob_le_card_mul_iSup {α : Type} [Fintype α] (μ : PMF α) (E : α → Prop) [DecidablePred E] (n : ℕ)
    (h : (Finset.univ.filter E).card ≤ n) : prob μ E ≤ (n : ℝ≥0∞) * ⨆ x, μ x := by
  rw [prob_eq_sum, ← Finset.sum_filter]
  calc ∑ x ∈ Finset.univ.filter E, μ x ≤ ∑ _x ∈ Finset.univ.filter E, ⨆ y, μ y :=
        Finset.sum_le_sum fun x _ => le_iSup (fun y => μ y) x
    _ = ((Finset.univ.filter E).card : ℝ≥0∞) * ⨆ y, μ y := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (n : ℝ≥0∞) * ⨆ y, μ y := by gcongr

/-- The uniform distribution's largest point mass is `1 / |α|`. -/
theorem iSup_uniform {α : Type} [Fintype α] [Nonempty α] :
    (⨆ x, PMF.uniformOfFintype α x) = (Fintype.card α : ℝ≥0∞)⁻¹ := by
  simp only [PMF.uniformOfFintype_apply, ciSup_const]

/-- An event of at most `n` points of a finite type has uniform probability at most `n / |α|`. -/
theorem prob_uniform_le {α : Type} [Fintype α] [Nonempty α] (E : α → Prop) [DecidablePred E] (n : ℕ)
    (h : (Finset.univ.filter E).card ≤ n) :
    prob (PMF.uniformOfFintype α) E ≤ (n : ℝ≥0∞) / Fintype.card α := by
  rw [div_eq_mul_inv, ← iSup_uniform]
  exact prob_le_card_mul_iSup _ E n h

theorem card_zmod : Fintype.card (ZMod P) = P := ZMod.card P

/-- A nonzero sum `Σ_i a_i x^(e_i)` over a finite field, with distinct exponents each at most `d`, vanishes at at most `d`
points. -/
theorem card_zeros_le {F : Type} [Field F] [Fintype F] [DecidableEq F] {n : ℕ} (a : Fin n → F) (e : Fin n → ℕ)
    (he : Function.Injective e) (d : ℕ) (hd : ∀ i, e i ≤ d) (ha : a ≠ 0) :
    (Finset.univ.filter fun x : F => ∑ i, a i * x ^ e i = 0).card ≤ d := by
  classical
  set Q : F[X] := ∑ i, C (a i) * X ^ e i with hQ
  have heval : ∀ x, Q.eval x = ∑ i, a i * x ^ e i := by
    intro x
    simp [hQ, eval_finsetSum]
  obtain ⟨i0, hi0⟩ : ∃ i, a i ≠ 0 := by
    by_contra h
    push Not at h
    exact ha (funext h)
  have hcoeff : Q.coeff (e i0) = a i0 := by
    rw [hQ, finsetSum_coeff, Finset.sum_eq_single i0]
    · simp
    · intro j _ hj
      simp [he.ne hj.symm]
    · simp
  have hQ0 : Q ≠ 0 := by
    intro h
    apply hi0
    rw [← hcoeff, h, coeff_zero]
  have hdeg : Q.natDegree ≤ d :=
    natDegree_sum_le_of_forall_le _ _ fun i _ => (natDegree_C_mul_X_pow_le _ _).trans (hd i)
  have hsub : (Finset.univ.filter fun x : F => ∑ i, a i * x ^ e i = 0) ⊆ Q.roots.toFinset := by
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
    rw [Multiset.mem_toFinset, mem_roots hQ0, IsRoot, heval]
    exact hx
  exact (Finset.card_le_card hsub).trans ((Multiset.toFinset_card_le _).trans ((card_roots' Q).trans hdeg))

end Accounting.TagProofs
