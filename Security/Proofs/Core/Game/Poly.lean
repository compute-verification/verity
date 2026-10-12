import Proofs.Core.Game.Basic
import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Mathlib.Algebra.Polynomial.Roots

/-!
# A live coin hits a zero of a fixed polynomial rarely

The per-coin bound behind every round-by-round step of the soundness proof: once the prover has
fixed a nonzero polynomial, a uniform coin is one of its zeros with probability at most its
degree over the field size (Schwartz–Zippel; Mathlib's `schwartz_zippel_totalDegree` and root
counting).
-/

namespace Verity.Game

open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]

/-- **Univariate.** A nonzero polynomial of degree `d` vanishes at a uniform point with
probability at most `d / |F|`. -/
theorem prCoin_eval_eq_zero_le (P : Polynomial F) (hP : P ≠ 0) :
    prCoin (fun r : F => P.eval r = 0) ≤ (P.natDegree : ℝ≥0∞) / Fintype.card F := by
  classical
  rw [prCoin_eq_card]
  have hsub : (Finset.univ.filter fun r : F => P.eval r = 0) ⊆ P.roots.toFinset := by
    intro r hr
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr
    simpa [Multiset.mem_toFinset, Polynomial.mem_roots hP] using hr
  have hcard : (Finset.univ.filter fun r : F => P.eval r = 0).card ≤ P.natDegree :=
    (Finset.card_le_card hsub).trans ((Multiset.toFinset_card_le _).trans (Polynomial.card_roots' P))
  gcongr

/-- Two different polynomials of degree at most `d` agree at a uniform point with probability at
most `d / |F|`. -/
theorem prCoin_eval_eq_eval_le (P Q : Polynomial F) (hPQ : P ≠ Q) (d : ℕ)
    (hP : P.natDegree ≤ d) (hQ : Q.natDegree ≤ d) :
    prCoin (fun r : F => P.eval r = Q.eval r) ≤ (d : ℝ≥0∞) / Fintype.card F := by
  have hne : P - Q ≠ 0 := sub_ne_zero.2 hPQ
  have hdeg : (P - Q).natDegree ≤ d := (Polynomial.natDegree_sub_le P Q).trans (max_le hP hQ)
  calc prCoin (fun r : F => P.eval r = Q.eval r)
      = prCoin (fun r : F => (P - Q).eval r = 0) := by
        congr 1; funext r; simp [sub_eq_zero]
    _ ≤ ((P - Q).natDegree : ℝ≥0∞) / Fintype.card F := prCoin_eval_eq_zero_le _ hne
    _ ≤ (d : ℝ≥0∞) / Fintype.card F := by gcongr

/-- **Multivariate (Schwartz–Zippel).** A nonzero polynomial of total degree `d` in `n` variables
vanishes at a uniform vector with probability at most `d / |F|`. -/
theorem prCoin_mvEval_eq_zero_le {n : ℕ} (p : MvPolynomial (Fin n) F) (hp : p ≠ 0) :
    prCoin (fun f : Fin n → F => MvPolynomial.eval f p = 0) ≤
      (p.totalDegree : ℝ≥0∞) / Fintype.card F := by
  classical
  rw [prCoin_eq_card]
  have h := MvPolynomial.schwartz_zippel_totalDegree hp (Finset.univ : Finset F)
  rw [Fintype.piFinset_univ, Finset.card_univ] at h
  have hq : (0 : ℚ≥0) < (Fintype.card F : ℚ≥0) := by exact_mod_cast Fintype.card_pos
  rw [div_le_div_iff₀ (by positivity) hq] at h
  set a := (Finset.univ.filter fun f : Fin n → F => MvPolynomial.eval f p = 0).card
  have hnat : a * Fintype.card F ≤ p.totalDegree * Fintype.card F ^ n := by exact_mod_cast h
  rw [Fintype.card_fun, Fintype.card_fin]
  have hF0 : (Fintype.card F : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hFt : (Fintype.card F : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hFn0 : ((Fintype.card F ^ n : ℕ) : ℝ≥0∞) ≠ 0 := by
    exact_mod_cast (pow_ne_zero _ Fintype.card_ne_zero)
  have hFnt : ((Fintype.card F ^ n : ℕ) : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  push_cast
  calc (a : ℝ≥0∞) / (Fintype.card F : ℝ≥0∞) ^ n
      = (a * Fintype.card F : ℝ≥0∞) / ((Fintype.card F : ℝ≥0∞) ^ n * Fintype.card F) := by
        rw [ENNReal.mul_div_mul_right _ _ hF0 hFt]
    _ ≤ (p.totalDegree * (Fintype.card F : ℝ≥0∞) ^ n) /
          ((Fintype.card F : ℝ≥0∞) ^ n * Fintype.card F) := by
        exact ENNReal.div_le_div_right (by exact_mod_cast hnat) _
    _ = (p.totalDegree : ℝ≥0∞) / Fintype.card F := by
        have h0 : (Fintype.card F : ℝ≥0∞) ^ n ≠ 0 := by simp at hFn0 ⊢
        have ht : (Fintype.card F : ℝ≥0∞) ^ n ≠ ⊤ := by simp at hFnt ⊢
        rw [mul_comm ((Fintype.card F : ℝ≥0∞) ^ n) (Fintype.card F : ℝ≥0∞),
          ENNReal.mul_div_mul_right _ _ h0 ht]

end Verity.Game
