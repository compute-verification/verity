import Proofs.Flock.Soundness.Audit.Pooled
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The pooled escape in `Claims.Pr`

The audit layer's pooled escape (`Audit/Pooled.lean`, in `prCoin`) read in `Claims.Pr` through
`Claims.Sampling.Pr_eq_toReal_prCoin`. It sits here, outside the audit layer, which imports only Mathlib, Flock's
spec, the interaction model and itself (`Soundness/DESIGN.md` §12).

* `pooled_escape_Pr`: `Pr[escape ∧ Σ_t |W_t| ≥ M] ≤ (1 − kd/N)^M`.
* `pooled_escape_log`: at `δ`, with `q = kd/N`, the total reaches `ln(1/δ)/q` on an escape with probability at most
  `δ`.
* `pooled_escape_log_adapted`: the same for a wrong-set map on whole runs that doesn't look ahead
  (`Periods.Predictable`), which is a `Pooled` (`Pooled.ofAdapted`).
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

variable {N kd : ℕ}

/-- `pooled_escape` in `Claims.Pr`. -/
theorem pooled_escape_Pr {hk : kd ≤ N} {T : ℕ} (A : Pooled N kd hk T) (M : ℕ) :
    Claims.Pr (fun ω => A.Escapes ω ∧ M ≤ A.total ω) ≤ (1 - (kd : ℝ) / N) ^ M := by
  have hle : (kd : ℝ≥0∞) / N ≤ 1 := ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast hk)
  have hne : (1 - (kd : ℝ≥0∞) / N) ^ M ≠ ⊤ :=
    ENNReal.pow_ne_top (ne_top_of_le_ne_top ENNReal.one_ne_top one_sub_div_le_one)
  rw [Claims.Sampling.Pr_eq_toReal_prCoin]
  refine (ENNReal.toReal_mono hne (pooled_escape A M)).trans_eq ?_
  rw [ENNReal.toReal_pow, ENNReal.toReal_sub_of_le hle ENNReal.one_ne_top, ENNReal.toReal_one,
    ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

/-- **The pooled escape at `δ`.** With `q = kd/N`, an adaptive adversary escapes every period's draw with at least
`ln(1/δ)/q` wrong units over the monitoring period, pooled, with probability at most `δ`. -/
theorem pooled_escape_log {hk : kd ≤ N} {T : ℕ} (A : Pooled N kd hk T) (hq : 0 < kd) {δ : ℝ} (hδ : 0 < δ) :
    Claims.Pr (fun ω => A.Escapes ω ∧ Real.log (1 / δ) / ((kd : ℝ) / N) ≤ (A.total ω : ℝ)) ≤ δ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast hq.trans_le hk
  have hq0 : (0 : ℝ) < (kd : ℝ) / N := div_pos (by exact_mod_cast hq) hN
  have hq1 : (kd : ℝ) / N ≤ 1 := (div_le_one hN).2 (by exact_mod_cast hk)
  set q : ℝ := (kd : ℝ) / N
  set x : ℝ := Real.log (1 / δ) / q with hx
  calc Claims.Pr (fun ω => A.Escapes ω ∧ x ≤ (A.total ω : ℝ))
      ≤ Claims.Pr (fun ω => A.Escapes ω ∧ ⌈x⌉₊ ≤ A.total ω) :=
        Claims.Pr_mono fun ω h => ⟨h.1, Nat.ceil_le.2 h.2⟩
    _ ≤ (1 - q) ^ ⌈x⌉₊ := pooled_escape_Pr A _
    _ ≤ Real.exp (-q) ^ ⌈x⌉₊ :=
        pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-q)]) _
    _ = Real.exp (-q * ⌈x⌉₊) := by
        rw [← Real.exp_nat_mul, mul_comm]
    _ ≤ Real.exp (-q * x) := Real.exp_le_exp.2 (mul_le_mul_of_nonpos_left (Nat.le_ceil x) (by linarith))
    _ = δ := by
        rw [hx, neg_mul, mul_div_cancel₀ _ hq0.ne', one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

/-- `pooled_escape_log` for a wrong-set map on whole runs that doesn't look ahead. -/
theorem pooled_escape_log_adapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N))
    (hW : Periods.Predictable (Ω := fun _ => (subset N kd hk).Ω) W) (hq : 0 < kd) {δ : ℝ}
    (hδ : 0 < δ) :
    Claims.Pr (fun ω => (∀ t, Disjoint (ω t).1 (W t ω)) ∧
      Real.log (1 / δ) / ((kd : ℝ) / N) ≤ ((∑ t, (W t ω).card : ℕ) : ℝ)) ≤ δ := by
  have h := pooled_escape_log (Pooled.ofAdapted W) hq hδ
  simp only [Pooled.Escapes, Pooled.total, Pooled.wrong_ofAdapted hW] at h
  exact h

end Law

end FlockSoundness.Audit
