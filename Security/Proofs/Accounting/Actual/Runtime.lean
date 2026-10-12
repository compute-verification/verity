import Proofs.Accounting.Actual.Windows
import Proofs.Nci.Reads

/-!
# The runtime corollary

At a `t'` at or before `t` at which the developer's region implements `P`, the windows divide it into `n ⌈t'/ω⌉` IUs
of at most `Γ` gates, each reading at most `K` (`Windows`), so B5 (`Nci.SecurityProofs.UpdatedMatMulReads`) gives
`G f(K) ≤ n ⌈t'/ω⌉ K`, and `⌈t'/ω⌉ ω ≤ t' + ω`.
-/

namespace Accounting.SecurityProofs.Actual

open Verity Nci Finset

theorem exists_le_of_runtime_le {L : Type*} (A : Placed L) {X : Finset L} {P : Circuit} {t : ℕ}
    (h : A.runtime X P ≤ t) : ∃ t' ≤ t, Implements P (A.cut (A.region X (Set.Iio t'))) := by
  by_contra hne
  push Not at hne
  have hle : ((t + 1 : ℕ) : ℕ∞) ≤ A.runtime X P := le_iInf₂ fun t' ht' => by
    exact_mod_cast Nat.lt_of_not_le fun hle => hne t' hle ht'
  have := hle.trans h
  norm_cast at this
  omega

theorem Runtime : Guarantees.Actual.Runtime := by
  intro L E _ _ A ℓ Φ M ends β link n X θ hθ ω Γ K b_w b_a G d P hω hX hC hΦ hM hL hw ha hcap hchain t ht
  obtain ⟨t', ht', himp⟩ := exists_le_of_runtime_le A ht
  obtain ⟨c, hG, hd, hAI, hPS⟩ := hchain t' himp
  set m := (t' + ω - 1) / ω with hm_def
  obtain ⟨Q, hQ⟩ := Windows A ℓ Φ M ends β link n X θ hθ t' ω hω hX hC hΦ hM hL
  have hunit : ∀ u, (A.declaration (univ.biUnion X) t' θ hθ).D.size (Q.gates u) ≤ Γ ∧
      (novelReads (A.declaration (univ.biUnion X) t' θ hθ) (Q.gates u) : ℝ) ≤ K := by
    intro u
    obtain ⟨⟨i, k⟩, rfl⟩ := finProdFinEquiv.surjective u
    obtain ⟨-, hs, hr⟩ := hQ i k
    exact ⟨hs.trans (hcap i).1, (Nat.cast_le.2 hr).trans (hcap i).2⟩
  have hB5 := Nci.SecurityProofs.UpdatedMatMulReads Γ K b_w b_a _ c d _ Q hw ha hd hunit hAI hPS
  have hsum : ∑ u, (novelReads (A.declaration (univ.biUnion X) t' θ hθ) (Q.gates u) : ℝ) ≤ (n * m : ℕ) * K :=
    calc _ ≤ ∑ _u : Fin (n * m), K := sum_le_sum fun u _ => (hunit u).2
      _ = _ := by simp
  have hnK : 0 ≤ (n : ℝ) * K := by
    rcases Nat.eq_zero_or_pos n with hn | hn
    · simp [hn]
    · exact mul_nonneg (Nat.cast_nonneg _) ((Nat.cast_nonneg _).trans (hcap ⟨0, hn⟩).2)
  set f := readFactor K b_w b_a d
  have hGf : (G : ℝ) * f ≤ m * ((n : ℝ) * K) := by
    rcases le_or_gt 0 f with hf | hf
    · calc (G : ℝ) * f ≤ c.G * f := mul_le_mul_of_nonneg_right (Nat.cast_le.2 hG) hf
        _ ≤ _ := hB5.trans hsum
        _ = _ := by push_cast; ring
    · exact (mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) hf.le).trans
        (mul_nonneg (Nat.cast_nonneg _) hnK)
  have hdiv : (G : ℝ) * f / (n * K) ≤ m := div_le_of_le_mul₀ hnK (Nat.cast_nonneg _) hGf
  have hm : (m : ℝ) * ω ≤ t' + ω := by
    have := Nat.div_mul_le_self (t' + ω - 1) ω
    rw [← hm_def] at this
    exact_mod_cast (show m * ω ≤ t' + ω by omega)
  have ht'' : (t' : ℝ) ≤ t := Nat.cast_le.2 ht'
  have hω' : (0 : ℝ) ≤ ω := Nat.cast_nonneg _
  calc (ω : ℝ) * (G * f / (n * K) - 1) ≤ ω * (m - 1) := mul_le_mul_of_nonneg_left (by linarith) hω'
    _ ≤ t := by linarith

end Accounting.SecurityProofs.Actual
