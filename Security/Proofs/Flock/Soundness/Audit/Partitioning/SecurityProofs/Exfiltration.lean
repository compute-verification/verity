import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.FreeCompute
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Rungs
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Bernoulli
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Witnesses

/-!
# Optimal partitioning against exfiltration: the proofs

The proofs of the exfiltration statements of `Partitioning.Guarantees`.

* `ExfiltrationOptimum`. A law certifying `ℓ` below the `(K+1)`-th rung has `miss (K + 1) ≤ δ`
  (`Exfil.miss_le_of_certifies`), so `ConverseExact` at `K + 1` gives the lower bound; when `K ≥ n` it is at most `0`.
  Independent draws just above `Λ/(K+1)` give the upper bound (`Exfil.exists_certifying_law`).
* `RefinementRule`. The unsplit side is `ExfiltrationOptimum`'s law at `ε / (c(s) + 1)`. On the split partition, the
  width condition and the counting of `Exfil.locBits_succ_le` and `Exfil.locBits_succ_ge` put `ℓ` below the `(K+1)`-th
  rung of `r·n` units of width `b'`, so `ConverseAsymptotic` at `K + 1` applies there, and
  `(r·n − K)·c(s/r) ≥ (n − K)·c(s)` since `r ≥ 1` and `f, o ≥ 0`.

The statements of their witnesses, `Witness.exfiltration` and `Witness.refinement`, are `Exfil.witness_exfiltration`
and `Exfil.witness_refinement`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs

open Finset

theorem ExfiltrationOptimum : Guarantees.ExfiltrationOptimum := by
  intro n b K δ ℓ hδ0 hδ1 hlo hhi
  refine ⟨fun L hL => ?_, fun ε hε => Exfil.exists_certifying_law n b K hδ0 hδ1 hlo hε⟩
  have hmiss := Exfil.miss_le_of_certifies (J := K + 1) hδ0.le hL (by push_cast; exact hhi)
  rcases le_or_gt (K + 1) n with hKn | hKn
  · have h := ConverseExact n (K + 1) L δ (Nat.succ_pos K) hKn hmiss
    push_cast at h
    linarith
  · have h1 : (n : ℝ) - K ≤ 0 := by
      have : (n : ℝ) ≤ K := by exact_mod_cast Nat.lt_succ_iff.1 hKn
      linarith
    have h2 : 0 ≤ 1 - δ ^ ((K + 1 : ℝ)⁻¹) := by
      have : δ ^ ((K + 1 : ℝ)⁻¹) ≤ 1 := Real.rpow_le_one hδ0.le hδ1 (by positivity)
      linarith
    exact (mul_nonpos_of_nonpos_of_nonneg h1 h2).trans (Exfil.meanDraws_nonneg L)

theorem RefinementRule : Guarantees.RefinementRule := by
  intro n r b b' K s f o δ ℓ hr hs hf ho hδ0 hδ1 hn hlo hhi hw
  have hr0 : (0 : ℝ) < r := by exact_mod_cast hr
  have hc : 0 ≤ unitCost f o s := add_nonneg hf (mul_nonneg ho hs.le)
  have hc' : 0 ≤ unitCost f o (s / r) := add_nonneg hf (mul_nonneg ho (div_nonneg hs.le hr0.le))
  have hK0 : (0 : ℝ) < K + 1 := by positivity
  refine ⟨fun ε hε => ?_, fun L' hL' => ?_⟩
  · obtain ⟨L, hL, hm⟩ := Exfil.exists_certifying_law n b K (ε := ε / (unitCost f o s + 1)) hδ0 hδ1 hlo
      (div_pos hε (by linarith))
    refine ⟨L, hL, ?_⟩
    rw [Exfil.cost_const]
    have hfrac : unitCost f o s / (unitCost f o s + 1) ≤ 1 := (div_le_one (by linarith)).2 (by linarith)
    calc meanDraws L * unitCost f o s
        ≤ (n * Real.log δ⁻¹ / (K + 1) + ε / (unitCost f o s + 1)) * unitCost f o s :=
          mul_le_mul_of_nonneg_right hm hc
      _ = unitCost f o s * (n * Real.log δ⁻¹ / (K + 1)) + ε * (unitCost f o s / (unitCost f o s + 1)) := by ring
      _ ≤ unitCost f o s * (n * Real.log δ⁻¹ / (K + 1)) + ε * 1 := by
          have := mul_le_mul_of_nonneg_left hfrac hε.le
          linarith
      _ = unitCost f o s * (n * Real.log δ⁻¹ / (K + 1)) + ε := by ring
  · have hKn : K + 1 ≤ n := by omega
    have hK1 : K + 1 ≤ r * n := hKn.trans (Nat.le_mul_of_pos_left n hr)
    -- `ℓ` is below the `(K+1)`-th rung of the split partition
    have hrung : ℓ < (b' : ℝ) * (K + 1 : ℕ) + locBits (r * n) (K + 1) := by
      have h1 := Exfil.locBits_succ_le hn
      have h2 := Exfil.locBits_succ_ge hr hKn
      have h3 : ((b : ℝ) + Real.logb 2 ((K : ℝ) + 2) / (K + 1)) * (K + 1) ≤
          ((b' : ℝ) + Real.logb 2 r) * (K + 1) := mul_le_mul_of_nonneg_right hw hK0.le
      have e : ((b : ℝ) + Real.logb 2 ((K : ℝ) + 2) / (K + 1)) * (K + 1) =
          b * (K + 1) + Real.logb 2 ((K : ℝ) + 2) := by field_simp
      rw [e] at h3
      push_cast
      linarith
    have hmiss := Exfil.miss_le_of_certifies hδ0.le hL' hrung
    have hA := ConverseAsymptotic (r * n) (K + 1) L' δ (Nat.succ_pos K) hK1 hδ0 hδ1 hmiss
    push_cast at hA
    have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
    have hnK : (K : ℝ) + 1 ≤ n := by exact_mod_cast hKn
    have hLHS : (1 - (K : ℝ) / n) * (1 - Real.log δ⁻¹ / (2 * (K + 1))) *
        (unitCost f o s * (n * Real.log δ⁻¹ / (K + 1))) =
        (n - K) * unitCost f o s * (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) := by
      field_simp
    have hA' : ((r : ℝ) * n - K) * (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) ≤
        meanDraws L' := by
      have e : ((r : ℝ) * n - (K + 1) + 1) / (K + 1) * Real.log δ⁻¹ * (1 - Real.log δ⁻¹ / (2 * (K + 1))) =
          ((r : ℝ) * n - K) * (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) := by
        field_simp
        ring
      linarith
    -- `(n − K)·c(s) ≤ (r·n − K)·c(s/r)`
    have hcmp : ((n : ℝ) - K) * unitCost f o s ≤ ((r : ℝ) * n - K) * unitCost f o (s / r) := by
      unfold unitCost
      have e : ((r : ℝ) * n - K) * (f + o * (s / r)) - (n - K) * (f + o * s) =
          (r - 1) * n * f + K * o * s * (r - 1) / r := by
        field_simp
        ring
      have hr1 : (0 : ℝ) ≤ r - 1 := by
        have : (1 : ℝ) ≤ r := by exact_mod_cast hr
        linarith
      have : 0 ≤ (r - 1) * n * f + K * o * s * (r - 1) / r :=
        add_nonneg (mul_nonneg (mul_nonneg hr1 hn0.le) hf)
          (div_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (Nat.cast_nonneg K) ho) hs.le) hr1) hr0.le)
      linarith
    rw [hLHS, Exfil.cost_const]
    rcases le_total 0 (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) with hy | hy
    · calc ((n : ℝ) - K) * unitCost f o s * (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1))))
          ≤ ((r : ℝ) * n - K) * unitCost f o (s / r) *
              (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) :=
            mul_le_mul_of_nonneg_right hcmp hy
        _ = ((r : ℝ) * n - K) * (Real.log δ⁻¹ / (K + 1) * (1 - Real.log δ⁻¹ / (2 * (K + 1)))) *
              unitCost f o (s / r) := by ring
        _ ≤ meanDraws L' * unitCost f o (s / r) := mul_le_mul_of_nonneg_right hA' hc'
    · have hnK' : (0 : ℝ) ≤ n - K := by linarith
      exact (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hnK' hc) hy).trans
        (mul_nonneg (Exfil.meanDraws_nonneg L') hc')

end FlockSoundness.Audit.Partitioning.SecurityProofs
