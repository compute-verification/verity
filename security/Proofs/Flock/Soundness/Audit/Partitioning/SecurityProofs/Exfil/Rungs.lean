module

public import Proofs.Flock.Soundness.Audit.Partitioning.Model
import Proofs.Flock.Soundness.Audit.SamplingDesign

/-! Exfiltration: what a certified leak forces. -/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil

open Finset
open scoped ENNReal

theorem sum_choose_pos (n K : ℕ) : 0 < ∑ j ∈ range (K + 1), n.choose j :=
  lt_of_lt_of_le (by simp) (Finset.single_le_sum (f := fun j => n.choose j) (fun _ _ => Nat.zero_le _)
    (Finset.mem_range.2 (Nat.succ_pos K)))

theorem locBits_mono (n : ℕ) {K K' : ℕ} (h : K ≤ K') : locBits n K ≤ locBits n K' := by
  unfold locBits
  apply Real.logb_le_logb_of_le (by norm_num) (by exact_mod_cast sum_choose_pos n K)
  exact_mod_cast Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 (by omega))

theorem miss_le_of_certifies {n b J : ℕ} {L : SamplingDesign n} {δ ℓ : ℝ} (hδ : 0 ≤ δ)
    (hL : CertifiesLeak L (fun _ => b) δ ℓ) (hℓ : ℓ < (b : ℝ) * J + locBits n J) :
    (L.miss J).toReal ≤ δ := by
  obtain ⟨H, K, hH, hK, hle⟩ := hL
  refine ENNReal.toReal_le_of_le_ofReal hδ (iSup₂_le fun B hB => ?_)
  by_contra hlt
  rw [not_le] at hlt
  have h1 : (B.card : ℝ) * b ≤ H := by simpa [Finset.sum_const, nsmul_eq_mul] using hH B hlt.le
  have h2 : B.card ≤ K := by
    by_contra h
    exact lt_irrefl _ ((hlt.trans_le (L.escape_le_miss (K := K + 1) (by omega))).trans hK)
  have h3 : (b : ℝ) * J ≤ (B.card : ℝ) * b := by
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast hB) (Nat.cast_nonneg b)
  have h4 := locBits_mono n (hB.trans h2)
  linarith

theorem choose_le_of_le_half {n j m : ℕ} (hjm : j ≤ m) (hm : m ≤ n / 2) : n.choose j ≤ n.choose m := by
  induction m, hjm using Nat.le_induction with
  | base => exact le_rfl
  | succ m _ ih => exact (ih (by omega)).trans (Nat.choose_le_succ_of_lt_half_left (by omega))

theorem locBits_succ_le {n K : ℕ} (hn : 2 * (K + 1) ≤ n) :
    locBits n (K + 1) ≤ Real.logb 2 ((K : ℝ) + 2) + Real.logb 2 (n.choose (K + 1)) := by
  have hc : 0 < n.choose (K + 1) := Nat.choose_pos (by omega)
  have hsum : ∑ j ∈ range (K + 1 + 1), n.choose j ≤ (K + 2) * n.choose (K + 1) := by
    calc ∑ j ∈ range (K + 1 + 1), n.choose j ≤ ∑ _j ∈ range (K + 1 + 1), n.choose (K + 1) :=
          Finset.sum_le_sum fun j hj => choose_le_of_le_half (by simp at hj; omega) (by omega)
      _ = (K + 2) * n.choose (K + 1) := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]
  unfold locBits
  rw [← Real.logb_mul (x := (K : ℝ) + 2) (y := (n.choose (K + 1) : ℝ)) (by positivity) (by exact_mod_cast hc.ne')]
  apply Real.logb_le_logb_of_le (by norm_num) (by exact_mod_cast sum_choose_pos n (K + 1))
  exact_mod_cast hsum

theorem pow_mul_choose_le (n r : ℕ) (hr : 1 ≤ r) : ∀ j, r ^ j * n.choose j ≤ (r * n).choose j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    rcases le_or_gt j n with hj | hj
    · have e1 := Nat.choose_succ_right_eq n j
      have e2 := Nat.choose_succ_right_eq (r * n) j
      have hf : r * (n - j) ≤ r * n - j := by
        refine Nat.le_sub_of_add_le ?_
        calc r * (n - j) + j ≤ r * (n - j) + r * j := Nat.add_le_add_left (Nat.le_mul_of_pos_left j hr) _
          _ = r * n := by rw [← Nat.mul_add, Nat.sub_add_cancel hj]
      refine Nat.le_of_mul_le_mul_right ?_ (Nat.succ_pos j)
      calc r ^ (j + 1) * n.choose (j + 1) * (j + 1) = r ^ j * r * (n.choose (j + 1) * (j + 1)) := by ring
        _ = r ^ j * n.choose j * (r * (n - j)) := by rw [e1]; ring
        _ ≤ (r * n).choose j * (r * n - j) := Nat.mul_le_mul ih hf
        _ = (r * n).choose (j + 1) * (j + 1) := e2.symm
    · rw [Nat.choose_eq_zero_of_lt (by omega : n < j + 1), mul_zero]
      exact Nat.zero_le _

theorem locBits_succ_ge {n r K : ℕ} (hr : 1 ≤ r) (hK : K + 1 ≤ n) :
    ((K : ℝ) + 1) * Real.logb 2 r + Real.logb 2 (n.choose (K + 1)) ≤ locBits (r * n) (K + 1) := by
  have hc : (0 : ℝ) < n.choose (K + 1) := by exact_mod_cast Nat.choose_pos hK
  have hr0 : (0 : ℝ) < r := by exact_mod_cast hr
  have h1 : ((K : ℝ) + 1) * Real.logb 2 r = Real.logb 2 ((r : ℝ) ^ (K + 1)) := by
    rw [Real.logb_pow, Nat.cast_add, Nat.cast_one]
  have h2 : r ^ (K + 1) * n.choose (K + 1) ≤ ∑ j ∈ range (K + 1 + 1), (r * n).choose j :=
    (pow_mul_choose_le n r hr (K + 1)).trans (Finset.single_le_sum (f := fun j => (r * n).choose j)
      (fun _ _ => Nat.zero_le _) (Finset.mem_range.2 (by omega)))
  unfold locBits
  rw [h1, ← Real.logb_mul (pow_pos hr0 _).ne' hc.ne']
  apply Real.logb_le_logb_of_le (by norm_num) (mul_pos (pow_pos hr0 _) hc)
  exact_mod_cast h2

end FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil
