module

public import Definitions.Pous.Guarantees
public import Proofs.Pous.Chain.Flat
public import Proofs.Pous.Graded.Tail
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pous.Chain.Family
import Proofs.Pous.Graded.Theorem

/-! The proof of `Pous.Guarantees.BandGradedMeetsFamily`, whose deployment condition holds in the ideal-permutation
model only, not at the reference Feistel-SHAKE256 `Π₂`. -/

@[expose] public section

namespace PousGraded

open Pous Pous.BandMulti Pous.Sponge PousSpongeChain Pous.BandChain
open scoped ENNReal

theorem union_term_fam (N k : ℕ) (hN2 : 2 ≤ N) (hN : N ≤ 2 ^ 64) (hk : k ≤ 1000) :
    (∑ j ∈ Finset.range k, ((N * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg N (2 ^ 9) 524288 512 5 (2 ^ 20) (265616286 * N - 524289 - (73 * k + 129)) ≤
        (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have hN1 : 1 ≤ N := by omega
  have ha : N * 2 ^ 9 ≤ 2 ^ 73 :=
    calc N * 2 ^ 9 ≤ 2 ^ 64 * 2 ^ 9 := Nat.mul_le_mul_right _ hN
      _ = 2 ^ 73 := by rw [← pow_add]
  calc _ ≤ (2 : ℝ≥0∞) ^ (73 * k) * ((2⁻¹ : ℝ≥0∞) ^ (73 * k + 129) + (2⁻¹ : ℝ≥0∞) ^ (73 * k + 129)) :=
        mul_le_mul' (sum_pow_fam N 73 k hN1 ha)
          (chainPiSeg_fam_le N _ 390 316 (73 * k + 129) hN1 (base_fam_le N hN1 hN) (nPos_sq_fam_le N hN1 hN)
            (by omega) (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := two_pow_mul_two_inv _ _ _ (by omega)

theorem union_term_one (k : ℕ) (hk : k ≤ 1000) :
    (∑ j ∈ Finset.range k, ((1 * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg 1 (2 ^ 9) 524288 512 5 (2 ^ 20) (265254744 - (9 * k + 129)) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 :=
  calc _ ≤ (2 : ℝ≥0∞) ^ (9 * k) * ((2⁻¹ : ℝ≥0∞) ^ (9 * k + 129) + (2⁻¹ : ℝ≥0∞) ^ (9 * k + 129)) :=
        mul_le_mul' (sum_pow_fam 1 9 k le_rfl (by norm_num))
          (chainPiSeg_fam_le 1 _ 69 59 (9 * k + 129) le_rfl (by norm_num [nPos]) (by norm_num [nPos])
            (by omega) (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := two_pow_mul_two_inv _ _ _ (by omega)

theorem p₀_fam_le (N k S : ℕ) (hN2 : 2 ≤ N) (hk : k ≤ 300) (hS : (S : ℝ) ≤ 18 / 19 * ((N : ℝ) * 2 ^ 28)) :
    p₀ (265616286 * N - 524289 - (73 * k + 129)) S (N * 2 ^ 9) 524288 ≤ 95923 / 100000 := by
  have hNr : (2 : ℝ) ≤ N := by exact_mod_cast hN2
  have hkr : (k : ℝ) ≤ 300 := by exact_mod_cast hk
  have h1 : 524289 ≤ 265616286 * N := by omega
  have h2 : 73 * k + 129 ≤ 265616286 * N - 524289 := by omega
  unfold p₀
  rw [Nat.cast_sub h2, Nat.cast_sub h1]
  push_cast
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

theorem p₀_one_le (k S : ℕ) (hk : k ≤ 300) (hS : (S : ℝ) ≤ 18 / 19 * (((1 : ℕ) : ℝ) * 2 ^ 28)) :
    p₀ (265254744 - (9 * k + 129)) S (1 * 2 ^ 9) 524288 ≤ 95923 / 100000 := by
  have hkr : (k : ℝ) ≤ 300 := by exact_mod_cast hk
  have h1 : 9 * k + 129 ≤ 265254744 := by omega
  unfold p₀
  rw [Nat.cast_sub h1]
  push_cast at hS ⊢
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

theorem binTail_ofReal (k t : ℕ) {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    binTail k t (ENNReal.ofReal r) =
      ENNReal.ofReal (∑ j ∈ Finset.range (t + 1), (k.choose j : ℝ) * (1 - r) ^ j * r ^ (k - j)) := by
  have h1r : 0 ≤ 1 - r := by linarith
  unfold binTail
  rw [ENNReal.ofReal_sum_of_nonneg fun j _ =>
    mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg h1r _)) (pow_nonneg hr0 _)]
  refine Finset.sum_congr rfl fun j _ => ?_
  have h1 : (1 : ℝ≥0∞) - ENNReal.ofReal r = ENNReal.ofReal (1 - r) := by
    rw [ENNReal.ofReal_sub _ hr0, ENNReal.ofReal_one]
  rw [h1, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_pow h1r, ← ENNReal.ofReal_pow hr0,
    ← ENNReal.ofReal_mul (Nat.cast_nonneg _),
    ← ENNReal.ofReal_mul (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg h1r _))]

theorem binTail_le_δ {k t : ℕ}
    (h : ∑ j ∈ Finset.range (t + 1), (k.choose j : ℝ) * (1 - 95923 / 100000) ^ j * (95923 / 100000) ^ (k - j) ≤
      1 / 100) :
    binTail k t (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  rw [binTail_ofReal k t (by norm_num) (by norm_num)]
  exact ENNReal.ofReal_le_ofReal (by norm_num [Params.δ]; linarith)

theorem choose_203_2 : Nat.choose 203 2 = 20503 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem choose_243_2 : Nat.choose 243 2 = 29403 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem choose_243_3 : Nat.choose 243 3 = 2362041 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem binTail_111_0 : binTail 111 0 (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ])

theorem binTail_160_1 : binTail 160 1 (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ])

theorem binTail_203_2 : binTail 203 2 (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ, choose_203_2])

theorem binTail_243_3 : binTail 243 3 (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ, choose_243_2, choose_243_3])

theorem band_graded_fam_dom (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : 5 ≤ d) (dom : ℕ → Bits 512)
    (hdom : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i)) (D_late Q_late k t : ℕ) (hk : k ≤ 300)
    (hbin : binTail k t (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ)) :
    GradedMeets (chainBandSeg N (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20) D_late Q_late k t ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  have hN1 : 1 ≤ N := Nat.one_le_iff_ne_zero.2 (NeZero.ne N)
  have hM := band_meets_fam_dom chainExPostFactoG_v6 N hN d hd dom hdom
  refine ⟨hM.1, hM.2.1, hM.2.2.1, hM.2.2.2.1, ?_⟩
  intro W A₁ A₂
  have hS := stateBound_fam_le N d dom
  have hk' : k ≤ 1000 := hk.trans (by norm_num)
  have hr1 : ENNReal.ofReal (95923 / 100000 : ℝ) ≤ 1 := ENNReal.ofReal_le_one.2 (by norm_num)
  rcases Nat.lt_or_ge N 2 with h1 | h2
  · obtain rfl : N = 1 := by omega
    have hINC := band_seg_timedINC chainExPostFactoG_v6 1 (2 ^ 9) d 524288 512 5 (2 ^ 20)
      (265254744 - (9 * k + 129)) dom (by norm_num) hd (by norm_num) hdom
    refine (theorem1_ii_graded (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20) (265254744 - (9 * k + 129))
      ((chainBandSeg 1 (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ) k t _ _ hr1
      (ENNReal.ofReal_le_ofReal (p₀_one_le k _ hk hS)) hINC W A₁ A₂).trans ?_
    exact (add_comm _ _).le.trans (add_le_add hbin (union_term_one k hk'))
  · have hINC := band_seg_timedINC chainExPostFactoG_v6 N (2 ^ 9) d 524288 512 5 (2 ^ 20)
      (265616286 * N - 524289 - (73 * k + 129)) dom (by omega) hd (by norm_num) hdom
    refine (theorem1_ii_graded (chainBandSeg N (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20)
      (265616286 * N - 524289 - (73 * k + 129)) ((chainBandSeg N (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ)
      k t _ _ hr1 (ENNReal.ofReal_le_ofReal (p₀_fam_le N k _ h2 hk hS)) hINC W A₁ A₂).trans ?_
    exact (add_comm _ _).le.trans (add_le_add hbin (union_term_fam N k h2 hN hk'))

theorem band_graded_meets_family : Pous.Guarantees.BandGradedMeetsFamily := by
  intro N _ hN d hd D_late Q_late _ _ _ _ tk htk
  have hb : tk.2 ≤ 300 ∧
      binTail tk.2 tk.1 (ENNReal.ofReal (95923 / 100000)) ≤ ENNReal.ofReal (Params.δ : ℝ) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at htk
    rcases htk with rfl | rfl | rfl | rfl
    · exact ⟨by norm_num, binTail_111_0⟩
    · exact ⟨by norm_num, binTail_160_1⟩
    · exact ⟨by norm_num, binTail_203_2⟩
    · exact ⟨by norm_num, binTail_243_3⟩
  exact ⟨fun salt => band_graded_fam_dom N hN d hd _ (segDom_inj salt hN (by norm_num)) D_late Q_late _ _ hb.1 hb.2,
    fun salt => band_graded_fam_dom N hN d hd _
      (dom_inj salt (le_trans (Nat.mul_le_mul_right _ hN) (by norm_num))) D_late Q_late _ _ hb.1 hb.2⟩

end PousGraded
