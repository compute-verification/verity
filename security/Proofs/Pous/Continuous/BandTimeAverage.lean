module

public import Proofs.Pous.Chain.FamilyRounds
public import Definitions.Pous.Guarantees
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pous.Chain.Family
import Proofs.Pous.Continuous.TimeAverage

/-! The proof of `Pous.Guarantees.BandFamilyContinuousTimeAverage`, the time-averaged bound on the band store. -/

@[expose] public section

namespace PousContinuous

open Pous Pous.BandChain Pous.BandMulti PousSpongeChain Pous.BandMulti.Rounds
open scoped ENNReal

def bandβ (N D k : ℕ) : ℕ :=
  if N = 1 then 524288 * (511 - D) - 69 * (512 - D) - 1 - (9 * k + 129) else 523898 * ((512 - D) * N) - 532588

theorem band_cont_tail_one (D k : ℕ) (hD : D ≤ 510) (hk : k ≤ 900) :
    (((βOne D k : ℕ) : ℝ≥0∞) + 1) * chainPiSeg 1 (2 ^ 9) 524288 512 D (2 ^ 20) (βOne D k) ≤
      (2⁻¹ : ℝ≥0∞) ^ (9 * k + 100) := by
  have hβ : βOne D k + 1 + 69 * (1 * 2 ^ 9 - 1 * D - 1 + 1) + (9 * k + 129) ≤ 524288 * (1 * 2 ^ 9 - 1 * D - 1) := by
    rw [show (1 : ℕ) * 2 ^ 9 = 512 by norm_num, Nat.one_mul]
    unfold βOne
    omega
  have hβ1 : ((βOne D k : ℕ) : ℝ≥0∞) + 1 ≤ 2 ^ 28 := by
    have h : βOne D k + 1 ≤ 2 ^ 28 := by
      rw [show (2 : ℕ) ^ 28 = 268435456 by norm_num]
      unfold βOne
      omega
    exact_mod_cast h
  calc _ ≤ (2 : ℝ≥0∞) ^ 28 * ((2⁻¹ : ℝ≥0∞) ^ (9 * k + 129) + (2⁻¹ : ℝ≥0∞) ^ (9 * k + 129)) :=
        mul_le_mul' hβ1 (chainPiSeg_rounds_le 1 D _ 69 59 (9 * k + 129) (by norm_num [Pous.Sponge.nPos])
          (by norm_num [Pous.Sponge.nPos]) hβ
          (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ (9 * k + 100) := two_pow_mul_two_inv _ _ _ (by omega)

theorem band_cont_tail_fam (N D : ℕ) (hN2 : 2 ≤ N) (hN : N ≤ 2 ^ 64) (hD : D ≤ 511) :
    (((523898 * ((512 - D) * N) - 532588 : ℕ) : ℝ≥0∞) + 1) *
      chainPiSeg N (2 ^ 9) 524288 512 D (2 ^ 20) (523898 * ((512 - D) * N) - 532588) ≤ (2⁻¹ : ℝ≥0∞) ^ 8206 := by
  have hN1 : 1 ≤ N := by omega
  have hm : 2 ≤ (512 - D) * N := le_trans (by norm_num) (Nat.mul_le_mul (by omega : 1 ≤ 512 - D) hN2)
  have hm' : (512 - D) * N ≤ 2 ^ 73 :=
    calc (512 - D) * N ≤ 512 * 2 ^ 64 := Nat.mul_le_mul (by omega) hN
      _ = 2 ^ 73 := by norm_num
  have hβ : 523898 * ((512 - D) * N) - 532588 + 1 + 390 * (N * 2 ^ 9 - N * D - 1 + 1) + 8299 ≤
      524288 * (N * 2 ^ 9 - N * D - 1) := by
    rw [seg_sub_rounds]
    generalize (512 - D) * N = m at hm ⊢
    omega
  have hβ1 : (((523898 * ((512 - D) * N) - 532588 : ℕ) : ℝ≥0∞) + 1) ≤ 2 ^ 92 := by
    have h : 523898 * ((512 - D) * N) - 532588 + 1 ≤ 2 ^ 92 := by
      generalize (512 - D) * N = m at hm' ⊢
      rw [show (2 : ℕ) ^ 73 = 9444732965739290427392 by norm_num] at hm'
      rw [show (2 : ℕ) ^ 92 = 4951760157141521099596496896 by norm_num]
      omega
    exact_mod_cast h
  calc _ ≤ (2 : ℝ≥0∞) ^ 92 * ((2⁻¹ : ℝ≥0∞) ^ 8299 + (2⁻¹ : ℝ≥0∞) ^ 8299) :=
        mul_le_mul' hβ1 (chainPiSeg_rounds_le N D _ 390 316 8299 (base_fam_le N hN1 hN) (nPos_sq_fam_le N hN1 hN) hβ
          (by norm_num))
    _ = (2⁻¹ : ℝ≥0∞) ^ 8206 := two_pow_mul_two_inv _ _ _ (by norm_num)

theorem band_cont_tail (N D k : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (hD : D ≤ 510) (hk : k ≤ 900) :
    (((bandβ N D k : ℕ) : ℝ≥0∞) + 1) * chainPiSeg N (2 ^ 9) 524288 512 D (2 ^ 20) (bandβ N D k) ≤
      (2⁻¹ : ℝ≥0∞) ^ (9 * k + 99) := by
  unfold bandβ
  split_ifs with h1
  · subst h1
    exact (band_cont_tail_one D k hD hk).trans (pow_le_pow_right_of_le_one' (by norm_num) (by omega))
  · have hN2 : 2 ≤ N := by
      have := NeZero.ne N
      omega
    exact (band_cont_tail_fam N D hN2 hN (by omega)).trans (pow_le_pow_right_of_le_one' (by norm_num) (by omega))

def BandContTimeAverageRow (D k : ℕ) : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, D ≤ d →
    ∀ dom : ℕ → Bits 512,
      ((∃ salt : Bits 192, dom = Pous.BandMulti.segDom salt (2 ^ 9)) ∨ ∃ salt : Bits 256, dom = Pous.Sponge.dom salt) →
    let sch := Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 dom
    let β := if N = 1 then 524288 * (511 - D) - 69 * (512 - D) - 1 - (9 * k + 129)
      else 523898 * ((512 - D) * N) - 532588
    ∀ (D_late Q_late T a b : ℕ), 0 < b → a ≤ b →
    ∀ (W : Bits sch.n) (mach : ContMachine sch) (f βe : ℝ), 0 < βe →
      pr (fun x : (Pous.Sponge.idealPerm (524288 + (524288 + 512))).Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) =>
          let pp := (sch.enc W x.1 x.2.1).1
          let C := (sch.enc W x.1 x.2.1).2
          let rs := contRecords mach D (2 ^ 20) D_late Q_late W pp C x.1 (arrAt a x.2.2) .start (histAt a x.2.2 T)
          contAccepts f rs ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              ((a : ℝ) / b) * ∑ t : Fin T,
                if contIdle mach D_late W pp C x.1 a x.2.2 t then
                  max (contFreed sch (contHeldAt mach D_late W pp C x.1 a x.2.2 t) - epsProof sch β) 0
                else 0)
        ≤ ENNReal.ofReal βe + T * (2⁻¹ : ℝ≥0∞) ^ (9 * k + 99)

theorem band_cont_time_average_row (D k : ℕ) (hD : D ≤ 510) (hk : k ≤ 900) : BandContTimeAverageRow D k := by
  intro N _ hN d hd dom hdom
  have hinj : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i) := by
    rcases hdom with ⟨salt, rfl⟩ | ⟨salt, rfl⟩
    · exact segDom_inj salt hN (by norm_num)
    · exact dom_inj salt (le_trans (Nat.mul_le_mul_right _ hN) (by norm_num))
  intro sch β D_late Q_late T a b hb hab W mach f βe hβe
  have hinc := band_seg_timedINC chainExPostFactoG_v6 N (2 ^ 9) d 524288 512 D (2 ^ 20) β dom
    (Nat.pos_of_ne_zero (NeZero.ne N)) hd (le_trans (Nat.succ_le_succ hD) (by norm_num)) hinj
  refine (continuous_time_average sch D (2 ^ 20) D_late Q_late β T a b _ hb hab hinc W mach f βe hβe).trans ?_
  rw [mul_assoc]
  exact add_le_add le_rfl (mul_le_mul' le_rfl (band_cont_tail N D k hN hD hk))

theorem band_family_continuous_time_average : Pous.Guarantees.BandFamilyContinuousTimeAverage := by
  intro N _ hN Dk hDk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hDk
  rcases hDk with rfl | rfl | rfl | rfl
  · exact band_cont_time_average_row 0 89 (by norm_num) (by norm_num) N hN
  · exact band_cont_time_average_row 1 93 (by norm_num) (by norm_num) N hN
  · exact band_cont_time_average_row 2 97 (by norm_num) (by norm_num) N hN
  · exact band_cont_time_average_row 5 111 (by norm_num) (by norm_num) N hN

theorem βOne_5_111 : βOne 5 111 = 265253616 := rfl

theorem band_epsProof_one (d : ℕ) (dom : ℕ → Bits 512) (β : ℕ) :
    epsProof (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) β = 1 - ((β : ℝ) - 512) / 2 ^ 28 := by
  show (1 : ℝ) - ((β : ℝ) - ((1 * 2 ^ 9 : ℕ) : ℝ)) / (((1 * 2 ^ 9 : ℕ) : ℝ) * ((524288 : ℕ) : ℝ)) = _
  norm_num

theorem band_epsProof_d0 (d : ℕ) (dom : ℕ → Bits 512) :
    epsProof (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) (βOne 0 89) = 561059 / 2 ^ 28 := by
  rw [band_epsProof_one, βOne_0_89]
  norm_num

theorem band_epsProof_d1 (d : ℕ) (dom : ℕ → Bits 512) :
    epsProof (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) (βOne 1 93) = 1085314 / 2 ^ 28 := by
  rw [band_epsProof_one, βOne_1_93]
  norm_num

theorem band_epsProof_d2 (d : ℕ) (dom : ℕ → Bits 512) :
    epsProof (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) (βOne 2 97) = 1609569 / 2 ^ 28 := by
  rw [band_epsProof_one, βOne_2_97]
  norm_num

theorem band_epsProof_d5 (d : ℕ) (dom : ℕ → Bits 512) :
    epsProof (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) (βOne 5 111) = 3182352 / 2 ^ 28 := by
  rw [band_epsProof_one, βOne_5_111]
  norm_num

end PousContinuous
