import Pous.SecurityProofs.Graded.Theorem
import Pous.SecurityProofs.Chain.Family

/-!
# DRAFT: the band store with up to `t` late rounds (statement unreviewed and unpinned; proof checked)

**Status: the statements `ChainBandGradedMeetsSeg`, `ChainBandGradedMeetsSegGlobal` and `BandGradedMeetsFamily` are
draft trusted surface** over the draft game `GradedMeets` (`Graded/Game.lean`), for @lean's statement review and a red
team. Their proof, `band_graded_meets_family`, is checked here and assumes nothing: `ChainExPostFactoG` is discharged by
`chainExPostFactoG_v6`, as for `BandMultiMeetsFamily`.

**The point.** `BandMultiMeetsFamily`'s band: `N` segments (`1 ≤ N ≤ 2^64`) of `2^9` blocks of 64 KiB, in-degree
`d ≥ 5`, capacity 512, both salt layouts, `D = 5`, `Q = 2^20`, `ε = 2^-128`, `δ = 1%`, with up to `t` late rounds:

| `t` | `k` | `P[Bin(k, 0.9593) ≥ k − t]` |
|-----|-----|-----------------------------|
| 0   | 111 | 0.00993 |
| 1   | 161 | 0.00974 |
| 2   | 204 | 0.00978 |
| 3   | 244 | 0.00978 |

Each `k` is the least with the tail at most `1%` at `r = 9593/10000`, the one-segment bound on `p₀`. For `N ≥ 2`
alone, `p₀ ≤ 959/1000` and the least `k` are 160, 202, 242.

**The certificate** (`band_graded_fam_dom`), at any `k ≤ 1000`: `theorem1_ii_graded` at `S = ⌊ρ·N·2^28⌋`, with the
recipe of `Family.lean` and a union exponent that grows with `k`:
- `N ≥ 2`: `e = 73k + 129` and `β = 265616286·N − 524289 − e`, so `chainPiSeg ≤ 2·2^-e` (`chainPiSeg_fam_le` at
  `a = 390`, `b = 316`, tight), `Σ_{j<k} (N·2^9)^j ≤ 2^(73k)` and the union term is `2^-128`;
- `N = 1`: `e = 9k + 129` and `β = 265254744 − e` (`a = 69`, `b = 59`), the same way.

Lowering `β` by at most `64,830` bits keeps `p₀ ≤ 9593/10000` in both cases (`p₀_fam_le`, `p₀_one_le`), and
`binTail_mono` replaces `p₀` by `9593/10000`. At `t = 0` and `k = 111` the statement is `BandMultiMeetsFamily`'s, by
`gradedMeets_zero_iff`.
-/

namespace PousGraded

open Pous Pous.BandMulti Pous.Sponge PousSpongeChain Pous.BandChain
open scoped ENNReal

/-! ## Statements (DRAFT) -/

/-- DRAFT. The `N`-segment band under the implementation's per-segment tags, with `k` rounds of which up to `t` may be
late. -/
def ChainBandGradedMeetsSeg (N : ℕ) [NeZero N] (d k t : ℕ) : Prop :=
  ∀ salt : Bits 192,
    GradedMeets (chainBandSeg N (2 ^ 9) d 524288 512 (segDom salt (2 ^ 9))) 5 (2 ^ 20) k t ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- DRAFT. The same with one global salt and global indices. -/
def ChainBandGradedMeetsSegGlobal (N : ℕ) [NeZero N] (d k t : ℕ) : Prop :=
  ∀ salt : Bits 256,
    GradedMeets (chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) 5 (2 ^ 20) k t ((2⁻¹ : ℝ≥0∞) ^ 128)

/-- DRAFT. **The band with grace `t`**: for every `1 ≤ N ≤ 2^64`, `d ≥ 5` and `(t, k)` in the table, both layouts. -/
def BandGradedMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    ∀ tk ∈ ([(0, 111), (1, 161), (2, 204), (3, 244)] : List (ℕ × ℕ)),
      ChainBandGradedMeetsSeg N d tk.2 tk.1 ∧ ChainBandGradedMeetsSegGlobal N d tk.2 tk.1

/-! ## The union term at any `k ≤ 1000` -/

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

/-! ## The threshold at any `k ≤ 1000` -/

theorem p₀_fam_le (N k S : ℕ) (hN2 : 2 ≤ N) (hk : k ≤ 1000) (hS : (S : ℝ) ≤ 18 / 19 * ((N : ℝ) * 2 ^ 28)) :
    p₀ (265616286 * N - 524289 - (73 * k + 129)) S (N * 2 ^ 9) 524288 ≤ 9593 / 10000 := by
  have hNr : (2 : ℝ) ≤ N := by exact_mod_cast hN2
  have hkr : (k : ℝ) ≤ 1000 := by exact_mod_cast hk
  have h1 : 524289 ≤ 265616286 * N := by omega
  have h2 : 73 * k + 129 ≤ 265616286 * N - 524289 := by omega
  unfold p₀
  rw [Nat.cast_sub h2, Nat.cast_sub h1]
  push_cast
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

theorem p₀_one_le (k S : ℕ) (hk : k ≤ 1000) (hS : (S : ℝ) ≤ 18 / 19 * (((1 : ℕ) : ℝ) * 2 ^ 28)) :
    p₀ (265254744 - (9 * k + 129)) S (1 * 2 ^ 9) 524288 ≤ 9593 / 10000 := by
  have hkr : (k : ℝ) ≤ 1000 := by exact_mod_cast hk
  have h1 : 9 * k + 129 ≤ 265254744 := by omega
  unfold p₀
  rw [Nat.cast_sub h1]
  push_cast at hS ⊢
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

/-! ## The sampling term -/

/-- The tail at `ENNReal.ofReal r` is the real tail. -/
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
    (h : ∑ j ∈ Finset.range (t + 1), (k.choose j : ℝ) * (1 - 9593 / 10000) ^ j * (9593 / 10000) ^ (k - j) ≤
      1 / 100) :
    binTail k t (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  rw [binTail_ofReal k t (by norm_num) (by norm_num)]
  exact ENNReal.ofReal_le_ofReal (by norm_num [Params.δ]; linarith)

theorem choose_204_2 : Nat.choose 204 2 = 20706 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem choose_244_2 : Nat.choose 244 2 = 29646 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem choose_244_3 : Nat.choose 244 3 = 2391444 := by
  rw [Nat.choose_eq_descFactorial_div_factorial]; rfl

theorem binTail_111_0 : binTail 111 0 (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ])

theorem binTail_161_1 : binTail 161 1 (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ])

theorem binTail_204_2 : binTail 204 2 (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ, choose_204_2])

theorem binTail_244_3 : binTail 244 3 (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  binTail_le_δ (by norm_num [Finset.sum_range_succ, choose_244_2, choose_244_3])

/-! ## The certificate -/

/-- **The `N`-segment band with grace `t` at `k ≤ 1000` rounds, for every domain injective on its blocks**, whenever
`P[Bin(k, 9593/10000) ≥ k − t] ≤ 1%`. Unconditional. -/
theorem band_graded_fam_dom (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : 5 ≤ d) (dom : ℕ → Bits 512)
    (hdom : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i)) (k t : ℕ) (hk : k ≤ 1000)
    (hbin : binTail k t (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ)) :
    GradedMeets (chainBandSeg N (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20) k t ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  have hN1 : 1 ≤ N := Nat.one_le_iff_ne_zero.2 (NeZero.ne N)
  have hM := band_meets_fam_dom chainExPostFactoG_v6 N hN d hd dom hdom
  refine ⟨hM.1, hM.2.1, hM.2.2.1, hM.2.2.2.1, ?_⟩
  intro W A₁ A₂
  have hS := stateBound_fam_le N d dom
  have hr1 : ENNReal.ofReal (9593 / 10000 : ℝ) ≤ 1 := ENNReal.ofReal_le_one.2 (by norm_num)
  rcases Nat.lt_or_ge N 2 with h1 | h2
  · obtain rfl : N = 1 := by omega
    have hINC := band_seg_timedINC chainExPostFactoG_v6 1 (2 ^ 9) d 524288 512 5 (2 ^ 20)
      (265254744 - (9 * k + 129)) dom (by norm_num) hd (by norm_num) hdom
    refine (theorem1_ii_graded (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20) (265254744 - (9 * k + 129))
      ((chainBandSeg 1 (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ) k t _ _ hr1
      (ENNReal.ofReal_le_ofReal (p₀_one_le k _ hk hS)) hINC W A₁ A₂).trans ?_
    exact (add_comm _ _).le.trans (add_le_add hbin (union_term_one k hk))
  · have hINC := band_seg_timedINC chainExPostFactoG_v6 N (2 ^ 9) d 524288 512 5 (2 ^ 20)
      (265616286 * N - 524289 - (73 * k + 129)) dom (by omega) hd (by norm_num) hdom
    refine (theorem1_ii_graded (chainBandSeg N (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20)
      (265616286 * N - 524289 - (73 * k + 129)) ((chainBandSeg N (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ)
      k t _ _ hr1 (ENNReal.ofReal_le_ofReal (p₀_fam_le N k _ h2 hk hS)) hINC W A₁ A₂).trans ?_
    exact (add_comm _ _).le.trans (add_le_add hbin (union_term_fam N k h2 hN hk))

/-- **`BandGradedMeetsFamily`, proved** (unconditional). -/
theorem band_graded_meets_family : BandGradedMeetsFamily := by
  intro N _ hN d hd tk htk
  have hb : tk.2 ≤ 1000 ∧
      binTail tk.2 tk.1 (ENNReal.ofReal (9593 / 10000)) ≤ ENNReal.ofReal (Params.δ : ℝ) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at htk
    rcases htk with rfl | rfl | rfl | rfl
    · exact ⟨by norm_num, binTail_111_0⟩
    · exact ⟨by norm_num, binTail_161_1⟩
    · exact ⟨by norm_num, binTail_204_2⟩
    · exact ⟨by norm_num, binTail_244_3⟩
  exact ⟨fun salt => band_graded_fam_dom N hN d hd _ (segDom_inj salt hN (by norm_num)) _ _ hb.1 hb.2,
    fun salt => band_graded_fam_dom N hN d hd _
      (dom_inj salt (le_trans (Nat.mul_le_mul_right _ hN) (by norm_num))) _ _ hb.1 hb.2⟩

end PousGraded
