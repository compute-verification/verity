import Proofs.Pous.Chain.Family

/-!
# The band store at fewer rounds: `D = 2` with `k = 97` and `D = 1` with `k = 93`

`Family.lean`'s certificate (`band_meets_fam_dom`, fixed at `D = 5`, `k = 111`) with the round budget `D` and the
challenge count `k` as parameters. The band's hardness (`band_seg_hard`), the glue (`band_seg_timedINC`) and
`theorem1_ii_seq` already hold for every `D ≤ d`; only the numbers change. `Security/Proofs/Pous.lean` states the
pinned forms, `BandMultiMeetsFamilyD2` and `BandMultiMeetsFamilyD1`.

**The certificate at `D` rounds.** `theorem1_ii_seq` at `S = ⌊ρ·N·2^28⌋ ≤ (18/19)·N·2^28` and loss
`s = N·2^9 − N·D − 1`, so `s + 1 = (512 − D)·N`:
- `N ≥ 2`, `β = 523898·(512 − D)·N − 532588 = (s + 1)·(524288 − 390) − 524288 − 8300`. As at `D = 5`, both terms of
  `chainPiSeg` are at most `2^-8299` (`nPos²·N·2^9·2 ≤ 2^390`, `nPos² ≤ 2^316`), and `Σ_{j<k} (N·2^9)^j ≤ 2^(73k)`
  leaves `|Hist|·π ≤ 2^-(8298 − 73k)`, at most `2^-128` for every `k ≤ 111`. `p₀ ≤ r` from
  `(1 − r)·2^28 + (18/19)·2^28 + 512 + 266294 ≤ 523898·(512 − D)` (the `266294` is `532588/2`, so it needs `N ≥ 2`).
- `N = 1`, `β = 524288·(511 − D) − 69·(512 − D) − 1 − (9k + 129)`: the freshness term is at most `2^-(9k + 129)`,
  `Σ_{j<k} 512^j ≤ 2^(9k)`, so `|Hist|·π ≤ 2^-128` with no slack. `p₀ ≤ r` from
  `(1 − r)·2^28 + (18/19)·2^28 + 512 ≤ β`.

| `D` | `k` | `β`, `N ≥ 2` | `p₀ ≤`, `N = 2` | `β`, `N = 1` | `p₀ ≤`, `N = 1` | `r` | `r^k` |
|---|---|---|---|---|---|---|---|
| 2 | 97 | `267187980·N − 532588` | 0.95301 | 266826399 | 0.95336 | 9535/10000 | ≈ 0.00987 |
| 1 | 93 | `267711878·N − 532588` | 0.95106 | 267350654 | 0.95141 | 9515/10000 | ≈ 0.00982 |

`N = 1` sets `k` at both points: `N ≥ 2` alone needs 96 at `D = 2` and 92 at `D = 1`.
-/

namespace Pous.BandMulti.Rounds

open Pous Pous.Labelling Pous.Sponge PousSpongeChain Pous.BandChain Pous.BandMulti
open scoped ENNReal

/-! ## The union term -/

/-- **The union `π` of `N` segments at `D` rounds is at most `2^-e` twice**, from `nPos²·N·2^9·2 ≤ 2^a` and
`nPos² ≤ 2^b`, for any exponent `e` with `β + 1 + a·(s + 1) + e ≤ 524288·s` (`s = N·2^9 − N·D − 1`) and
`e + b ≤ 524800`. -/
theorem chainPiSeg_rounds_le (N D β a b e : ℕ)
    (hbase : (nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 * (N * 2 ^ 9) * 2 ≤ 2 ^ a)
    (hnp : (nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 ≤ 2 ^ b)
    (hβ : β + 1 + a * (N * 2 ^ 9 - N * D - 1 + 1) + e ≤ 524288 * (N * 2 ^ 9 - N * D - 1))
    (he : e + b ≤ 524800) :
    chainPiSeg N (2 ^ 9) 524288 512 D (2 ^ 20) β ≤ (2⁻¹ : ℝ≥0∞) ^ e + (2⁻¹ : ℝ≥0∞) ^ e := by
  unfold chainPiSeg
  refine add_le_add ?_ ?_
  · generalize N * 2 ^ 9 - N * D - 1 = s at hβ ⊢
    have hβ1 : β + 1 + a * (s + 1) ≤ 524288 * s := by omega
    have hb : ((((nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 * (N * 2 ^ 9) * 2) ^ (s + 1) : ℕ) : ℝ≥0∞) ≤
        (2 : ℝ≥0∞) ^ (a * (s + 1)) := by
      have h := (Nat.cast_le (α := ℝ≥0∞)).2 (Nat.pow_le_pow_left hbase (s + 1))
      rwa [← pow_mul, Nat.cast_pow (2 : ℕ), Nat.cast_ofNat] at h
    calc (2 : ℝ≥0∞) ^ β * 2 *
            ((((nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 * (N * 2 ^ 9) * 2) ^ (s + 1) : ℕ) : ℝ≥0∞) / 2 ^ (524288 * s)
        ≤ (2 : ℝ≥0∞) ^ β * 2 * 2 ^ (a * (s + 1)) / 2 ^ (524288 * s) := by gcongr
      _ = (2⁻¹ : ℝ≥0∞) ^ (524288 * s - (β + 1 + a * (s + 1))) := by
          rw [div_eq_mul_inv, ENNReal.inv_pow, ← pow_succ, ← pow_add]
          exact PousDenseChain.two_pow_mul_inv_pow _ _ hβ1
      _ ≤ (2⁻¹ : ℝ≥0∞) ^ e := pow_le_pow_right_of_le_one' (by norm_num) (by omega)
  · have hE2 : 524288 + 512 = 524800 := by norm_num
    rw [hE2]
    have hb : (((nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 : ℕ) : ℝ≥0∞) ≤ (2 : ℝ≥0∞) ^ b := by
      have h := (Nat.cast_le (α := ℝ≥0∞)).2 hnp
      rwa [Nat.cast_pow (2 : ℕ) b, Nat.cast_ofNat] at h
    calc (((nPos (N * 2 ^ 9) (2 ^ 20)) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 524800
        ≤ (2 : ℝ≥0∞) ^ b / 2 ^ 524800 := by gcongr
      _ = (2⁻¹ : ℝ≥0∞) ^ (524800 - b) := by
          rw [div_eq_mul_inv, ENNReal.inv_pow]; exact PousDenseChain.two_pow_mul_inv_pow _ _ (by omega)
      _ ≤ (2⁻¹ : ℝ≥0∞) ^ e := pow_le_pow_right_of_le_one' (by norm_num) (by omega)

/-- `N·2^9 − N·D = (512 − D)·N`. -/
theorem seg_sub_rounds (N D : ℕ) : N * 2 ^ 9 - N * D = (512 - D) * N := by
  rw [Nat.sub_mul, Nat.mul_comm D N, show 512 * N = N * 2 ^ 9 by ring]

/-- **The union term for `2 ≤ N ≤ 2^64` at `D ≤ 511` rounds and `k ≤ 111` challenges**: `|Hist|·π ≤ 2^-128` at
`β = 523898·(512 − D)·N − 532588` (in fact `≤ 2^-(8298 − 73k)`). -/
theorem union_term_rounds (N D k : ℕ) (hN2 : 2 ≤ N) (hN : N ≤ 2 ^ 64) (hD : D ≤ 511) (hk : 73 * k + 129 ≤ 8299) :
    (∑ j ∈ Finset.range k, ((N * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg N (2 ^ 9) 524288 512 D (2 ^ 20) (523898 * ((512 - D) * N) - 532588) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have hN1 : 1 ≤ N := by omega
  have ha : N * 2 ^ 9 ≤ 2 ^ 73 :=
    calc N * 2 ^ 9 ≤ 2 ^ 64 * 2 ^ 9 := Nat.mul_le_mul_right _ hN
      _ = 2 ^ 73 := by rw [← pow_add]
  have hm : 2 ≤ (512 - D) * N := le_trans (by norm_num) (Nat.mul_le_mul (by omega : 1 ≤ 512 - D) hN2)
  have hβ : 523898 * ((512 - D) * N) - 532588 + 1 + 390 * (N * 2 ^ 9 - N * D - 1 + 1) + 8299 ≤
      524288 * (N * 2 ^ 9 - N * D - 1) := by
    rw [seg_sub_rounds]
    generalize (512 - D) * N = m at hm ⊢
    omega
  calc _ ≤ (2 : ℝ≥0∞) ^ (73 * k) * ((2⁻¹ : ℝ≥0∞) ^ 8299 + (2⁻¹ : ℝ≥0∞) ^ 8299) :=
        mul_le_mul' (sum_pow_fam N 73 k hN1 ha)
          (chainPiSeg_rounds_le N D _ 390 316 8299 (base_fam_le N hN1 hN) (nPos_sq_fam_le N hN1 hN) hβ
            (by norm_num))
    _ = (2⁻¹ : ℝ≥0∞) ^ (8298 - 73 * k) := two_pow_mul_two_inv _ _ _ (by omega)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by omega)

/-- The one-segment preprocessor width at `D` rounds and `k` challenges, with no slack in the union term. -/
def βOne (D k : ℕ) : ℕ := 524288 * (511 - D) - 69 * (512 - D) - 1 - (9 * k + 129)

/-- **The union term for one segment at `D ≤ 510` rounds and `k ≤ 1000` challenges**: `|Hist|·π ≤ 2^-128` at
`β = βOne D k`, exactly. -/
theorem union_term_one_rounds (D k : ℕ) (hD : D ≤ 510) (hk : k ≤ 1000) :
    (∑ j ∈ Finset.range k, ((1 * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg 1 (2 ^ 9) 524288 512 D (2 ^ 20) (βOne D k) ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have hβ : βOne D k + 1 + 69 * (1 * 2 ^ 9 - 1 * D - 1 + 1) + (9 * k + 129) ≤ 524288 * (1 * 2 ^ 9 - 1 * D - 1) := by
    rw [show (1 : ℕ) * 2 ^ 9 = 512 by norm_num, Nat.one_mul]
    unfold βOne
    omega
  calc _ ≤ (2 : ℝ≥0∞) ^ (9 * k) * ((2⁻¹ : ℝ≥0∞) ^ (9 * k + 129) + (2⁻¹ : ℝ≥0∞) ^ (9 * k + 129)) :=
        mul_le_mul' (sum_pow_fam 1 9 k le_rfl (by norm_num))
          (chainPiSeg_rounds_le 1 D _ 69 59 (9 * k + 129) (by norm_num [nPos]) (by norm_num [nPos]) hβ
            (by omega))
    _ = (2⁻¹ : ℝ≥0∞) ^ 128 := two_pow_mul_two_inv _ _ _ (by omega)

/-! ## The sampling term -/

/-- `ofReal p₀^k ≤ 1%` whenever `p₀ ≤ r` and `r^k ≤ 1/100`. -/
theorem sampling_of_le_k {p r : ℝ} (k : ℕ) (hp : p ≤ r) (hr0 : 0 ≤ r) (hr : r ^ k ≤ 1 / 100) :
    ENNReal.ofReal p ^ k ≤ ENNReal.ofReal (Params.δ : ℝ) :=
  calc ENNReal.ofReal p ^ k ≤ ENNReal.ofReal r ^ k := pow_le_pow_left' (ENNReal.ofReal_le_ofReal hp) _
    _ = ENNReal.ofReal (r ^ k) := (ENNReal.ofReal_pow hr0 _).symm
    _ ≤ ENNReal.ofReal (Params.δ : ℝ) := ENNReal.ofReal_le_ofReal (by norm_num [Params.δ]; linarith)

/-- **The sampling term for `N ≥ 2` at `D ≤ 511` rounds**: `p₀ ≤ r` at `β = 523898·(512 − D)·N − 532588` and any
`S ≤ (18/19)·N·2^28`, when `(1 − r)·2^28 + (18/19)·2^28 + 512 + 266294 ≤ 523898·(512 − D)`. -/
theorem sampling_term_rounds (N S D k : ℕ) (r : ℝ) (hN2 : 2 ≤ N) (hD : D ≤ 511)
    (hS : (S : ℝ) ≤ 18 / 19 * ((N : ℝ) * 2 ^ 28))
    (hr : (1 - r) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - D) : ℕ) : ℝ))
    (hr0 : 0 ≤ r) (hrk : r ^ k ≤ 1 / 100) :
    ENNReal.ofReal (p₀ (523898 * ((512 - D) * N) - 532588) S (N * 2 ^ 9) 524288) ^ k ≤
      ENNReal.ofReal (Params.δ : ℝ) := by
  refine sampling_of_le_k k ?_ hr0 hrk
  have hm1 : 1 ≤ 512 - D := by omega
  generalize 512 - D = m at hm1 hr ⊢
  have hNr : (2 : ℝ) ≤ N := by exact_mod_cast hN2
  have hmr : (1 : ℝ) ≤ m := by exact_mod_cast hm1
  have h2m : 2 ≤ m * N := le_trans (by norm_num) (Nat.mul_le_mul hm1 hN2)
  have hle : 532588 ≤ 523898 * (m * N) := by generalize m * N = t at h2m ⊢; omega
  unfold p₀
  rw [Nat.cast_sub hle]
  push_cast at hr ⊢
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hr (by positivity : (0 : ℝ) ≤ N)]

/-- **The sampling term for one segment**: `p₀ ≤ r` at `β` and any `S ≤ (18/19)·2^28`, when
`(1 − r)·2^28 + (18/19)·2^28 + 512 ≤ β`. -/
theorem sampling_term_one_rounds (β S k : ℕ) (r : ℝ) (hS : (S : ℝ) ≤ 18 / 19 * (((1 : ℕ) : ℝ) * 2 ^ 28))
    (hr : (1 - r) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ (β : ℝ)) (hr0 : 0 ≤ r) (hrk : r ^ k ≤ 1 / 100) :
    ENNReal.ofReal (p₀ β S (1 * 2 ^ 9) 524288) ^ k ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  refine sampling_of_le_k k ?_ hr0 hrk
  unfold p₀
  push_cast at hS ⊢
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

/-! ## The certificate -/

/-- **The `N`-segment band meets the requirement at `D` rounds and `k` challenges for every domain injective on its
`N·2^9` blocks**, given `ChainExPostFactoG`, for `1 ≤ N ≤ 2^64` and every `d ≥ D`, from the four numeric facts of
the certificate: `p₀ ≤ r₁` at one segment, `p₀ ≤ r₂` at `N ≥ 2`, and `r₁^k, r₂^k ≤ 1/100`. -/
theorem band_meets_rounds_dom (hepf : ChainExPostFactoG) (D k : ℕ) (r₁ r₂ : ℝ) (hD : D ≤ 510)
    (hk : 73 * k + 129 ≤ 8299)
    (h₁ : (1 - r₁) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ ((βOne D k : ℕ) : ℝ)) (hr₁0 : 0 ≤ r₁)
    (hr₁ : r₁ ^ k ≤ 1 / 100)
    (h₂ : (1 - r₂) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - D) : ℕ) : ℝ)) (hr₂0 : 0 ≤ r₂)
    (hr₂ : r₂ ^ k ≤ 1 / 100)
    (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : D ≤ d)
    (dom : ℕ → Bits 512) (hdom : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i)) :
    Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 dom) .sequential D (2 ^ 20) k ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  have hN1 : 1 ≤ N := Nat.one_le_iff_ne_zero.2 (NeZero.ne N)
  have hB : D + 1 ≤ 2 ^ 9 := le_trans (Nat.succ_le_succ hD) (by norm_num)
  refine ⟨le_rfl, chainScheme_correct _ _ _ _, le_rfl, ?_, ?_⟩
  · show ((((N * 2 ^ 9) * 524288 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * (((N * 2 ^ 9) * 524288 : ℕ) : ℚ)
    rw [Nat.add_zero]
    exact le_mul_of_one_le_left (Nat.cast_nonneg _) (by norm_num [Pous.Params.expansion])
  · intro W A₁ A₂
    have hS := stateBound_fam_le N d dom
    rcases Nat.lt_or_ge N 2 with h1 | h2
    · obtain rfl : N = 1 := by omega
      have hINC := band_seg_timedINC hepf 1 (2 ^ 9) d 524288 512 D (2 ^ 20) (βOne D k) dom (by norm_num) hd
        hB hdom
      refine (PousTargetsDense.theorem1_ii_seq (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) D (2 ^ 20) (βOne D k)
        ((chainBandSeg 1 (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ) k _ hINC W A₁ A₂).trans ?_
      exact (add_comm _ _).le.trans (add_le_add (sampling_term_one_rounds _ _ k r₁ hS h₁ hr₁0 hr₁)
        (union_term_one_rounds D k hD (by omega)))
    · have hINC := band_seg_timedINC hepf N (2 ^ 9) d 524288 512 D (2 ^ 20) (523898 * ((512 - D) * N) - 532588)
        dom (by omega) hd hB hdom
      refine (PousTargetsDense.theorem1_ii_seq (chainBandSeg N (2 ^ 9) d 524288 512 dom) D (2 ^ 20)
        (523898 * ((512 - D) * N) - 532588) ((chainBandSeg N (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ)
        k _ hINC W A₁ A₂).trans ?_
      exact (add_comm _ _).le.trans (add_le_add (sampling_term_rounds N _ D k r₂ h2 (by omega) hS h₂ hr₂0 hr₂)
        (union_term_rounds N D k h2 hN (by omega) hk))

/-- **The family at `D` rounds and `k` challenges, both domain layouts**, from the certificate's four numeric facts. -/
theorem band_meets_rounds (hepf : ChainExPostFactoG) (D k : ℕ) (r₁ r₂ : ℝ) (hD : D ≤ 510)
    (hk : 73 * k + 129 ≤ 8299)
    (h₁ : (1 - r₁) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ ((βOne D k : ℕ) : ℝ)) (hr₁0 : 0 ≤ r₁)
    (hr₁ : r₁ ^ k ≤ 1 / 100)
    (h₂ : (1 - r₂) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - D) : ℕ) : ℝ)) (hr₂0 : 0 ≤ r₂)
    (hr₂ : r₂ ^ k ≤ 1 / 100)
    (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : D ≤ d) :
    (∀ salt : Bits 192,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (segDom salt (2 ^ 9))) .sequential D (2 ^ 20) k
        ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential D (2 ^ 20) k
        ((2⁻¹ : ℝ≥0∞) ^ 128)) :=
  ⟨fun salt => band_meets_rounds_dom hepf D k r₁ r₂ hD hk h₁ hr₁0 hr₁ h₂ hr₂0 hr₂ N hN d hd _
      (segDom_inj salt hN (by norm_num)),
    fun salt => band_meets_rounds_dom hepf D k r₁ r₂ hD hk h₁ hr₁0 hr₁ h₂ hr₂0 hr₂ N hN d hd _
      (dom_inj salt (le_trans (Nat.mul_le_mul_right _ hN) (by norm_num)))⟩

/-! ## `D = 2`, `k = 97` and `D = 1`, `k = 93` -/

/-- The certificate's numbers at `D = 2`, `k = 97`, `r = 9535/10000`. -/
theorem βOne_2_97 : βOne 2 97 = 266826399 := rfl

theorem cert_d2_one : (1 - (9535 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ ((βOne 2 97 : ℕ) : ℝ) := by
  rw [βOne_2_97]; norm_num

theorem cert_d2_fam :
    (1 - (9535 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - 2) : ℕ) : ℝ) := by
  norm_num

theorem cert_d2_pow : (9535 / 10000 : ℝ) ^ 97 ≤ 1 / 100 := by norm_num

/-- The certificate's numbers at `D = 1`, `k = 93`, `r = 9515/10000`. -/
theorem βOne_1_93 : βOne 1 93 = 267350654 := rfl

theorem cert_d1_one : (1 - (9515 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ ((βOne 1 93 : ℕ) : ℝ) := by
  rw [βOne_1_93]; norm_num

theorem cert_d1_fam :
    (1 - (9515 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - 1) : ℕ) : ℝ) := by
  norm_num

theorem cert_d1_pow : (9515 / 10000 : ℝ) ^ 93 ≤ 1 / 100 := by norm_num

/-- **The band store at `D = 2`, `k = 97`**, every `1 ≤ N ≤ 2^64` and `d ≥ 2`, both domain layouts, unconditional
(`ChainExPostFactoG` discharged by `chainExPostFactoG_v6`). `r = 9535/10000` at both `N = 1` and `N ≥ 2`. -/
theorem band_meets_family_d2 : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 2 ≤ d →
    (∀ salt : Bits 192,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (segDom salt (2 ^ 9))) .sequential 2 (2 ^ 20) 97
        ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
        ((2⁻¹ : ℝ≥0∞) ^ 128)) := fun N _ hN d hd =>
  band_meets_rounds chainExPostFactoG_v6 2 97 _ _ (by norm_num) (by norm_num) cert_d2_one (by norm_num) cert_d2_pow
    cert_d2_fam (by norm_num) cert_d2_pow N hN d hd

/-- **The band store at `D = 1`, `k = 93`**, every `1 ≤ N ≤ 2^64` and `d ≥ 1`, both domain layouts, unconditional.
`r = 9515/10000` at both `N = 1` and `N ≥ 2`. -/
theorem band_meets_family_d1 : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 1 ≤ d →
    (∀ salt : Bits 192,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (segDom salt (2 ^ 9))) .sequential 1 (2 ^ 20) 93
        ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
        ((2⁻¹ : ℝ≥0∞) ^ 128)) := fun N _ hN d hd =>
  band_meets_rounds chainExPostFactoG_v6 1 93 _ _ (by norm_num) (by norm_num) cert_d1_one (by norm_num) cert_d1_pow
    cert_d1_fam (by norm_num) cert_d1_pow N hN d hd

/-- At `N = 1` the global layout at `D = 2` is the one-segment band `chainBand (2^9) d`. -/
theorem band_meets_64_d2 (d : ℕ) (hd : 2 ≤ d) (salt : Bits 256) :
    Pous.Meets (chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 2 (2 ^ 20) 97
      ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  have h := (band_meets_family_d2 1 (by norm_num) d hd).2 salt
  rwa [chainBandSeg_one] at h

/-- At `N = 1` the global layout at `D = 1` is the one-segment band `chainBand (2^9) d`. -/
theorem band_meets_64_d1 (d : ℕ) (hd : 1 ≤ d) (salt : Bits 256) :
    Pous.Meets (chainBand (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 1 (2 ^ 20) 93
      ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  have h := (band_meets_family_d1 1 (by norm_num) d hd).2 salt
  rwa [chainBandSeg_one] at h

/-! ## The band at `D = 0`, `k = 89`

At `D = 0` the responder makes no `Π₂` call and answers from its state alone, so the band needs no bound on its
in-degree (every `d ≥ 0`). The same recipe gives `k = 89`, which `N = 1` (`p₀ ≤ 0.949459`) and `N ≥ 2`
(`p₀ ≤ 0.949106` at `N = 2`) both need, with `r = 9495/10000` (`r^89 ≈ 0.00993`). `band_meets_family_d0` is
`BandMultiMeetsFamily` with `D = 0`, `k = 89` and no bound on `d`; it is `Pous.Guarantees.BandMultiMeetsFamilyD0`,
pinned (approved by Daniel, 3 Oct 2026). -/

/-- The certificate's numbers at `D = 0`, `k = 89`, `r = 9495/10000`. -/
theorem βOne_0_89 : βOne 0 89 = 267874909 := rfl

theorem cert_d0_one : (1 - (9495 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 ≤ ((βOne 0 89 : ℕ) : ℝ) := by
  rw [βOne_0_89]; norm_num

theorem cert_d0_fam :
    (1 - (9495 / 10000 : ℝ)) * 2 ^ 28 + 18 / 19 * 2 ^ 28 + 512 + 266294 ≤ ((523898 * (512 - 0) : ℕ) : ℝ) := by
  norm_num

theorem cert_d0_pow : (9495 / 10000 : ℝ) ^ 89 ≤ 1 / 100 := by norm_num

/-- **The band store at `D = 0` rounds and `k = 89`**, every `1 ≤ N ≤ 2^64` and every `d`, both domain layouts,
unconditional. `Pous.Guarantees.BandMultiMeetsFamilyD0`; here `Pous.SecurityProofs.BandMultiMeetsFamilyD0`. -/
theorem band_meets_family_d0 : ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ,
    (∀ salt : Bits 192,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (segDom salt (2 ^ 9))) .sequential 0 (2 ^ 20) 89
        ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      Pous.Meets (chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt)) .sequential 0 (2 ^ 20) 89
        ((2⁻¹ : ℝ≥0∞) ^ 128)) := fun N _ hN d =>
  band_meets_rounds chainExPostFactoG_v6 0 89 _ _ (by norm_num) (by norm_num) cert_d0_one (by norm_num) cert_d0_pow
    cert_d0_fam (by norm_num) cert_d0_pow N hN d (Nat.zero_le d)

end Pous.BandMulti.Rounds
