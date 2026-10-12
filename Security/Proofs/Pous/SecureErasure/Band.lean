import Proofs.Pous.SecureErasure.Reduction
import Proofs.Pous.Chain.Family

/-!
# Secure erasure on the band

`Pous.Guarantees.SecureErasureMeetsFamily` and `SecureErasureMeets96GiB`. The band's sequential audit at `2152` reads
(`band_audit_2152`: `PousTargetsDense.theorem1_ii_seq` with the band's `TimedINC`, `band_seg_timedINC` with
`chainExPostFactoG_v6`), and `secure_erasure_meets` on top of it.

**The certificate at `s = 2152`**, `S = ⌊ρ·N·2^28⌋ ≤ (18/19)·N·2^28`, as `Security/Proofs/Pous/Chain/Family.lean`
does at `k = 111` but with the sampling term pushed to `2^-129`:
- `N ≥ 2`, `β = 265616286·N − 681516`: `β + 1 + 390·507N + 157227 = 524288·(507N − 1)`, so both terms of `chainPiSeg`
  are at most `2^-157227`, and `Σ_{j<2152} (N·2^9)^j ≤ 2^(73·2152)` leaves `|Hist|·π ≤ 2^-130`. `p₀ ≤ 9593/10000`
  (it needs `N ≥ 2`).
- `N = 1`, `β = 265235245`: `β + 1 + 69·507 + 19499 = 524288·506`, `Σ_{j<2152} 512^j ≤ 2^(9·2152)`, so
  `|Hist|·π ≤ 2^-130`; `p₀ ≤ 9593/10000` with about 2100 bits to spare.
- `(9593/10000)^2152 ≤ 2^-129`.
- Erasure: `S_φ ≥ (1/19)·|C| + 256` and `⌊ρ·|C|⌋ > ρ·|C| − 1` give `⌊ρ·|C|⌋ + S_φ ≥ |C| + 256`, so the foreign
  content's term is at most `2152 · 2^-256 ≤ 2^-130`.
-/

namespace PousErase

open Pous Pous.BandMulti Pous.Labelling Pous.Sponge PousSpongeChain Pous.BandChain
open scoped ENNReal

theorem two_inv_pow_succ_add_self (n : ℕ) :
    (2⁻¹ : ℝ≥0∞) ^ (n + 1) + (2⁻¹ : ℝ≥0∞) ^ (n + 1) = (2⁻¹ : ℝ≥0∞) ^ n := by
  rw [pow_succ, ← div_eq_mul_inv, ENNReal.add_halves]

/-! ## The sampling term -/

-- `norm_num` evaluates `(9593/10000)^2152`, past the default exponent threshold of 256.
set_option exponentiation.threshold 3000 in
theorem r_pow_2152 : (9593 / 10000 : ℝ) ^ 2152 ≤ (2⁻¹ : ℝ) ^ 129 := by norm_num

theorem sampling_2152 {p : ℝ} (hp : p ≤ 9593 / 10000) : ENNReal.ofReal p ^ 2152 ≤ (2⁻¹ : ℝ≥0∞) ^ 129 :=
  calc ENNReal.ofReal p ^ 2152 ≤ ENNReal.ofReal (9593 / 10000) ^ 2152 :=
        pow_le_pow_left' (ENNReal.ofReal_le_ofReal hp) _
    _ = ENNReal.ofReal ((9593 / 10000) ^ 2152) := (ENNReal.ofReal_pow (by norm_num) _).symm
    _ ≤ ENNReal.ofReal ((2⁻¹ : ℝ) ^ 129) := ENNReal.ofReal_le_ofReal r_pow_2152
    _ = (2⁻¹ : ℝ≥0∞) ^ 129 := by
        rw [ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]

/-- `p₀ ≤ 9593/10000` for `N ≥ 2` at `β = 265616286·N − 681516` and any `S ≤ (18/19)·N·2^28`. -/
theorem p₀_fam_le (N S : ℕ) (hN2 : 2 ≤ N) (hS : (S : ℝ) ≤ 18 / 19 * ((N : ℝ) * 2 ^ 28)) :
    p₀ (265616286 * N - 681516) S (N * 2 ^ 9) 524288 ≤ 9593 / 10000 := by
  have hNr : (2 : ℝ) ≤ N := by exact_mod_cast hN2
  unfold p₀
  rw [Nat.cast_sub (by omega)]
  push_cast
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  nlinarith

/-- `p₀ ≤ 9593/10000` for one segment at `β = 265235245` and any `S ≤ (18/19)·2^28`. -/
theorem p₀_one_le (S : ℕ) (hS : (S : ℝ) ≤ 18 / 19 * (((1 : ℕ) : ℝ) * 2 ^ 28)) :
    p₀ 265235245 S (1 * 2 ^ 9) 524288 ≤ 9593 / 10000 := by
  unfold p₀
  push_cast at hS ⊢
  rw [sub_le_comm, le_div_iff₀ (by positivity)]
  linarith

/-! ## The union term -/

/-- `|Hist|·π ≤ 2^-130` at `2152` reads for `2 ≤ N ≤ 2^64`. -/
theorem union_fam_2152 (N : ℕ) (hN2 : 2 ≤ N) (hN : N ≤ 2 ^ 64) :
    (∑ j ∈ Finset.range 2152, ((N * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg N (2 ^ 9) 524288 512 5 (2 ^ 20) (265616286 * N - 681516) ≤ (2⁻¹ : ℝ≥0∞) ^ 130 := by
  have hN1 : 1 ≤ N := by omega
  have ha : N * 2 ^ 9 ≤ 2 ^ 73 :=
    calc N * 2 ^ 9 ≤ 2 ^ 64 * 2 ^ 9 := Nat.mul_le_mul_right _ hN
      _ = 2 ^ 73 := by rw [← pow_add]
  calc _ ≤ (2 : ℝ≥0∞) ^ (73 * 2152) * ((2⁻¹ : ℝ≥0∞) ^ 157227 + (2⁻¹ : ℝ≥0∞) ^ 157227) :=
        mul_le_mul' (sum_pow_fam N 73 2152 hN1 ha)
          (chainPiSeg_fam_le N _ 390 316 157227 hN1 (base_fam_le N hN1 hN) (nPos_sq_fam_le N hN1 hN)
            (by omega) (by norm_num))
    _ = (2⁻¹ : ℝ≥0∞) ^ 130 := two_pow_mul_two_inv _ _ _ (by norm_num)

/-- `|Hist|·π ≤ 2^-130` at `2152` reads for one segment. -/
theorem union_one_2152 :
    (∑ j ∈ Finset.range 2152, ((1 * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      chainPiSeg 1 (2 ^ 9) 524288 512 5 (2 ^ 20) 265235245 ≤ (2⁻¹ : ℝ≥0∞) ^ 130 :=
  calc _ ≤ (2 : ℝ≥0∞) ^ (9 * 2152) * ((2⁻¹ : ℝ≥0∞) ^ 19499 + (2⁻¹ : ℝ≥0∞) ^ 19499) :=
        mul_le_mul' (sum_pow_fam 1 9 2152 le_rfl (by norm_num))
          (chainPiSeg_fam_le 1 _ 69 59 19499 le_rfl (by norm_num [nPos]) (by norm_num [nPos]) (by norm_num)
            (by norm_num))
    _ = (2⁻¹ : ℝ≥0∞) ^ 130 := two_pow_mul_two_inv _ _ _ (by norm_num)

/-! ## The audit at 2152 reads -/

/-- **The band's sequential audit at `2152` reads**: every `⌊ρ·|C|⌋`-bit prover is accepted with probability at most
`2^-130 + 2^-129`, for every domain injective on the `N·2^9` blocks and `1 ≤ N ≤ 2^64`. -/
theorem band_audit_2152 (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : 5 ≤ d) (dom : ℕ → Bits 512)
    (hdom : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i)) :
    AuditSecure (chainBandSeg N (2 ^ 9) d 524288 512 dom) .sequential
      ((chainBandSeg N (2 ^ 9) d 524288 512 dom).stateBound Params.ρ) 5 (2 ^ 20) 2152 0
      ((2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro W A₁ A₂
  rw [zero_add]
  have hS := stateBound_fam_le N d dom
  rcases Nat.lt_or_ge N 2 with h1 | h2
  · obtain rfl : N = 1 := by have := NeZero.ne N; omega
    have hINC := band_seg_timedINC chainExPostFactoG_v6 1 (2 ^ 9) d 524288 512 5 (2 ^ 20) 265235245 dom
      (by norm_num) hd (by norm_num) hdom
    refine (PousTargetsDense.theorem1_ii_seq (chainBandSeg 1 (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20) 265235245
      ((chainBandSeg 1 (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ) 2152 _ hINC W A₁ A₂).trans ?_
    exact add_le_add union_one_2152 (sampling_2152 (p₀_one_le _ hS))
  · have hINC := band_seg_timedINC chainExPostFactoG_v6 N (2 ^ 9) d 524288 512 5 (2 ^ 20) (265616286 * N - 681516)
      dom (by omega) hd (by norm_num) hdom
    refine (PousTargetsDense.theorem1_ii_seq (chainBandSeg N (2 ^ 9) d 524288 512 dom) 5 (2 ^ 20)
      (265616286 * N - 681516) ((chainBandSeg N (2 ^ 9) d 524288 512 dom).stateBound Pous.Params.ρ) 2152 _
      hINC W A₁ A₂).trans ?_
    exact add_le_add (union_fam_2152 N h2 hN) (sampling_2152 (p₀_fam_le N _ h2 hS))

/-! ## The foreign content's term -/

/-- `⌊ρ·L⌋ + S_φ ≥ L + 256` when `S_φ ≥ (1 − ρ)·L + 2λ`. -/
theorem floor_add_ge (L S_φ : ℕ)
    (h : (1 - Params.ρ) * (L : ℚ) + 2 * (Params.secParam : ℚ) ≤ (S_φ : ℚ)) :
    L + 256 ≤ ⌊Params.ρ * (L : ℚ)⌋₊ + S_φ := by
  have hfl := Nat.lt_floor_add_one (Params.ρ * (L : ℚ))
  have hlt : ((L + 256 : ℕ) : ℚ) < ((⌊Params.ρ * (L : ℚ)⌋₊ + S_φ + 1 : ℕ) : ℚ) := by
    unfold Params.secParam at h
    push_cast at h ⊢
    linarith
  have := (Nat.cast_lt (α := ℚ)).1 hlt
  omega

/-- The foreign content's term is at most `2^-130` once `S' + S_φ ≥ L + 256`. -/
theorem foreign_term_le (L S' S_φ : ℕ) (h : L + 256 ≤ S' + S_φ) :
    ((2152 : ℕ) : ℝ≥0∞) * 2 ^ L / (2 ^ S' * 2 ^ S_φ) ≤ (2⁻¹ : ℝ≥0∞) ^ 130 :=
  calc ((2152 : ℕ) : ℝ≥0∞) * 2 ^ L / (2 ^ S' * 2 ^ S_φ) ≤ (2 : ℝ≥0∞) ^ 12 * 2 ^ L / 2 ^ (L + 256) := by
        rw [← pow_add]
        gcongr
        · norm_num
        · exact one_le_two
    _ = (2⁻¹ : ℝ≥0∞) ^ 244 := by
        rw [← pow_add, div_eq_mul_inv, ENNReal.inv_pow, PousDenseChain.two_pow_mul_inv_pow _ _ (by omega)]
        congr 1
        omega
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 130 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

/-! ## The requirement -/

/-- **The band meets `EraseMeets`** at `f = 1/19`, `s = 2152`, `ε = 2^-128`, for every domain injective on its
`N·2^9` blocks and `1 ≤ N ≤ 2^64`. -/
theorem erase_meets_dom (N : ℕ) [NeZero N] (hN : N ≤ 2 ^ 64) (d : ℕ) (hd : 5 ≤ d) (dom : ℕ → Bits 512)
    (hdom : Function.Injective (fun i : Fin (N * 2 ^ 9) => dom i)) :
    EraseMeets (chainBandSeg N (2 ^ 9) d 524288 512 dom) (1 - Params.ρ) 5 (2 ^ 20) 2152
      ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  set sch := chainBandSeg N (2 ^ 9) d 524288 512 dom
  have haudit := band_audit_2152 N hN d hd dom hdom
  have hfloor : ⌊(1 - (1 - Params.ρ)) * ((sch.B * sch.ℓ : ℕ) : ℚ)⌋₊ = sch.stateBound Params.ρ := by
    rw [sub_sub_cancel]
    rfl
  have hsum : (2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130 = (2⁻¹ : ℝ≥0∞) ^ 128 := by
    rw [add_comm _ ((2⁻¹ : ℝ≥0∞) ^ 129), add_assoc, two_inv_pow_succ_add_self 129, two_inv_pow_succ_add_self 128]
  refine ⟨le_rfl, ?_, ?_⟩
  · rw [hfloor]
    intro W A₁ A₂
    refine (haudit W A₁ A₂).trans ?_
    simp only [zero_add]
    rw [← hsum]
    exact le_self_add
  · intro S_φ hS_φ K V _ _ dig W P
    refine (secure_erasure_meets sch K V dig (sch.B * sch.ℓ) (sch.stateBound Params.ρ) S_φ 5 (2 ^ 20) 2152 0 _
      haudit W P).trans ?_
    rw [zero_add, ← hsum]
    exact add_le_add_right (foreign_term_le _ _ _ (floor_add_ge _ _ hS_φ)) _

theorem secure_erasure_meets_family : Pous.Guarantees.SecureErasureMeetsFamily := fun N _ hN d hd =>
  ⟨fun salt => erase_meets_dom N hN d hd _ (segDom_inj salt hN (by norm_num)),
    fun salt => erase_meets_dom N hN d hd _ (dom_inj salt (le_trans (Nat.mul_le_mul_right _ hN) (by norm_num)))⟩

theorem secure_erasure_meets_96GiB : Pous.Guarantees.SecureErasureMeets96GiB := fun d hd salt =>
  (secure_erasure_meets_family 3072 (by norm_num) d hd).1 salt

end PousErase
