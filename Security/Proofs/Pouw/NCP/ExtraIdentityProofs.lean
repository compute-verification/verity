import Definitions.Pouw.NCP.ExtraIdentity
import Proofs.Pouw.NCP.GameProofs

/-!
# The admissible R with an extra checkpoint identity

`R` is an involution in blocks of 64: position `r < 32` of a block pairs with `32 + (r + 24) mod 32`, so every step's
image straddles two steps of the block's other half. The signs are `−1` on `[32, 64)` and `+1` elsewhere, so `R²` is
`−1` on `[0, 64)` and `+1` beyond: neither `I` nor `−I`. On `[32, 64)`, `R·P·R = −P_{[0,32)}`, which gives
`c_{k+64} = c_{k+32} + c_{32}` at zero input. `R` maps every prefix `[0, 64t)` onto itself and `R² = diag(s)`, so
`(I + R)·P_{64t}·(I − R) = P_{64t}·(I − R²) = 2·P_{64}`, and `c_{2k+64t} = 0` identically.
-/

namespace Pouw.NCP.ExtraIdentityProofs

open Finset Pouw.NCP Pouw.NCP.GameProofs

/-- The pairing, blockwise in blocks of 64. -/
def cexF (j : ℕ) : ℕ :=
  if j % 64 < 32 then j - j % 64 + 32 + (j % 64 + 24) % 32 else j - j % 64 + (j % 64 + 8) % 32

/-- `−1` on `[32, 64)`, `+1` elsewhere. -/
def cexSign (j : ℕ) : ℤ := if 32 ≤ j ∧ j < 64 then -1 else 1

/-- The counterexample `R`. -/
def cexR (k : ℕ) : Mat := permMat k cexF cexSign

theorem cexF_lt {k : ℕ} (h64 : 64 ∣ k) : ∀ j < k, cexF j < k := by
  obtain ⟨m, rfl⟩ := h64
  intro j hj
  unfold cexF
  split_ifs <;> omega

theorem cexF_invol (j : ℕ) : cexF (cexF j) = j := by
  unfold cexF
  split_ifs <;> omega

theorem cex_admissible {k : ℕ} (hk : 128 ≤ k) (h64 : 64 ∣ k) : Admissible k 16 (cexR k) := by
  have hlt := cexF_lt h64
  have hsq0 : mm k (cexR k) (cexR k) 0 0 = -1 := by
    rw [cexR, permMat_sq hlt (show 0 < k by omega), cexF_invol]
    simp [cexF, cexSign]
  have hsq64 : mm k (cexR k) (cexR k) 64 64 = 1 := by
    rw [cexR, permMat_sq hlt (show 64 < k by omega), cexF_invol]
    simp [cexF, cexSign]
  refine ⟨permMat_signed hlt (fun j _ => cexF_invol j) (fun j => ?_),
    fun t ht => ⟨fun j hj1 hj2 i hi1 hi2 => ?_, fun t' ht' h => ?_⟩, fun h => ?_, fun h => ?_⟩
  · unfold cexSign
    split_ifs <;> simp
  · simp only [cexR, permMat, cexF]
    split_ifs <;> omega
  · obtain ⟨m, rfl⟩ := h64
    obtain ⟨i1, hi1, hi1', hr1⟩ := h (t * 16 + 7) (by omega) (by omega)
    obtain ⟨i2, hi2, hi2', hr2⟩ := h (t * 16 + 8) (by omega) (by omega)
    simp only [cexR, permMat, cexF] at hr1 hr2
    split_ifs at hr1 hr2 <;> omega
  · have := h 0 (by omega) 0 (by omega)
    rw [hsq0] at this
    norm_num at this
  · have := h 64 (by omega) 64 (by omega)
    rw [hsq64] at this
    norm_num at this

theorem cexF_mid {m : ℕ} (hm : m < 32) : cexF (32 + m) = (m + 8) % 32 := by
  unfold cexF
  split_ifs <;> omega

theorem col {k : ℕ} (hk : 128 ≤ k) {m : ℕ} (hm : m < 32) (a : ℕ) :
    cexR k a (32 + m) = if a = (m + 8) % 32 then -1 else 0 := by
  simp only [cexR, permMat, cexF_mid hm, show 32 + m < k by omega, true_and, cexSign,
    show 32 ≤ 32 + m by omega, show 32 + m < 64 by omega, and_self, ite_true]

theorem row {k : ℕ} (hk : 128 ≤ k) {m : ℕ} (hm : m < 32) (b : ℕ) :
    cexR k (32 + m) b = if b = (m + 8) % 32 then 1 else 0 := by
  by_cases hb : b = (m + 8) % 32
  · subst hb
    have e : cexF ((m + 8) % 32) = 32 + m := by rw [← cexF_mid hm, cexF_invol]
    have h1 : ¬ (32 ≤ (m + 8) % 32 ∧ (m + 8) % 32 < 64) := by omega
    simp only [cexR, permMat, e, cexSign, h1, show (m + 8) % 32 < k by omega, and_self, ite_true, ite_false]
  · have : ¬ (32 + m = cexF b) := fun e => hb (by rw [← cexF_mid hm, e, cexF_invol])
    simp only [cexR, permMat, this, and_false, ite_false, hb]

theorem term_block1 {k : ℕ} (E1 F1 R : Mat) (i j l : ℕ) (hl : l < k) :
    stackX k 0 E1 R i l * stackY k 0 R F1 l j = E1 i l * F1 l j := by
  simp [stackX, stackY, hcat3, vcat3, hl]

theorem term_block2 {k : ℕ} (hk : 128 ≤ k) (E1 F1 : Mat) (i j m : ℕ) (hm : m < 32) :
    stackX k 0 E1 (cexR k) i (k + 32 + m) * stackY k 0 (cexR k) F1 (k + 32 + m) j =
      E1 i ((m + 8) % 32) * F1 ((m + 8) % 32) j := by
  have e1 : ¬ (k + 32 + m < k) := by omega
  have e2 : k + 32 + m < 2 * k := by omega
  have e3 : k + 32 + m - k = 32 + m := by omega
  have hc : (m + 8) % 32 < k := by omega
  have hX : stackX k 0 E1 (cexR k) i (k + 32 + m) = -E1 i ((m + 8) % 32) := by
    simp only [stackX, hcat3, e1, e2, e3, ite_false, ite_true, Pi.add_apply, Pi.zero_apply, zero_add, E2, mm,
      col hk hm, mul_ite, mul_neg, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_range, hc]
  have hY : stackY k 0 (cexR k) F1 (k + 32 + m) j = -F1 ((m + 8) % 32) j := by
    simp only [stackY, vcat3, e1, e2, e3, ite_false, ite_true, Pi.add_apply, Pi.zero_apply, zero_add, F2, mm,
      row hk hm, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_range, hc]
  rw [hX, hY, neg_mul_neg]

theorem relation {k : ℕ} (hk : 128 ≤ k) (E1 F1 : Mat) (i j : ℕ) :
    ∑ l ∈ range (k + 64), stackX k 0 E1 (cexR k) i l * stackY k 0 (cexR k) F1 l j =
      ∑ l ∈ range (k + 32), stackX k 0 E1 (cexR k) i l * stackY k 0 (cexR k) F1 l j +
        ∑ l ∈ range 32, stackX k 0 E1 (cexR k) i l * stackY k 0 (cexR k) F1 l j := by
  rw [show k + 64 = (k + 32) + 32 by omega, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_congr rfl fun m hm => term_block2 hk E1 F1 i j m (mem_range.1 hm),
    Finset.sum_congr rfl fun l hl => term_block1 E1 F1 (cexR k) i j l (by have := mem_range.1 hl; omega)]
  exact Finset.sum_nbij' (fun m => (m + 8) % 32) (fun l => (l + 24) % 32)
    (fun a ha => by simp only [mem_range] at ha ⊢; omega) (fun a ha => by simp only [mem_range] at ha ⊢; omega)
    (fun a ha => by simp only [mem_range] at ha; omega) (fun a ha => by simp only [mem_range] at ha; omega)
    (fun _ _ => rfl)

/-- The unit noise: `1` at `(0, 0)`, `0` elsewhere. -/
def unitNoise : Mat := fun a b => if a = 0 ∧ b = 0 then 1 else 0

theorem unitNoise_range (a b : ℕ) : InNoise6 (unitNoise a b) := by
  unfold unitNoise InNoise6
  split_ifs <;> norm_num

theorem nontrivial {k : ℕ} (hk : 128 ≤ k) :
    ∑ l ∈ range 32, stackX k 0 unitNoise (cexR k) 0 l * stackY k 0 (cexR k) unitNoise l 0 ≠ 0 := by
  rw [Finset.sum_congr rfl fun l hl => term_block1 unitNoise unitNoise (cexR k) 0 0 l
    (by have := mem_range.1 hl; omega)]
  simp [unitNoise]

theorem colK {k : ℕ} {m : ℕ} (hm : m < k) (a : ℕ) : cexR k a m = if a = cexF m then cexSign m else 0 := by
  simp only [cexR, permMat, hm, true_and]

theorem rowK {k : ℕ} (m : ℕ) {b : ℕ} (hb : b < k) : cexR k m b = if b = cexF m then cexSign b else 0 := by
  simp only [cexR, permMat, hb, true_and]
  by_cases h : b = cexF m
  · subst h
    simp [cexF_invol]
  · have : ¬ m = cexF b := fun e => h (by rw [e, cexF_invol])
    simp [h, this]

theorem e2_val {k : ℕ} (h64 : 64 ∣ k) (E1 : Mat) (i : ℕ) {m : ℕ} (hm : m < k) :
    mm k E1 (cexR k) i m = E1 i (cexF m) * cexSign m := by
  simp only [mm, colK hm, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_range, cexF_lt h64 m hm, ite_true]

theorem rf_val {k : ℕ} (h64 : 64 ∣ k) (F1 : Mat) (j : ℕ) {m : ℕ} (hm : m < k) :
    mm k (cexR k) F1 m j = cexSign (cexF m) * F1 (cexF m) j := by
  unfold mm
  rw [Finset.sum_congr rfl fun b hb => by rw [rowK m (mem_range.1 hb)]]
  simp only [ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_range, cexF_lt h64 m hm, ite_true]

theorem sign_prod {a : ℕ} (ha : 64 ≤ a) : cexSign (cexF a) * cexSign a = 1 := by
  unfold cexSign cexF
  split_ifs <;> omega

theorem zero_checkpoint {k : ℕ} (h64 : 64 ∣ k) {t : ℕ} (ht : 1 ≤ t) (htk : 64 * t ≤ k)
    (E1 F1 : Mat) (i j : ℕ) :
    ∑ l ∈ range (2 * k + 64 * t), stackX k 0 E1 (cexR k) i l * stackY k 0 (cexR k) F1 l j = 0 := by
  set f := cexF
  set σ := cexSign
  let e : ℕ → ℤ := fun a => E1 i a
  let g : ℕ → ℤ := fun a => F1 a j
  let W : ℕ → ℤ := fun a => (1 - σ (f a) * σ a) * (e a * g a)
  have hτ : 64 ∣ 64 * t := dvd_mul_right 64 t
  rw [show 2 * k + 64 * t = (k + k) + 64 * t by omega, Finset.sum_range_add, Finset.sum_range_add]
  have h1 : ∑ l ∈ range k, stackX k 0 E1 (cexR k) i l * stackY k 0 (cexR k) F1 l j = ∑ a ∈ range k, e a * g a :=
    Finset.sum_congr rfl fun l hl => term_block1 E1 F1 (cexR k) i j l (mem_range.1 hl)
  have h2 : ∑ m ∈ range k, stackX k 0 E1 (cexR k) i (k + m) * stackY k 0 (cexR k) F1 (k + m) j =
      ∑ a ∈ range k, -(σ (f a) * σ a * (e a * g a)) := by
    have hpt : ∀ m ∈ range k, stackX k 0 E1 (cexR k) i (k + m) * stackY k 0 (cexR k) F1 (k + m) j =
        e (f m) * σ m * -(σ (f m) * g (f m)) := fun m hm => by
      have hm' := mem_range.1 hm
      have e1 : ¬ (k + m < k) := by omega
      have e2 : k + m < 2 * k := by omega
      have e3 : k + m - k = m := by omega
      simp only [stackX, stackY, hcat3, vcat3, e1, e2, e3, ite_false, ite_true, Pi.add_apply, Pi.zero_apply,
        zero_add, E2, F2, e2_val h64 E1 i hm', rf_val h64 F1 j hm']
      rfl
    rw [Finset.sum_congr rfl hpt]
    exact Finset.sum_nbij' f f (fun a ha => mem_range.2 (cexF_lt h64 a (mem_range.1 ha)))
      (fun a ha => mem_range.2 (cexF_lt h64 a (mem_range.1 ha))) (fun a _ => cexF_invol a)
      (fun a _ => cexF_invol a) (fun a _ => by simp only [f, cexF_invol]; ring)
  have h3 : ∑ m ∈ range (64 * t), stackX k 0 E1 (cexR k) i (k + k + m) * stackY k 0 (cexR k) F1 (k + k + m) j =
      -∑ a ∈ range (64 * t), W a := by
    have hpt : ∀ m ∈ range (64 * t),
        stackX k 0 E1 (cexR k) i (k + k + m) * stackY k 0 (cexR k) F1 (k + k + m) j =
          -(e m * g m) + σ (f m) * (e m * g (f m)) - σ m * (e (f m) * g m) +
            σ m * σ (f m) * (e (f m) * g (f m)) := fun m hm => by
      have hm' : m < k := by have := mem_range.1 hm; omega
      have e1 : ¬ (k + k + m < k) := by omega
      have e2 : ¬ (k + k + m < 2 * k) := by omega
      have e3 : k + k + m - 2 * k = m := by omega
      simp only [stackX, stackY, hcat3, vcat3, e1, e2, e3, ite_false, Pi.add_apply, Pi.zero_apply, Pi.neg_apply,
        zero_add, E3, F3, E2, F2, e2_val h64 E1 i hm', rf_val h64 F1 j hm']
      simp only [e, g]
      ring
    rw [Finset.sum_congr rfl hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_add_distrib]
    have hcross : ∑ m ∈ range (64 * t), σ m * (e (f m) * g m) = ∑ m ∈ range (64 * t), σ (f m) * (e m * g (f m)) :=
      Finset.sum_nbij' f f (fun a ha => mem_range.2 (cexF_lt hτ a (mem_range.1 ha)))
        (fun a ha => mem_range.2 (cexF_lt hτ a (mem_range.1 ha))) (fun a _ => cexF_invol a)
        (fun a _ => cexF_invol a) (fun a _ => by simp only [f, cexF_invol])
    have hlast : ∑ m ∈ range (64 * t), σ m * σ (f m) * (e (f m) * g (f m)) =
        ∑ a ∈ range (64 * t), σ (f a) * σ a * (e a * g a) :=
      Finset.sum_nbij' f f (fun a ha => mem_range.2 (cexF_lt hτ a (mem_range.1 ha)))
        (fun a ha => mem_range.2 (cexF_lt hτ a (mem_range.1 ha))) (fun a _ => cexF_invol a)
        (fun a _ => cexF_invol a) (fun a _ => by simp only [f, cexF_invol])
    rw [hcross, hlast, Finset.sum_neg_distrib]
    have : ∑ a ∈ range (64 * t), W a = ∑ a ∈ range (64 * t), (e a * g a - σ (f a) * σ a * (e a * g a)) :=
      Finset.sum_congr rfl fun a _ => by simp only [W]; ring
    rw [this, Finset.sum_sub_distrib]
    ring
  have hsub : ∑ a ∈ range k, W a = ∑ a ∈ range (64 * t), W a := by
    refine (Finset.sum_subset (Finset.range_mono htk) fun a ha hna => ?_).symm
    have ha64 : 64 ≤ a := by simp only [mem_range] at ha hna; omega
    simp only [W, σ, f, sign_prod ha64, sub_self, zero_mul]
  rw [h1, h2, h3, ← hsub]
  have : ∑ a ∈ range k, e a * g a + ∑ a ∈ range k, -(σ (f a) * σ a * (e a * g a)) = ∑ a ∈ range k, W a := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun a _ => by simp only [W]; ring
  rw [this, add_neg_cancel]

/-- **The finding: an admissible R with a checkpoint identity other than the final one.** -/
theorem admissibleExtraIdentity : ExtraIdentity.AdmissibleExtraIdentity := fun k hk h64 =>
  ⟨cexR k, cex_admissible hk h64, relation hk,
    ⟨unitNoise, unitNoise, unitNoise_range, unitNoise_range, nontrivial hk⟩,
    fun _ ht htk E1 F1 i j => zero_checkpoint h64 ht htk E1 F1 i j⟩

end Pouw.NCP.ExtraIdentityProofs
