import Definitions.Core.Accel.Card
import Mathlib.Tactic

/-!
# The float codes' integers, as the over-approximated units read them

`CardWithinAccel` reads a register holding a float code of an exact integer as that integer's two's complement
(`Card.decode`, `Enc.dec`, `decInt`), and the over-approximated units read their operands back as integers: DFMA and
DMMA through `Lane.int53`, HFMA2 through `Lane.half11`. Here the two agree: a code `floatInt` reads as `n` is within
±2^(F+1) (`floatInt_natAbs_le`), so the decoded words read back as `n` through `int53` for FP64 (`int53_dec`) and
through `half11` for each half of an f16 or bf16 pair (`half11_f16`, `half11_bf16`).
-/

namespace Verity.Accel

theorem floatInt_natAbs_le {E F c : ℕ} {n : ℤ} (h : floatInt E F c = some n) : n.natAbs ≤ 2 ^ (F + 1) := by
  unfold floatInt at h
  simp only at h
  split_ifs at h with h0 h1 <;> cases h
  · simp
  all_goals
    simp only [Int.natAbs_mul, Int.natAbs_natCast, Int.natAbs_neg, Int.natAbs_one, one_mul]
    exact h1.2.2.2

theorem decInt_of {E F bits c : ℕ} {n : ℤ} (h : floatInt E F c = some n) :
    decInt E F bits c = (BitVec.ofInt bits n).toNat := by
  simp [decInt, h]

theorem decInt_lt {E F bits c : ℕ} (hc : c < 2 ^ bits) : decInt E F bits c < 2 ^ bits := by
  unfold decInt
  split
  · exact BitVec.isLt _
  · exact hc

theorem toInt_ofInt_small {bits : ℕ} {n : ℤ} (h : 2 * n.natAbs < 2 ^ bits) : (BitVec.ofInt bits n).toInt = n := by
  rw [BitVec.toInt_ofInt]
  zify at h
  have h1 := neg_abs_le n
  have h2 := le_abs_self n
  apply Int.bmod_eq_of_le_mul_two <;> push_cast <;> linarith

theorem int53_dec (lo hi : Word) {n : ℤ} (h : floatInt 11 52 (hi.toNat * 2 ^ 32 + lo.toNat) = some n) :
    Lane.int53 (Enc.f64lo.dec lo hi) (Enc.f64hi.dec hi lo) = n := by
  have hn := floatInt_natAbs_le h
  norm_num at hn
  simp only [Enc.dec, decInt_of h, Lane.int53]
  have hD := (BitVec.ofInt 64 n).isLt
  have hw : (BitVec.ofNat 32 ((BitVec.ofInt 64 n).toNat / 2 ^ 32)).toNat * 2 ^ 32 +
      (BitVec.ofNat 32 (BitVec.ofInt 64 n).toNat).toNat = (BitVec.ofInt 64 n).toNat := by
    simp only [BitVec.toNat_ofNat]
    omega
  rw [hw, BitVec.ofNat_toNat, BitVec.setWidth_eq, toInt_ofInt_small (by norm_num; omega), ite_eq_left_iff]
  intro h'
  norm_num at h'
  omega

theorem half11_of {x : Word} {o : ℕ} {n : ℤ} (hx : x.toNat / 2 ^ (16 * o) % 2 ^ 16 = (BitVec.ofInt 16 n).toNat)
    (hn : n.natAbs ≤ 2 ^ 11) : Lane.half11 x o = n := by
  simp only [Lane.half11]
  have hv : BitVec.ofNat 16 (x.toNat / 2 ^ (16 * o)) = BitVec.ofInt 16 n := by
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_ofNat, hx]
  rw [hv, toInt_ofInt_small (by omega), ite_eq_left_iff]
  intro h'
  omega

theorem half11_pair {E F : ℕ} (hF : F + 1 ≤ 11) (w : Word) (o : ℕ) (ho : o < 2) {n : ℤ}
    (h : floatInt E F (w.toNat / 2 ^ (16 * o) % 2 ^ 16) = some n) :
    Lane.half11 (BitVec.ofNat 32 (decInt E F 16 (w.toNat % 2 ^ 16) + decInt E F 16 (w.toNat / 2 ^ 16) * 2 ^ 16)) o =
      n := by
  have hn := (floatInt_natAbs_le h).trans (Nat.pow_le_pow_right (by norm_num) hF)
  refine half11_of ?_ hn
  have hw := w.isLt
  have hL := decInt_lt (E := E) (F := F) (show w.toNat % 2 ^ 16 < 2 ^ 16 from Nat.mod_lt _ (by norm_num))
  have hH := decInt_lt (E := E) (F := F) (show w.toNat / 2 ^ 16 % 2 ^ 16 < 2 ^ 16 from Nat.mod_lt _ (by norm_num))
  have hH' : w.toNat / 2 ^ 16 % 2 ^ 16 = w.toNat / 2 ^ 16 := Nat.mod_eq_of_lt (by omega)
  rw [hH'] at hH
  simp only [BitVec.toNat_ofNat]
  interval_cases o
  · simp only [Nat.mul_zero, pow_zero, Nat.div_one] at h ⊢
    rw [decInt_of h] at hL ⊢
    omega
  · simp only [Nat.mul_one] at h ⊢
    rw [hH'] at h
    rw [decInt_of h] at hH ⊢
    omega

theorem half11_f16 (w : Word) (o : ℕ) (ho : o < 2) {n : ℤ}
    (h : floatInt 5 10 (w.toNat / 2 ^ (16 * o) % 2 ^ 16) = some n) : Lane.half11 (Enc.f16x2.dec w w) o = n :=
  half11_pair (by norm_num) w o ho h

theorem half11_bf16 (w : Word) (o : ℕ) (ho : o < 2) {n : ℤ}
    (h : floatInt 8 7 (w.toNat / 2 ^ (16 * o) % 2 ^ 16) = some n) : Lane.half11 (Enc.bf16x2.dec w w) o = n :=
  half11_pair (by norm_num) w o ho h

end Verity.Accel
