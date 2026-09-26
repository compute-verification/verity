/-
Level 3, L3-F: the executable's GF(2^128) arithmetic (`Flock.F128`, PROTOCOL.md §2.1) is polynomial arithmetic over GF(2)
modulo the GHASH polynomial `G = X^128 + X^7 + X^2 + X + 1`. An element `⟨lo, hi⟩` is the polynomial of the bits of
`2^64·hi + lo`.
-/
import FlockLevel3.Bits
import Flock.Field

open Polynomial Flock

namespace FlockLevel3

/-- The 128-bit number `2^64·hi + lo`. -/
def w128 (lo hi : UInt64) : ℕ := 2 ^ 64 * hi.toNat + lo.toNat

noncomputable def poly64 (x : UInt64) : P := natPoly x.toNat

/-- The polynomial of an element of `F128`. -/
noncomputable def F128.poly (a : F128) : P := natPoly (w128 a.lo a.hi)

theorem natPoly_w128 (lo hi : UInt64) : natPoly (w128 lo hi) = X ^ 64 * poly64 hi + poly64 lo :=
  natPoly_split _ _ 64 lo.toNat_lt

theorem w128_lt (lo hi : UInt64) : w128 lo hi < 2 ^ 128 := by
  unfold w128; have := lo.toNat_lt; have := hi.toNat_lt; omega

theorem poly64_xor (x y : UInt64) : poly64 (x ^^^ y) = poly64 x + poly64 y := by
  simp [poly64, UInt64.toNat_xor, natPoly_xor]

theorem F128.poly_add (a b : F128) : F128.poly (a + b) = F128.poly a + F128.poly b := by
  show F128.poly ⟨a.lo ^^^ b.lo, a.hi ^^^ b.hi⟩ = _
  simp only [F128.poly, natPoly_w128, poly64_xor]
  ring

theorem toNat_toUInt64 (i : ℕ) (h : i < 64) : i.toUInt64.toNat = i := by
  simp [Nat.toUInt64, UInt64.toNat_ofNat]
  omega

/-- One step of the carry-less product: adding `a · x^i` (`i < 64`) into `(lo, hi)`. -/
theorem step128 (a lo hi : UInt64) (i : ℕ) (hi64 : i < 64) :
    w128 (lo ^^^ (a <<< i.toUInt64)) (if i = 0 then hi else hi ^^^ (a >>> (64 - i).toUInt64)) =
      w128 lo hi ^^^ (a.toNat <<< i) := by
  apply Nat.eq_of_testBit_eq
  intro j
  have hlt : ∀ x : UInt64, x.toNat < 2 ^ 64 := fun x => x.toNat_lt
  unfold w128
  rw [Nat.testBit_two_pow_mul_add _ (hlt _), Nat.testBit_xor, Nat.testBit_two_pow_mul_add _ (hlt _)]
  simp only [UInt64.toNat_xor, Nat.testBit_xor, UInt64.toNat_shiftLeft, Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft,
    toNat_toUInt64 i hi64, Nat.mod_eq_of_lt hi64]
  by_cases hj : j < 64
  · simp [hj]
  · simp only [hj, ite_false]
    by_cases hi0 : i = 0
    · subst hi0
      simp [Nat.testBit_lt_two_pow (lt_of_lt_of_le (hlt a) (Nat.pow_le_pow_right (by norm_num) (show 64 ≤ j by omega)))]
    · simp only [hi0, if_false, UInt64.toNat_xor, Nat.testBit_xor, UInt64.toNat_shiftRight,
        toNat_toUInt64 (64 - i) (by omega), Nat.mod_eq_of_lt (show 64 - i < 64 by omega), Nat.testBit_shiftRight]
      simp [show j ≥ i by omega, show 64 - i + (j - 64) = j - i by omega]

theorem natPoly_mod_succ (n f : ℕ) :
    natPoly (n % 2 ^ (f + 1)) = C (b2 (n.testBit 0)) + X * natPoly ((n >>> 1) % 2 ^ f) := by
  ext k
  rw [coeff_natPoly, coeff_add, coeff_C]
  rcases k with _ | k
  · simp [Nat.testBit_mod_two_pow]
  · rw [coeff_X_mul, coeff_natPoly, Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow, Nat.testBit_shiftRight]
    simp [show 1 + k = k + 1 by omega]

theorem and_one_eq_one (x : UInt64) : (x &&& 1 = 1) ↔ x.toNat.testBit 0 = true := by
  rw [UInt64.ext_iff, UInt64.toNat_and]
  simp [Nat.and_one_is_mod, Nat.testBit_zero]

/-- The carry-less product loop: after the remaining `fuel` bits of `bb`, from bit `i`. -/
theorem clmulGo_spec (a : UInt64) : ∀ (fuel i : ℕ) (bb lo hi : UInt64), i + fuel ≤ 64 →
    natPoly (w128 (F128.clmulGo a fuel i bb lo hi).1 (F128.clmulGo a fuel i bb lo hi).2) =
      natPoly (w128 lo hi) + X ^ i * poly64 a * natPoly (bb.toNat % 2 ^ fuel) := by
  intro fuel
  induction fuel with
  | zero => intro i bb lo hi _; simp [F128.clmulGo, Nat.mod_one]
  | succ f ih =>
    intro i bb lo hi hf
    have hshr : (bb >>> 1).toNat = bb.toNat >>> 1 := by simp [UInt64.toNat_shiftRight]
    rw [natPoly_mod_succ]
    by_cases hb : bb &&& 1 = 1
    · have hbit := (and_one_eq_one bb).mp hb
      simp only [F128.clmulGo, hb, ite_true]
      rw [ih (i + 1) _ _ _ (by omega), step128 a lo hi i (by omega), natPoly_xor, natPoly_shiftLeft, hshr, hbit]
      simp only [poly64, b2_true, map_one]
      ring
    · have hbit : bb.toNat.testBit 0 = false := by
        cases h : bb.toNat.testBit 0
        · rfl
        · exact absurd ((and_one_eq_one bb).mpr h) hb
      simp only [F128.clmulGo, hb, ite_false]
      rw [ih (i + 1) _ _ _ (by omega), hshr, hbit]
      simp only [b2_false, map_zero, zero_add]
      ring

theorem clmul_spec (a b : UInt64) :
    natPoly (w128 (F128.clmul a b).1 (F128.clmul a b).2) = poly64 a * poly64 b := by
  unfold F128.clmul
  rw [clmulGo_spec a 64 0 b 0 0 (by norm_num)]
  simp [w128, poly64, Nat.mod_eq_of_lt b.toNat_lt]

end FlockLevel3
