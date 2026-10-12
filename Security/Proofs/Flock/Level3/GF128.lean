/-
Level 3, L3-F: the executable's GF(2^128) arithmetic (`Flock.F128`, PROTOCOL.md §2.1) is polynomial arithmetic over GF(2)
modulo the GHASH polynomial `G = X^128 + X^7 + X^2 + X + 1`. An element `⟨lo, hi⟩` is the polynomial of the bits of
`2^64·hi + lo`.
-/
import Proofs.Flock.Level3.Bits
import Flock.Field
import Mathlib.Tactic.ComputeDegree
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Mathlib.Algebra.CharP.Basic

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

namespace FlockLevel3

/-- The GHASH polynomial's low part: `G = X^128 + S`. -/
noncomputable def S : P := X ^ 7 + X ^ 2 + X + 1
noncomputable def G : P := X ^ 128 + S

theorem S_degree_lt : S.degree < 128 := by
  unfold S; compute_degree!

theorem G_monic : G.Monic := by
  unfold G; exact Polynomial.monic_X_pow_add S_degree_lt

theorem G_degree : G.degree = 128 := by
  unfold G; rw [Polynomial.degree_add_eq_left_of_degree_lt (by rw [degree_X_pow]; exact S_degree_lt), degree_X_pow]; rfl

section reduce
variable (r2 r3 : UInt64)

/-- The words `reduce` computes from the top 128 bits `h = 2^64·r3 + r2`: `h·S` below bit 128 (`tl`, `th`) and above
(`o`). -/
def tl : UInt64 := r2 ^^^ (r2 <<< 1) ^^^ (r2 <<< 2) ^^^ (r2 <<< 7)
def th : UInt64 := r3 ^^^ ((r3 <<< 1) ||| (r2 >>> 63)) ^^^ ((r3 <<< 2) ||| (r2 >>> 62)) ^^^ ((r3 <<< 7) ||| (r2 >>> 57))
def ov : UInt64 := (r3 >>> 63) ^^^ (r3 >>> 62) ^^^ (r3 >>> 57)

/-- `h + h·x + h·x^2 + h·x^7` as bits. -/
def hS : ℕ := let h := w128 r2 r3; h ^^^ (h <<< 1) ^^^ (h <<< 2) ^^^ (h <<< 7)

theorem hi_false (x : UInt64) (k : ℕ) (hk : 64 ≤ k) : x.toNat.testBit k = false :=
  Nat.testBit_lt_two_pow (lt_of_lt_of_le x.toNat_lt (Nat.pow_le_pow_right (by norm_num) hk))

theorem h_bit (k : ℕ) : (w128 r2 r3).testBit k = if k < 64 then r2.toNat.testBit k else r3.toNat.testBit (k - 64) :=
  Nat.testBit_two_pow_mul_add _ r2.toNat_lt k

theorem tl_bit (j : ℕ) (hj : j < 64) : (tl r2).toNat.testBit j = (hS r2 r3).testBit j := by
  simp only [tl, hS, UInt64.toNat_xor, Nat.testBit_xor, UInt64.toNat_shiftLeft, Nat.testBit_mod_two_pow,
    Nat.testBit_shiftLeft, h_bit]
  simp [hj, show j - 1 < 64 by omega, show j - 2 < 64 by omega, show j - 7 < 64 by omega]

theorem th_bit (j : ℕ) (hj : j < 64) : (th r2 r3).toNat.testBit j = (hS r2 r3).testBit (64 + j) := by
  simp only [th, hS, UInt64.toNat_xor, Nat.testBit_xor, UInt64.toNat_shiftLeft, UInt64.toNat_or, Nat.testBit_or,
    Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, UInt64.toNat_shiftRight, Nat.testBit_shiftRight, h_bit]
  by_cases h7 : 7 ≤ j
  · simp [hj, h7, show 1 ≤ j by omega, show 2 ≤ j by omega, show 1 ≤ 64 + j by omega, show 2 ≤ 64 + j by omega,
      show 7 ≤ 64 + j by omega,
      show ¬ 63 + j < 64 by omega, show ¬ 62 + j < 64 by omega, show ¬ 57 + j < 64 by omega,
      show ¬ 64 + j - 1 < 64 by omega, show ¬ 64 + j - 2 < 64 by omega, show ¬ 64 + j - 7 < 64 by omega,
      show 63 + j - 64 = j - 1 by omega, show 62 + j - 64 = j - 2 by omega, show 57 + j - 64 = j - 7 by omega,
      show 64 + j - 1 - 64 = j - 1 by omega, show 64 + j - 2 - 64 = j - 2 by omega, show 64 + j - 7 - 64 = j - 7 by omega,
      hi_false r2 (63 + j) (by omega), hi_false r2 (62 + j) (by omega), hi_false r2 (57 + j) (by omega)]
  · interval_cases j <;> simp [hi_false r2 64 (by norm_num), hi_false r2 65 (by norm_num), hi_false r2 66 (by norm_num),
      hi_false r2 67 (by norm_num), hi_false r2 68 (by norm_num), hi_false r2 69 (by norm_num), hi_false r2 70 (by norm_num)]

theorem ov_bit (j : ℕ) : (ov r3).toNat.testBit j = (hS r2 r3).testBit (128 + j) := by
  simp only [ov, hS, UInt64.toNat_xor, Nat.testBit_xor, UInt64.toNat_shiftRight, Nat.testBit_shiftRight,
    Nat.testBit_shiftLeft, h_bit]
  simp [show 1 ≤ 128 + j by omega, show 2 ≤ 128 + j by omega, show 7 ≤ 128 + j by omega,
    show ¬ 127 + j < 64 by omega, show ¬ 126 + j < 64 by omega, show ¬ 121 + j < 64 by omega,
    show ¬ 128 + j - 1 < 64 by omega, show ¬ 128 + j - 2 < 64 by omega, show ¬ 128 + j - 7 < 64 by omega,
    show 127 + j - 64 = 63 + j by omega, show 126 + j - 64 = 62 + j by omega, show 121 + j - 64 = 57 + j by omega,
    show 128 + j - 1 - 64 = 63 + j by omega, show 128 + j - 2 - 64 = 62 + j by omega, show 128 + j - 7 - 64 = 57 + j by omega,
    show 128 + j - 64 = 64 + j by omega, show ¬ 128 + j < 64 by omega, hi_false r3 (64 + j) (by omega)]

theorem hS_split : hS r2 r3 = 2 ^ 128 * (ov r3).toNat + w128 (tl r2) (th r2 r3) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_two_pow_mul_add _ (w128_lt _ _), w128, Nat.testBit_two_pow_mul_add _ (UInt64.toNat_lt _)]
  by_cases h1 : i < 64
  · simp [h1, show i < 128 by omega, tl_bit r2 r3 i h1]
  · by_cases h2 : i < 128
    · simp [h1, h2, th_bit r2 r3 (i - 64) (by omega), show 64 + (i - 64) = i by omega]
    · simp [h2, ov_bit r2 r3 (i - 128), show 128 + (i - 128) = i by omega]

theorem natPoly_hS : natPoly (hS r2 r3) = natPoly (w128 r2 r3) * S := by
  simp only [hS, natPoly_xor, natPoly_shiftLeft, S]
  ring

theorem ov_lt : (ov r3).toNat < 2 ^ 7 := by
  have h : ∀ s, 57 ≤ s → r3.toNat >>> s < 2 ^ 7 := by
    intro s hs
    rw [Nat.shiftRight_eq_div_pow]
    apply Nat.div_lt_of_lt_mul
    calc r3.toNat < 2 ^ 64 := r3.toNat_lt
      _ ≤ 2 ^ s * 2 ^ 7 := by rw [← pow_add]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  simp only [ov, UInt64.toNat_xor, UInt64.toNat_shiftRight]
  exact Nat.xor_lt_two_pow (Nat.xor_lt_two_pow (h _ (by decide)) (h _ (by decide))) (h _ (by decide))

end reduce

end FlockLevel3

namespace FlockLevel3

theorem two_eq_zero_P : (2 : P) = 0 := by
  have h : ((2 : ℕ) : P) = 0 := CharP.cast_eq_zero P 2
  exact_mod_cast h

theorem shl_lit (x s : UInt64) (k : ℕ) (hs : s.toNat = k) (hk : k < 64) (hx : x.toNat < 2 ^ (64 - k)) :
    (x <<< s).toNat = x.toNat <<< k := by
  rw [UInt64.toNat_shiftLeft, hs, Nat.mod_eq_of_lt hk, Nat.shiftLeft_eq, Nat.mod_eq_of_lt]
  calc x.toNat * 2 ^ k < 2 ^ (64 - k) * 2 ^ k := Nat.mul_lt_mul_of_pos_right hx (by positivity)
    _ = 2 ^ 64 := by rw [← pow_add]; congr 1; omega

theorem poly_reduce (r0 r1 r2 r3 : UInt64) :
    F128.poly (F128.reduce r0 r1 r2 r3) =
      X ^ 64 * (poly64 r1 + poly64 (th r2 r3)) + poly64 r0 + poly64 (tl r2) + poly64 (ov r3) * S := by
  have ho := ov_lt r3
  have e1 : (ov r3 <<< 1).toNat = (ov r3).toNat <<< 1 := shl_lit _ _ 1 rfl (by norm_num) (by omega)
  have e2 : (ov r3 <<< 2).toNat = (ov r3).toNat <<< 2 := shl_lit _ _ 2 rfl (by norm_num) (by omega)
  have e7 : (ov r3 <<< 7).toNat = (ov r3).toNat <<< 7 := shl_lit _ _ 7 rfl (by norm_num) (by omega)
  show natPoly (w128 (r0 ^^^ tl r2 ^^^ ov r3 ^^^ (ov r3 <<< 1) ^^^ (ov r3 <<< 2) ^^^ (ov r3 <<< 7)) (r1 ^^^ th r2 r3)) = _
  rw [natPoly_w128]
  simp only [poly64, UInt64.toNat_xor, natPoly_xor, e1, e2, e7, natPoly_shiftLeft, S]
  ring

theorem natPoly_hS_split (r2 r3 : UInt64) :
    natPoly (w128 r2 r3) * S = X ^ 128 * poly64 (ov r3) + (X ^ 64 * poly64 (th r2 r3) + poly64 (tl r2)) := by
  rw [← natPoly_hS]
  have h := hS_split r2 r3
  have hw := w128_lt (tl r2) (th r2 r3)
  rw [h, natPoly_split _ _ 128 hw, natPoly_w128]
  simp only [poly64]

theorem reduce_spec (r0 r1 r2 r3 : UInt64) :
    F128.poly (F128.reduce r0 r1 r2 r3) =
      (poly64 r0 + X ^ 64 * poly64 r1 + X ^ 128 * poly64 r2 + X ^ 192 * poly64 r3) %ₘ G := by
  have hHS := natPoly_hS_split r2 r3
  have hdeg : (F128.poly (F128.reduce r0 r1 r2 r3)).degree < G.degree := by
    rw [G_degree]; exact natPoly_degree_lt _ 128 (w128_lt _ _)
  refine ((div_modByMonic_unique (natPoly (w128 r2 r3) + poly64 (ov r3)) _ G_monic ⟨?_, hdeg⟩).2).symm
  rw [poly_reduce, natPoly_w128] at *
  unfold G
  linear_combination (norm := skip) hHS
  ring_nf
  rw [two_eq_zero_P]
  ring

end FlockLevel3

namespace FlockLevel3

theorem clmul_spec' (a b : UInt64) :
    X ^ 64 * poly64 (F128.clmul a b).2 + poly64 (F128.clmul a b).1 = poly64 a * poly64 b := by
  rw [← clmul_spec, natPoly_w128]

theorem F128.poly_eq (a : F128) : F128.poly a = X ^ 64 * poly64 a.hi + poly64 a.lo := natPoly_w128 _ _

/-- **GF(2^128) multiplication (L3-F).** The executable's product is the product of polynomials modulo the GHASH
polynomial. -/
theorem F128.poly_mul (a b : F128) : F128.poly (a * b) = (F128.poly a * F128.poly b) %ₘ G := by
  show F128.poly (F128.mul a b) = _
  unfold F128.mul
  simp only
  rw [reduce_spec]
  congr 1
  have h0 := clmul_spec' a.lo b.lo
  have h2 := clmul_spec' a.hi b.hi
  have hm := clmul_spec' (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)
  simp only [poly64_xor] at *
  rw [F128.poly_eq, F128.poly_eq]
  linear_combination (norm := skip) h0 + X ^ 64 * h0 + X ^ 64 * hm + X ^ 64 * h2 + X ^ 128 * h2
  ring_nf
  rw [two_eq_zero_P]
  ring

theorem F128.poly_degree_lt (a : F128) : (F128.poly a).degree < 128 := natPoly_degree_lt _ 128 (w128_lt _ _)

theorem F128.poly_injective : Function.Injective F128.poly := by
  intro a b h
  have := natPoly_injective _ _ h
  unfold w128 at this
  have hl := a.lo.toNat_lt; have hh := a.hi.toNat_lt; have hl' := b.lo.toNat_lt; have hh' := b.hi.toNat_lt
  have e1 : a.lo.toNat = b.lo.toNat := by omega
  have e2 : a.hi.toNat = b.hi.toNat := by omega
  cases a; cases b
  simp only [F128.mk.injEq]
  exact ⟨UInt64.toNat_inj.mp e1, UInt64.toNat_inj.mp e2⟩

theorem F128.poly_zero : F128.poly 0 = 0 := by
  show natPoly (w128 0 0) = 0
  simp [w128]

theorem F128.poly_one : F128.poly 1 = 1 := by
  show natPoly (w128 1 0) = 1
  simp [w128, natPoly_one]

end FlockLevel3
