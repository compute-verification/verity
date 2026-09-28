/-
Level 3, L3-F: the GHASH polynomial `G` is irreducible, so `F128` is the field GF(2^128) (`GF128`, the field structure
on the executable's representation, whose `+`, `*` and inverse are the executable's `add`, `mul` and `inv`). The certificate is Rabin's test,
checked by the kernel on the executable's own multiplication:
- `x^(2^128) = x` (128 squarings of `x`, `decide +kernel`), so every irreducible factor of `G` has degree dividing 128;
- `x^(2^64) + x` has an inverse `w` (`w · (x^(2^64) + x) = 1`, 64 squarings and one multiplication), so no irreducible
  factor of degree dividing 64 divides `G`.
The only divisor of 128 that does not divide 64 is 128 itself.
-/
import Proofs.Flock.Level3.GF128Ring
import Mathlib.FieldTheory.Finite.Extension
import Mathlib.Algebra.Field.TransferInstance

open Polynomial Flock

namespace FlockLevel3

/-- `a^(2^k)` by `k` squarings, with the executable's multiplication. -/
def sqIter : ℕ → F128 → F128
  | 0, a => a
  | k + 1, a => sqIter k (a * a)

theorem toAdj_sqIter (k : ℕ) (a : F128) : toAdj (sqIter k a) = toAdj a ^ (2 ^ k) := by
  induction k generalizing a with
  | zero => simp [sqIter]
  | succ k ih => rw [sqIter, ih, toAdj_mul, ← sq, ← pow_mul, pow_succ']

theorem monomial_one_eq : F128.monomial 1 = ⟨2, 0⟩ := by decide +kernel

theorem toAdj_x : toAdj (F128.monomial 1) = AdjoinRoot.root G := by
  rw [monomial_one_eq, toAdj, F128.poly]
  have hw : w128 2 0 = 1 <<< 1 := by decide +kernel
  rw [hw, natPoly_shiftLeft, natPoly_one, pow_one, mul_one, AdjoinRoot.mk_X]

theorem sqIter_128_x : sqIter (2 ^ (6 + 1)) (F128.monomial 1) = F128.monomial 1 := by decide +kernel

theorem root_frob : AdjoinRoot.root G ^ (2 ^ (2 ^ (6 + 1))) = AdjoinRoot.root G := by
  rw [← toAdj_x, ← toAdj_sqIter, sqIter_128_x]

/- The facts below are stated for a generic exponent and instantiated at the end: rewriting inside a term that holds
`X ^ 2^128` itself makes the elaborator and the kernel unfold the power. -/

theorem mk_X_pow (n : ℕ) : AdjoinRoot.mk G (X ^ n) = AdjoinRoot.root G ^ n := by
  rw [map_pow, AdjoinRoot.mk_X]

theorem dvd_of_root_pow (n : ℕ) (h : AdjoinRoot.root G ^ n = AdjoinRoot.root G) : G ∣ X ^ n - X := by
  apply AdjoinRoot.mk_eq_mk.mp
  rw [mk_X_pow, h, AdjoinRoot.mk_X]

theorem G_dvd_frob : G ∣ X ^ (2 ^ (2 ^ (6 + 1))) - X := dvd_of_root_pow _ root_frob

/-- `(x^(2^64) + x)⁻¹`, computed once by the executable's `F128.inv` and checked below by one multiplication. -/
def bezoutW : F128 := ⟨0x41a8cafd2c95b018, 0x9e4af928ddbc838a⟩

theorem bezout_exec : bezoutW * (sqIter (2 ^ 6) (F128.monomial 1) + F128.monomial 1) = 1 := by decide +kernel

theorem adj_add_self (y : AdjoinRoot G) : y + y = 0 := by
  obtain ⟨p, rfl⟩ := AdjoinRoot.mk_surjective y
  rw [← map_add, ← two_mul, two_eq_zero_P, zero_mul, map_zero]

theorem bezout_gen (n : ℕ) (w : F128) (hw : w * (sqIter n (F128.monomial 1) + F128.monomial 1) = 1) :
    G ∣ F128.poly w * (X ^ (2 ^ n) - X) - 1 := by
  apply AdjoinRoot.mk_eq_mk.mp
  rw [map_mul, map_sub, mk_X_pow, AdjoinRoot.mk_X, map_one]
  have hneg : -(AdjoinRoot.root G) = AdjoinRoot.root G := neg_eq_of_add_eq_zero_left (adj_add_self _)
  rw [sub_eq_add_neg, hneg, ← toAdj_x, ← toAdj_sqIter, ← toAdj_add]
  show toAdj w * _ = 1
  rw [← toAdj_mul, hw, toAdj_one]

theorem G_bezout : G ∣ F128.poly bezoutW * (X ^ (2 ^ (2 ^ 6)) - X) - 1 := bezout_gen _ bezoutW bezout_exec

theorem G_natDegree : G.natDegree = 2 ^ (6 + 1) := by rw [natDegree_eq_of_degree_eq_some G_degree]; norm_num

/-- An irreducible factor of `X^(2^m) - X` over GF(2) has degree dividing `m`, and conversely. -/
theorem deg_dvd_of_dvd_frob (h : P) (hh : Irreducible h) (m : ℕ) (hm : h ∣ X ^ (2 ^ m) - X) : h.natDegree ∣ m :=
  hh.natDegree_dvd_of_dvd_X_pow_card_pow_sub_X (n := m) (by rw [Nat.card_zmod]; exact hm)

theorem dvd_frob_of_deg_dvd (h : P) (hh : Irreducible h) (m : ℕ) (hm : h.natDegree ∣ m) : h ∣ X ^ (2 ^ m) - X := by
  have := (hh.natDegree_dvd_iff_dvd_X_pow_card_pow_sub_X (n := m)).mp hm
  rwa [Nat.card_zmod] at this

/-- Rabin's test for a polynomial of degree `2^(k+1)`: it divides `X^(2^(2^(k+1))) - X`, and a Bezout witness makes
`X^(2^(2^k)) - X` invertible modulo it. -/
theorem irreducible_of_rabin (F w : P) (k : ℕ) (hF : F.natDegree = 2 ^ (k + 1)) (hF0 : F ≠ 0)
    (h1 : F ∣ X ^ (2 ^ (2 ^ (k + 1))) - X) (h2 : F ∣ w * (X ^ (2 ^ (2 ^ k)) - X) - 1) : Irreducible F := by
  have hFu : ¬ IsUnit F := fun hu => by
    have := natDegree_eq_zero_of_isUnit hu
    rw [hF] at this; exact absurd this (by positivity)
  obtain ⟨h, hh, hdvd⟩ := WfDvdMonoid.exists_irreducible_factor hFu hF0
  obtain ⟨i, hi, hdi⟩ := (Nat.dvd_prime_pow Nat.prime_two).mp (deg_dvd_of_dvd_frob h hh _ (dvd_trans hdvd h1))
  by_cases hk : i = k + 1
  · obtain ⟨c, hc⟩ := hdvd
    have hc0 : c ≠ 0 := fun h0 => hF0 (by rw [hc, h0, mul_zero])
    have hdeg : F.natDegree = h.natDegree + c.natDegree := by rw [hc]; exact natDegree_mul hh.ne_zero hc0
    have hc_deg : c.natDegree = 0 := by rw [hF, hdi, hk] at hdeg; omega
    have hcu : IsUnit c := by
      rw [eq_C_of_natDegree_eq_zero hc_deg]
      exact isUnit_C.mpr (isUnit_iff_ne_zero.mpr (fun h0 => hc0 (by
        rw [eq_C_of_natDegree_eq_zero hc_deg, h0, map_zero])))
    rw [hc]
    exact (associated_mul_unit_right h c hcu).irreducible hh
  · have hsub : h.natDegree ∣ 2 ^ k := by rw [hdi]; exact pow_dvd_pow 2 (by omega)
    have hX := dvd_frob_of_deg_dvd h hh _ hsub
    have hone : h ∣ 1 := by
      have ha : h ∣ w * (X ^ (2 ^ (2 ^ k)) - X) := dvd_mul_of_dvd_right hX _
      have hb : h ∣ w * (X ^ (2 ^ (2 ^ k)) - X) - 1 := dvd_trans hdvd h2
      simpa using dvd_sub ha hb
    exact absurd (isUnit_of_dvd_one hone) hh.not_isUnit

/-- **The GHASH polynomial is irreducible over GF(2).** -/
theorem G_irreducible : Irreducible G :=
  irreducible_of_rabin G (F128.poly bezoutW) 6 G_natDegree G_monic.ne_zero G_dvd_frob G_bezout

instance instFactIrreduciblePG : Fact (Irreducible G) := ⟨G_irreducible⟩

/-! ### Frobenius and the inverse -/

theorem sq_eq_expand (f : P) : f ^ 2 = expand (ZMod 2) 2 f := by
  rw [← map_frobenius_expand, ZMod.frobenius_zmod, map_id]

theorem pow_two_pow_eq_expand (f : P) (m : ℕ) : f ^ (2 ^ m) = expand (ZMod 2) (2 ^ m) f := by
  induction m with
  | zero => simp
  | succ m ih => rw [pow_succ, pow_mul, ih, sq_eq_expand, expand_expand, mul_comm]

/-- The Frobenius identity for every residue class, from the one for the root. -/
theorem frob_all (m : ℕ) (h : AdjoinRoot.root G ^ (2 ^ m) = AdjoinRoot.root G) (y : AdjoinRoot G) :
    y ^ (2 ^ m) = y := by
  obtain ⟨q, rfl⟩ := AdjoinRoot.mk_surjective y
  rw [← map_pow, pow_two_pow_eq_expand, ← AdjoinRoot.aeval_eq, ← AdjoinRoot.aeval_eq, expand_aeval, h]

theorem inv_of_frob (N : ℕ) (hN : 2 ≤ N) {F : Type*} [Field F] (y : F) (hf : y ^ N = y) (hy : y ≠ 0) :
    y * y ^ (N - 2) = 1 := by
  have h1 : y * y ^ (N - 1) = y * 1 := by
    rw [mul_one, ← pow_succ', show N - 1 + 1 = N by omega, hf]
  have h2 : y ^ (N - 1) = 1 := mul_left_cancel₀ hy h1
  rw [← pow_succ', show N - 2 + 1 = N - 1 by omega, h2]

theorem toAdj_invGo (a : F128) : ∀ (k : ℕ) (acc : F128),
    toAdj (F128.invGo a k acc) = toAdj acc ^ (2 ^ k) * toAdj a ^ (2 ^ k - 1) := by
  intro k
  induction k with
  | zero => intro acc; simp [F128.invGo]
  | succ k ih =>
    intro acc
    have h1 : 1 ≤ 2 ^ k := Nat.one_le_two_pow
    show toAdj (F128.invGo a k (F128.mul (F128.mul acc acc) a)) = _
    rw [ih]
    show toAdj ((acc * acc) * a) ^ 2 ^ k * toAdj a ^ (2 ^ k - 1) = _
    rw [toAdj_mul, toAdj_mul, mul_pow, mul_pow, ← sq, ← pow_mul, mul_assoc, ← pow_add, pow_succ, mul_comm (2 ^ k) 2]
    congr 2
    omega

theorem toAdj_inv (a : F128) : toAdj (F128.inv a) = toAdj a ^ (2 ^ (127 + 1) - 2) := by
  show toAdj (F128.mul (F128.invGo a 127 F128.one) (F128.invGo a 127 F128.one)) = _
  change toAdj (F128.invGo a 127 1 * F128.invGo a 127 1) = _
  rw [toAdj_mul, toAdj_invGo, toAdj_one, one_pow, one_mul, ← sq, ← pow_mul]
  congr 1

theorem sqIter_frob : sqIter (127 + 1) (F128.monomial 1) = F128.monomial 1 := by decide +kernel

theorem root_frob' : AdjoinRoot.root G ^ (2 ^ (127 + 1)) = AdjoinRoot.root G := by
  rw [← toAdj_x, ← toAdj_sqIter, sqIter_frob]

theorem toAdj_mul_inv (a : F128) (ha : a ≠ 0) : toAdj a * toAdj (F128.inv a) = 1 := by
  have hy : toAdj a ≠ 0 := fun h => ha (toAdj_injective (by rw [h, toAdj_zero]))
  rw [toAdj_inv]
  exact inv_of_frob _ (by norm_num) _ (frob_all _ root_frob' _) hy

theorem toAdj_inv_zero : toAdj (F128.inv 0) = 0 := by
  rw [toAdj_inv, toAdj_zero, zero_pow (by norm_num)]

/-- GF(2^128): the field GF(2)[X]/(G) on the executable's representation. -/
def GF128 := F128

noncomputable instance instFieldGF128 : Field GF128 := equivAdj.field

namespace GF128

def of (a : F128) : GF128 := a

theorem add_eq (a b : F128) : of a + of b = of (a + b) := by
  show equivAdj.symm (equivAdj a + equivAdj b) = a + b
  apply equivAdj.injective
  simp only [Equiv.apply_symm_apply, equivAdj_apply, toAdj_add]

theorem mul_eq (a b : F128) : of a * of b = of (a * b) := by
  show equivAdj.symm (equivAdj a * equivAdj b) = a * b
  apply equivAdj.injective
  simp only [Equiv.apply_symm_apply, equivAdj_apply, toAdj_mul]

theorem zero_eq : (0 : GF128) = of 0 := by
  show equivAdj.symm 0 = 0
  apply equivAdj.injective
  simp only [Equiv.apply_symm_apply, equivAdj_apply, toAdj_zero]

theorem one_eq : (1 : GF128) = of 1 := by
  show equivAdj.symm 1 = 1
  apply equivAdj.injective
  simp only [Equiv.apply_symm_apply, equivAdj_apply, toAdj_one]

/-- `x · x⁻¹ = 1`: the executable's `F256.xinv` is the inverse of the generator `x` (checked by the kernel). -/
theorem xinv_mul_x : of F256.xinv * of (F128.monomial 1) = 1 := by
  rw [mul_eq, one_eq]
  exact congrArg of (by decide)

theorem inv_eq (a : F128) : (of a)⁻¹ = of (F128.inv a) := by
  by_cases ha : a = 0
  · subst ha
    have : F128.inv 0 = 0 := toAdj_injective (by rw [toAdj_inv_zero, toAdj_zero])
    rw [this, ← zero_eq, inv_zero]
  · apply inv_eq_of_mul_eq_one_right
    rw [mul_eq, one_eq]
    exact congrArg of (toAdj_injective (by rw [toAdj_mul, toAdj_mul_inv a ha, toAdj_one]))

end GF128

end FlockLevel3

