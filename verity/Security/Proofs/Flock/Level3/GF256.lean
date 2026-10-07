/-
Level 3, L3-F: the executable's GF(2^256) (`Flock.F256`, PROTOCOL.md §2.4) is the field GF128[Y]/(Y^2 + Y + x⁻¹): an element
`c0 + c1·u` is the residue class of `c0 + c1·Y`, and `F256.mul` (three products, `p0 + x⁻¹·p1`, `p2 + p0`) is the quotient
ring's product. `Y^2 + Y + x⁻¹` is irreducible because the trace of `x⁻¹` is 1 (Artin–Schreier: the trace vanishes on every
`y^2 + y`), computed by the kernel with the executable's multiplication. The field is transported onto the synonym `GF256`,
whose `+` and `*` are the executable's.
-/
import Proofs.Flock.Level3.GF128Field
import Mathlib.Algebra.Polynomial.SpecificDegree
import Mathlib.Algebra.CharP.Lemmas

open Polynomial Flock

namespace FlockLevel3

open GF128 in
theorem GF128.two_eq_zero : (2 : GF128) = 0 := by
  rw [show (2 : GF128) = 1 + 1 by norm_num, one_eq, add_eq, zero_eq]
  exact congrArg of (by decide)

abbrev K := GF128

/-- `x⁻¹` in GF128. -/
def xK : K := GF128.of F256.xinv

/-- The defining polynomial of the ladder: `Y^2 + Y + x⁻¹`. -/
noncomputable def Q : K[X] := X ^ 2 + (X + C xK)

theorem lin_degree : (X + C xK : K[X]).degree ≤ 1 := by
  rw [show (X + C xK : K[X]) = C 1 * X + C xK by simp]
  exact degree_linear_le

theorem one_lt_two_wb : (1 : WithBot ℕ) < 2 := by decide

theorem Q_monic : Q.Monic := by
  unfold Q; apply monic_X_pow_add
  exact lt_of_le_of_lt lin_degree (by exact_mod_cast one_lt_two_wb)

theorem Q_degree : Q.degree = 2 := by
  unfold Q
  rw [degree_add_eq_left_of_degree_lt, degree_X_pow]
  · rfl
  · rw [degree_X_pow]; exact lt_of_le_of_lt lin_degree (by exact_mod_cast one_lt_two_wb)

theorem two_eq_zero_KX : (2 : K[X]) = 0 := by
  rw [← map_ofNat C 2, GF128.two_eq_zero, map_zero]

theorem F256.mul_def (a b : F256) :
    a * b = ⟨a.c0 * b.c0 + F256.xinv * (a.c1 * b.c1), (a.c0 + a.c1) * (b.c0 + b.c1) + a.c0 * b.c0⟩ := rfl

theorem F256.add_def (a b : F256) : a + b = ⟨a.c0 + b.c0, a.c1 + b.c1⟩ := rfl

/-- The residue class of `c0 + c1·Y`. -/
noncomputable def phi (a : F256) : AdjoinRoot Q := AdjoinRoot.mk Q (C (GF128.of a.c0) + C (GF128.of a.c1) * X)

theorem phi_add (a b : F256) : phi (a + b) = phi a + phi b := by
  rw [F256.add_def, phi, phi, phi, ← map_add]
  refine congrArg (AdjoinRoot.mk Q) ?_
  simp only [← GF128.add_eq, map_add]
  ring

theorem phi_mul (a b : F256) : phi (a * b) = phi a * phi b := by
  rw [F256.mul_def, phi, phi, phi]
  simp only [← GF128.add_eq, ← GF128.mul_eq]
  rw [← map_mul, AdjoinRoot.mk_eq_mk]
  refine ⟨-C (GF128.of a.c1 * GF128.of b.c1), ?_⟩
  unfold Q xK
  simp only [map_add, map_mul]
  linear_combination (norm := skip)
  ring_nf
  rw [two_eq_zero_KX]
  ring

theorem phi_injective : Function.Injective phi := by
  intro a b h
  rw [phi, phi, AdjoinRoot.mk_eq_mk] at h
  have hdeg : ((C (GF128.of a.c0) + C (GF128.of a.c1) * X) - (C (GF128.of b.c0) + C (GF128.of b.c1) * X)).degree
      < Q.degree := by
    rw [Q_degree]
    have : (C (GF128.of a.c0) + C (GF128.of a.c1) * X) - (C (GF128.of b.c0) + C (GF128.of b.c1) * X) =
        C (GF128.of a.c1 - GF128.of b.c1) * X + C (GF128.of a.c0 - GF128.of b.c0) := by simp only [map_sub]; ring
    rw [this]
    exact lt_of_le_of_lt degree_linear_le (by exact_mod_cast one_lt_two_wb)
  have h0 := by_contra (fun hne => Q_monic.not_dvd_of_degree_lt hne hdeg h)
  have e0 := congrArg (fun p => p.coeff 0) (sub_eq_zero.mp h0)
  have e1 := congrArg (fun p => p.coeff 1) (sub_eq_zero.mp h0)
  simp at e0 e1
  cases a; cases b
  simp only [F256.mk.injEq]
  exact ⟨e0, e1⟩

theorem phi_surjective : Function.Surjective phi := by
  intro y
  obtain ⟨p, rfl⟩ := AdjoinRoot.mk_surjective y
  have hdeg : (p %ₘ Q).degree ≤ 1 := by
    have h2 := degree_modByMonic_lt p Q_monic
    rw [Q_degree] at h2
    rw [degree_le_iff_coeff_zero]
    intro m hm
    have hm2 : ((2 : ℕ) : WithBot ℕ) ≤ m := by
      have : (1 : ℕ) < m := by exact_mod_cast hm
      exact_mod_cast this
    exact coeff_eq_zero_of_degree_lt (lt_of_lt_of_le h2 (by exact_mod_cast hm2))
  refine ⟨⟨(p %ₘ Q).coeff 0, (p %ₘ Q).coeff 1⟩, ?_⟩
  rw [phi, AdjoinRoot.mk_eq_mk]
  have hr := eq_X_add_C_of_degree_le_one hdeg
  refine ⟨-(p /ₘ Q), ?_⟩
  have hq := modByMonic_eq_sub_mul_div p Q
  change C ((p %ₘ Q).coeff 0) + C ((p %ₘ Q).coeff 1) * X - p = Q * -(p /ₘ Q)
  rw [add_comm, ← hr, hq]
  ring

noncomputable def equivPhi : F256 ≃ AdjoinRoot Q := Equiv.ofBijective phi ⟨phi_injective, phi_surjective⟩

theorem equivPhi_apply (a : F256) : equivPhi a = phi a := rfl

/-! ### `Y^2 + Y + x⁻¹` is irreducible -/

instance instCharPGF128OfNatNat : CharP GF128 2 := (CharP.charP_iff_prime_eq_zero Nat.prime_two).mpr (by exact_mod_cast GF128.two_eq_zero)

theorem GF128.pow_eq (a : F128) (n : ℕ) : GF128.of a ^ n = equivAdj.symm (toAdj a ^ n) := rfl

theorem GF128.frob_of (a : F128) : GF128.of a ^ (2 ^ (127 + 1)) = GF128.of a := by
  rw [GF128.pow_eq, frob_all _ root_frob' (toAdj a)]
  exact equivAdj.symm_apply_apply a

theorem GF128.frob (z : GF128) : z ^ (2 ^ (127 + 1)) = z := GF128.frob_of z

/-- The trace `GF128 → GF(2)`: `T(z) = Σ_{i < 128} z^(2^i)`. -/
noncomputable def T (z : GF128) : GF128 := ∑ i ∈ Finset.range (127 + 1), z ^ (2 ^ i)

theorem T_add (a b : GF128) : T (a + b) = T a + T b := by
  simp only [T, add_pow_char_pow, Finset.sum_add_distrib]

theorem T_sq (z : GF128) : T (z ^ 2) = T z := by
  have h1 := Finset.sum_range_succ' (fun i => z ^ (2 ^ i)) (127 + 1)
  have h2 := Finset.sum_range_succ (fun i => z ^ (2 ^ i)) (127 + 1)
  rw [h2, GF128.frob, pow_zero, pow_one] at h1
  unfold T
  have : ∀ i, (z ^ 2) ^ (2 ^ i) = z ^ (2 ^ (i + 1)) := fun i => by rw [← pow_mul, pow_succ', mul_comm]
  simp only [this]
  exact (add_right_cancel h1).symm

theorem T_zero : T 0 = 0 := by
  simp [T]

/-- `x0 + x1 + … ` over `k` squarings: `acc + Σ_{i < k} z^(2^i)` with the executable's operations. -/
def traceGo : ℕ → F128 → F128 → F128
  | 0, _, acc => acc
  | k + 1, z, acc => traceGo k (z * z) (acc + z)

theorem traceGo_eq (k : ℕ) : ∀ (z acc : F128),
    GF128.of (traceGo k z acc) = GF128.of acc + ∑ i ∈ Finset.range k, GF128.of z ^ (2 ^ i) := by
  induction k with
  | zero => intro z acc; simp [traceGo]
  | succ k ih =>
    intro z acc
    rw [traceGo, ih, Finset.sum_range_succ', ← GF128.add_eq, ← GF128.mul_eq]
    have : ∀ i, (GF128.of z * GF128.of z) ^ (2 ^ i) = GF128.of z ^ (2 ^ (i + 1)) := fun i => by
      rw [← sq, ← pow_mul, pow_succ', mul_comm]
    simp only [this, pow_zero, pow_one]
    ring

theorem traceGo_xinv : traceGo (127 + 1) F256.xinv 0 = 1 := by decide +kernel

theorem T_xK : T xK = 1 := by
  have h := traceGo_eq (127 + 1) F256.xinv 0
  rw [traceGo_xinv, ← GF128.zero_eq, zero_add, ← GF128.one_eq] at h
  exact h.symm

theorem Q_no_root (y : K) : ¬ IsRoot Q y := by
  intro h
  have h0 : y ^ 2 + y + xK = 0 := by
    have := h.eq_zero
    simp only [Q, eval_add, eval_pow, eval_X, eval_C] at this
    rw [← add_assoc] at this; exact this
  have := congrArg T h0
  rw [T_add, T_add, T_sq, T_xK, T_zero] at this
  have h2 : T y + T y = 0 := by rw [← two_mul, GF128.two_eq_zero, zero_mul]
  rw [h2, zero_add] at this
  exact one_ne_zero this

theorem Q_irreducible : Irreducible Q := by
  apply irreducible_of_degree_le_three_of_not_isRoot _ Q_no_root
  rw [natDegree_eq_of_degree_eq_some Q_degree]
  decide

instance instFactIrreduciblePolynomialKQ : Fact (Irreducible Q) := ⟨Q_irreducible⟩

/-- GF(2^256): the field GF128[Y]/(Y^2 + Y + x⁻¹) on the executable's representation. -/
def GF256 := F256

noncomputable instance instFieldGF256 : Field GF256 := equivPhi.field

namespace GF256

def of (a : F256) : GF256 := a

theorem add_eq (a b : F256) : of a + of b = of (a + b) := by
  show equivPhi.symm (equivPhi a + equivPhi b) = a + b
  apply equivPhi.injective
  simp only [Equiv.apply_symm_apply, equivPhi_apply, phi_add]

theorem mul_eq (a b : F256) : of a * of b = of (a * b) := by
  show equivPhi.symm (equivPhi a * equivPhi b) = a * b
  apply equivPhi.injective
  simp only [Equiv.apply_symm_apply, equivPhi_apply, phi_mul]

end GF256

end FlockLevel3
