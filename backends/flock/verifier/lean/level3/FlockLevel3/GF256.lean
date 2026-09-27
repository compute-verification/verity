/-
Level 3, L3-F: the executable's GF(2^256) (`Flock.F256`, PROTOCOL.md §2.4) is the ring GF128[Y]/(Y^2 + Y + x⁻¹): an element
`c0 + c1·u` is the residue class of `c0 + c1·Y`, and `F256.mul` (three products, `p0 + x⁻¹·p1`, `p2 + p0`) is the quotient
ring's product. The structure is transported onto the synonym `GF256`, whose `+` and `*` are the executable's.
-/
import FlockLevel3.GF128Field

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

/-- GF(2^256) with the ring structure of GF128[Y]/(Y^2 + Y + x⁻¹), on the executable's representation. -/
def GF256 := F256

noncomputable instance : CommRing GF256 := equivPhi.commRing

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
