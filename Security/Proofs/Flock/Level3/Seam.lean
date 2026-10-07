/-
The refinement seam with the soundness lane's `Arith F K` (PROTOCOL.md §17.4): the embedding `F ⊂ K` and `u` on the
executable's representation. An executable `F256` value `⟨c0, c1⟩` is `embed c0 + embed c1 · u` (`Arith.ofLimbs`), and `u`
is a root of `Y^2 + Y + x⁻¹`.
-/
import Proofs.Flock.Level3.GF256
import Mathlib.Tactic.LinearCombination

open Polynomial Flock

namespace FlockLevel3

/-- A GF128 element as the executable's value. -/
def GF128.val (a : GF128) : F128 := a

/-- A GF256 element as its residue class. -/
noncomputable def toQ (x : GF256) : AdjoinRoot Q := equivPhi x

theorem toQ_injective : Function.Injective toQ := equivPhi.injective
theorem toQ_of (a : F256) : toQ (GF256.of a) = phi a := rfl
theorem toQ_mul (x y : GF256) : toQ (x * y) = toQ x * toQ y := equivPhi.apply_symm_apply _
theorem toQ_add (x y : GF256) : toQ (x + y) = toQ x + toQ y := equivPhi.apply_symm_apply _
theorem toQ_zero : toQ 0 = 0 := equivPhi.apply_symm_apply _
theorem toQ_one : toQ 1 = 1 := equivPhi.apply_symm_apply _

theorem phi_limbs (c0 c1 : F128) :
    phi ⟨c0, c1⟩ = AdjoinRoot.of Q (GF128.of c0) + AdjoinRoot.of Q (GF128.of c1) * AdjoinRoot.root Q := by
  rw [phi, map_add, map_mul, AdjoinRoot.mk_X]; rfl

theorem phi_ofF (a : F128) : phi ⟨a, 0⟩ = AdjoinRoot.of Q (GF128.of a) := by
  rw [phi_limbs, show GF128.of (0 : F128) = 0 from GF128.zero_eq.symm, map_zero, zero_mul, add_zero]

/-- The embedding `F ⊂ K`: `c ↦ ⟨c, 0⟩`. -/
noncomputable def embed : GF128 →+* GF256 where
  toFun a := GF256.of ⟨a.val, 0⟩
  map_one' := toQ_injective (by rw [toQ_of, phi_ofF, toQ_one]; exact map_one _)
  map_zero' := toQ_injective (by rw [toQ_of, phi_ofF, toQ_zero]; exact map_zero _)
  map_mul' a b := toQ_injective (by rw [toQ_mul, toQ_of, toQ_of, toQ_of, phi_ofF, phi_ofF, phi_ofF]; exact map_mul _ _ _)
  map_add' a b := toQ_injective (by rw [toQ_add, toQ_of, toQ_of, toQ_of, phi_ofF, phi_ofF, phi_ofF]; exact map_add _ _ _)

theorem toQ_embed (a : GF128) : toQ (embed a) = AdjoinRoot.of Q a := phi_ofF a.val

/-- `u = ⟨0, 1⟩` (`F256.u`). -/
def u : GF256 := GF256.of F256.u

theorem toQ_u : toQ u = AdjoinRoot.root Q := by
  show phi ⟨0, 1⟩ = _
  rw [phi_limbs, show GF128.of (0 : F128) = 0 from GF128.zero_eq.symm, show GF128.of (1 : F128) = 1 from GF128.one_eq.symm,
    map_zero, map_one, zero_add, one_mul]

/-- **Limbs.** The executable's `⟨c0, c1⟩` is `embed c0 + embed c1 · u` (the soundness model's `Arith.ofLimbs`). -/
theorem ofLimbs (c0 c1 : F128) : GF256.of ⟨c0, c1⟩ = embed (GF128.of c0) + embed (GF128.of c1) * u :=
  toQ_injective (by rw [toQ_add, toQ_mul, toQ_embed, toQ_embed, toQ_u, toQ_of, phi_limbs])

/-- `u^2 + u + x⁻¹ = 0` in K (§2.4). -/
theorem u_root : u * u + u + embed xK = 0 := by
  apply toQ_injective
  rw [toQ_add, toQ_add, toQ_mul, toQ_u, toQ_embed, toQ_zero]
  have h : AdjoinRoot.mk Q (X ^ 2 + (X + C xK)) = 0 := AdjoinRoot.mk_self
  rw [map_add, map_add, map_pow, AdjoinRoot.mk_X, AdjoinRoot.mk_C] at h
  linear_combination h

end FlockLevel3
