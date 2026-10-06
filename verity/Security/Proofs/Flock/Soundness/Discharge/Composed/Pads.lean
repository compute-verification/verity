import Mathlib.LinearAlgebra.Vandermonde

/-!
# The pads code reaches every value at its opened positions

The inner code (`Zk.codeAt`) is `Σ yᵢ Xⁱ` at `X = j + 1`, its message the pads `h` then `K_PAD` padding symbols, so
padding symbol `j` contributes `x^(len + j)` at an opened point `x`. `padsOnto_pow`: at at most `nPad` distinct nonzero
opened points, the padding reaches every vector of values there (a scaled Vandermonde matrix), as `Table.PadsOnto`
states for each rep.
-/

namespace FlockSoundness.Discharge.Composed

open Matrix

theorem padsOnto_pow {F : Type} [Field F] {nQ nPad : ℕ} (len : ℕ) (x : Fin nQ → F) (hx : Function.Injective x)
    (h0 : ∀ k, x k ≠ 0) (hQ : nQ ≤ nPad) :
    Function.Surjective fun p : Fin nPad → F => fun k => ∑ j, x k ^ (len + j.val) * p j := by
  intro v
  let M : Matrix (Fin nQ) (Fin nQ) F := fun k j => x k ^ len * x k ^ j.val
  have hM : M = diagonal (fun k => x k ^ len) * vandermonde x := by
    ext k j
    simp [M, diagonal_mul, vandermonde_apply]
  have hdet : IsUnit M.det := by
    rw [hM, det_mul, det_diagonal]
    refine isUnit_iff_ne_zero.mpr (mul_ne_zero (Finset.prod_ne_zero_iff.mpr fun k _ => pow_ne_zero _ (h0 k)) ?_)
    exact det_vandermonde_ne_zero_iff.mpr hx
  let a := M⁻¹ *ᵥ v
  have ha : M *ᵥ a = v := by
    simp only [a, mulVec_mulVec, mul_nonsing_inv _ hdet, one_mulVec]
  refine ⟨fun j => if h : j.val < nQ then a ⟨j.val, h⟩ else 0, ?_⟩
  funext k
  rw [← ha]
  simp only [mulVec, dotProduct, M]
  symm
  apply Fintype.sum_of_injective (Fin.castLE hQ) (Fin.castLE_injective hQ)
  · intro j hj
    have : ¬ j.val < nQ := fun h => hj ⟨⟨j.val, h⟩, Fin.ext rfl⟩
    simp [this]
  · intro j
    simp [pow_add]

end FlockSoundness.Discharge.Composed
