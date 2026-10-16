import Proofs.Flock.Claims.RootFind
import Proofs.Flock.Level3.ArithFacts
import Definitions.Claims

/-!
# The root finder over GF(2^128)

Flock's decoder works over `GF128` (the executable's GF(2^128), `FlockLevel3.GF128`: `2^128` elements, characteristic
2). `roots128` is Berlekamp's trace root finder (`RootFind.roots`) there, at `n = 128` squarings and the packing basis
`x^v`, `v < 128` (`basis128`), which separates the field (`separates_basis128`: the trace form is nondegenerate and the
basis spans). It lists exactly the roots of a nonzero polynomial, each once (`roots128_spec`), and its counted version
meets `Claims.DecoderComplexity` at the input's length, with `roots128Cost = 64·129²·(d + 3)^4`
(`roots128_complexity`).
-/

namespace Claims.Flock

open RootFind FlockLevel3 Polynomial

/-- GF(2^128)'s packing basis: `x^v` for `v < 128`. -/
noncomputable def basis128 : List GF128 := (List.finRange 128).map fun v => pack (Pi.single v 1)

/-- The packing basis separates GF(2^128). -/
theorem separates_basis128 : Separates 128 basis128 := by
  sorry

/-- **The root finder over GF(2^128)**. -/
noncomputable def roots128 (f : List GF128) : List GF128 := roots 128 basis128 f

/-- `roots128` with its field-operation count, from the same computation. -/
noncomputable def roots128Counted (f : List GF128) : List GF128 × ℕ :=
  ((rootsC 128 basis128 f).val, (rootsC 128 basis128 f).ops)

/-- The operation bound, a polynomial in the input's length `d`: `64·129²·(d + 3)^4`. -/
noncomputable def roots128Cost : Polynomial ℕ := C (64 * 129 * 129) * (X + C 3) ^ 4

/-- **`roots128` lists exactly the roots of a nonzero polynomial, each once.** -/
theorem roots128_spec {f : List GF128} (hf : toPoly f ≠ 0) :
    (roots128 f).Nodup ∧ ∀ a, a ∈ roots128 f ↔ (toPoly f).IsRoot a :=
  roots_spec card_F separates_basis128 hf

/-- **`roots128`'s complexity**: its counted version's value is `roots128`'s, and its count is at most
`roots128Cost` at the input's length. -/
theorem roots128_complexity : DecoderComplexity roots128 roots128Counted List.length roots128Cost := by
  intro f
  refine ⟨rfl, ?_⟩
  have hb : basis128.length = 128 := by simp [basis128]
  have h := rootsC_ops_le 128 basis128 f
  rw [costBound, hb] at h
  simpa [roots128Counted, roots128Cost] using h

end Claims.Flock
