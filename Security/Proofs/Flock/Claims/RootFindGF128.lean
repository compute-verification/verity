import Proofs.Flock.Level3.ArithFacts
import Proofs.Flock.Claims.Counted
import Proofs.Flock.Claims.RootFindCost

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

theorem trace_zero {F : Type} [CommRing F] (n : ℕ) : trace n (0 : F) = 0 := by
  simp [trace]

/-- Over `F` with `2^n` elements, `n > 0`, the trace is not identically zero: `Σ_{i<n} X^(2^i)` has degree
`2^(n−1)`, so fewer roots than `F` has elements. -/
theorem exists_trace_ne_zero {F : Type} [Field F] [Fintype F] [DecidableEq F] {n : ℕ} (hn : 0 < n)
    (hcard : Fintype.card F = 2 ^ n) : ∃ z : F, trace n z ≠ 0 := by
  by_contra h
  push Not at h
  set P : F[X] := ∑ i ∈ Finset.range n, X ^ 2 ^ i
  have hev : ∀ z, P.eval z = trace n z := fun z => by simp [P, trace, eval_finsetSum]
  have hc : P.coeff (2 ^ (n - 1)) = 1 := by
    simp only [P, finsetSum_coeff, coeff_X_pow]
    rw [Finset.sum_eq_single (n - 1)]
    · simp
    · intro i _ hne
      simp only [ite_eq_right_iff]
      intro h
      exact absurd (Nat.pow_right_injective le_rfl h).symm hne
    · intro h
      simp only [Finset.mem_range] at h
      omega
  have hP : P ≠ 0 := fun h0 => by rw [h0, coeff_zero] at hc; exact zero_ne_one hc
  have hdeg : P.natDegree ≤ 2 ^ (n - 1) := by
    refine natDegree_sum_le_of_forall_le _ _ fun i hi => ?_
    rw [natDegree_X_pow]
    exact Nat.pow_le_pow_right (by norm_num) (by simp only [Finset.mem_range] at hi; omega)
  have hroots : (Finset.univ : Finset F) ⊆ P.roots.toFinset := fun z _ => by
    rw [Multiset.mem_toFinset, mem_roots hP, IsRoot.def, hev, h z]
  have h1 := Finset.card_le_card hroots
  rw [Finset.card_univ, hcard] at h1
  have h2 := (Multiset.toFinset_card_le P.roots).trans (card_roots' P)
  have : 2 ^ (n - 1) < 2 ^ n := Nat.pow_lt_pow_right (by norm_num) (by omega)
  omega

theorem pack_add (a b : Fin 128 → ZMod 2) : pack (a + b) = pack a + pack b := by
  have h := unpack_add (pack a) (pack b)
  rw [unpack_pack, unpack_pack] at h
  rw [← h, pack_unpack]

theorem pack_zero : pack 0 = 0 := by
  have h := pack_add 0 0
  rw [add_zero] at h
  simpa using h

/-- The packing basis separates GF(2^128): if `Tr(x^v·c) = 0` for every `v`, then by additivity `Tr(y·c) = 0` for
every `y`, so the trace vanishes on all of GF(2^128) (`y = z·c⁻¹`), which it does not (`exists_trace_ne_zero`). -/
theorem separates_basis128 : Separates 128 basis128 := by
  intro c hc
  by_contra h
  push Not at h
  have hv : ∀ v : Fin 128, trace 128 (pack (Pi.single v 1) * c) = 0 := fun v =>
    h _ (List.mem_map.2 ⟨v, List.mem_finRange v, rfl⟩)
  have hy : ∀ y : GF128, trace 128 (y * c) = 0 := by
    intro y
    have e : y = ∑ v : Fin 128, pack (Pi.single v (unpack y v)) := by
      conv_lhs => rw [← pack_unpack y, ← Finset.univ_sum_single (unpack y)]
      exact map_sum (AddMonoidHom.mk' pack pack_add) _ _
    rw [e, show (∑ v : Fin 128, pack (Pi.single v (unpack y v))) * c =
        ∑ v : Fin 128, pack (Pi.single v (unpack y v)) * c from Finset.sum_mul _ _ _]
    rw [show trace 128 (∑ v : Fin 128, pack (Pi.single v (unpack y v)) * c) =
        ∑ v : Fin 128, trace 128 (pack (Pi.single v (unpack y v)) * c) from
      map_sum (AddMonoidHom.mk' (trace 128) (trace_add 128)) _ _]
    refine Finset.sum_eq_zero fun v _ => ?_
    rcases (by decide : ∀ z : ZMod 2, z = 0 ∨ z = 1) (unpack y v) with h0 | h1
    · rw [h0, Pi.single_zero, pack_zero, zero_mul, trace_zero]
    · rw [h1]; exact hv v
  obtain ⟨z, hz⟩ := exists_trace_ne_zero (F := GF128) (by norm_num) card_F
  exact hz (by simpa [mul_assoc, inv_mul_cancel₀ hc] using hy (z * c⁻¹))

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
