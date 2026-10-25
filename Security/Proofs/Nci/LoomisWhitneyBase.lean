import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.Real.Sqrt

/-!
# The discrete Loomis–Whitney inequality, on Mathlib alone

A finite set of triples has at most `√(|π₁₂| |π₁₃| |π₂₃|)` elements, where the `π` are its three coordinate-pair images.
The proof slices the set by its first coordinate: a slice lies in the product of its two projections' slices and injects
into the third projection, so its size squared is at most the product of the three, and Cauchy–Schwarz sums the slices.

NCI's main theorem applies it to the products `W_ℓ[i,j] x_t[j]` an IU computes at its other positions, as index triples
`(i, j, t)`: their projections are the weight entries `(i, j)` they use, the input entries `(t, j)` they use, and the
output entries `(t, i)` they contribute to (`card_products_le_sqrt`), and summing over the multiplications `ℓ` with
Cauchy–Schwarz bounds them by `√(p Q Y)` (`sum_card_products_le_sqrt`). The cover theorem's partial-sum bounds apply it
to a block's products as cells of a matrix cube. Both import it from here, which needs nothing but Mathlib.
-/

namespace Nci.SecurityProofs

open Finset

section LoomisWhitney

variable {α β γ : Type*} [DecidableEq α] [DecidableEq β] [DecidableEq γ]

private theorem card_image_eq_sum_fibres {ω δ : Type*} [DecidableEq δ] (T : Finset (α × ω)) (f : α × ω → α × δ)
    (hf : ∀ x, (f x).1 = x.1) : #(T.image f) = ∑ a ∈ T.image Prod.fst, #{y ∈ T.image f | y.1 = a} :=
  card_eq_sum_card_fiberwise fun y hy => by
    obtain ⟨x, hx, rfl⟩ := mem_image.1 hy
    exact hf x ▸ mem_image_of_mem _ hx

/-- **The discrete Loomis–Whitney inequality** in three dimensions: a finite set of triples, squared, has at most the
product of the sizes of its three coordinate-pair projections. -/
theorem loomis_whitney (T : Finset (α × β × γ)) :
    #T ^ 2 ≤ #(T.image fun x => (x.1, x.2.1)) * #(T.image fun x => (x.1, x.2.2)) *
      #(T.image fun x => (x.2.1, x.2.2)) := by
  have slice : ∀ a, #{x ∈ T | x.1 = a} ^ 2 ≤ #{y ∈ T.image (fun x => (x.1, x.2.1)) | y.1 = a} *
      (#{y ∈ T.image (fun x => (x.1, x.2.2)) | y.1 = a} * #(T.image fun x => (x.2.1, x.2.2))) := by
    intro a
    have hAB : #{x ∈ T | x.1 = a} ≤ #{y ∈ T.image (fun x => (x.1, x.2.1)) | y.1 = a} *
        #{y ∈ T.image (fun x => (x.1, x.2.2)) | y.1 = a} := by
      rw [← card_product]
      refine card_le_card_of_injOn (fun x => ((x.1, x.2.1), (x.1, x.2.2))) (fun x hx => ?_) fun x _ y _ h => ?_
      · obtain ⟨hxT, rfl⟩ := mem_filter.1 hx
        exact mem_product.2 ⟨mem_filter.2 ⟨mem_image_of_mem _ hxT, rfl⟩, mem_filter.2 ⟨mem_image_of_mem _ hxT, rfl⟩⟩
      · simp only [Prod.mk.injEq] at h
        exact Prod.ext h.1.1 (Prod.ext h.1.2 h.2.2)
    have hC : #{x ∈ T | x.1 = a} ≤ #(T.image fun x => (x.2.1, x.2.2)) := by
      refine card_le_card_of_injOn (fun x => (x.2.1, x.2.2)) (fun x hx => mem_image_of_mem _ (mem_filter.1 hx).1)
        fun x hx y hy h => ?_
      rw [mem_coe, mem_filter] at hx hy
      simp only [Prod.mk.injEq] at h
      exact Prod.ext (hx.2.trans hy.2.symm) (Prod.ext h.1 h.2)
    rw [sq, ← mul_assoc]
    exact Nat.mul_le_mul hAB hC
  rw [card_eq_sum_card_image Prod.fst T, card_image_eq_sum_fibres T (fun x => (x.1, x.2.1)) fun _ => rfl,
    card_image_eq_sum_fibres T (fun x => (x.1, x.2.2)) fun _ => rfl, mul_assoc,
    sum_mul _ _ #(T.image fun x => (x.2.1, x.2.2))]
  exact sum_sq_le_sum_mul_sum_of_sq_le_mul _ (fun _ _ => Nat.zero_le _) (fun _ _ => Nat.zero_le _) fun a _ => slice a

end LoomisWhitney

section Products

variable {ι κ τ : Type*} [DecidableEq ι] [DecidableEq κ] [DecidableEq τ]

/-- Loomis–Whitney over the index triples `(i, j, t)` of products `W[i,j] x_t[j]`: their number squared is at most the
product of the weight entries `(i, j)` they use, the input entries `(t, j)` they use and the output entries `(t, i)` they
contribute to. -/
theorem card_products_sq_le (T : Finset (ι × κ × τ)) :
    #T ^ 2 ≤ #(T.image fun x => (x.1, x.2.1)) * #(T.image fun x => (x.2.2, x.2.1)) *
      #(T.image fun x => (x.2.2, x.1)) := by
  have inputs : #(T.image fun x => (x.2.2, x.2.1)) = #(T.image fun x => (x.2.1, x.2.2)) := by
    rw [← card_image_of_injective (T.image fun x => (x.2.1, x.2.2)) Prod.swap_injective, image_image]
    rfl
  have outputs : #(T.image fun x => (x.2.2, x.1)) = #(T.image fun x => (x.1, x.2.2)) := by
    rw [← card_image_of_injective (T.image fun x => (x.1, x.2.2)) Prod.swap_injective, image_image]
    rfl
  rw [inputs, outputs, mul_right_comm]
  exact loomis_whitney T

/-- The draft's form: an IU's products at one multiplication, as index triples `(i, j, t)`, number at most `√(p q y)` when
they use at most `p` weight entries `(i, j)` and `q` input entries `(t, j)`, and contribute to at most `y` output entries
`(t, i)`. -/
theorem card_products_le_sqrt (T : Finset (ι × κ × τ)) {p q y : ℕ}
    (hp : #(T.image fun x => (x.1, x.2.1)) ≤ p) (hq : #(T.image fun x => (x.2.2, x.2.1)) ≤ q)
    (hy : #(T.image fun x => (x.2.2, x.1)) ≤ y) : (#T : ℝ) ≤ √(p * q * y) :=
  Real.le_sqrt_of_sq_le <| by
    exact_mod_cast (card_products_sq_le T).trans (Nat.mul_le_mul (Nat.mul_le_mul hp hq) hy)

/-- Loomis–Whitney at each multiplication `ℓ`, summed with Cauchy–Schwarz: an IU whose products at multiplication `ℓ`
(at its other positions) use at most `p` weight entries, `q ℓ` input entries and contribute to `y ℓ` output entries
computes at most `√(p Q Y)` of them in all, with `Q = ∑ q ℓ` and `Y = ∑ y ℓ`. -/
theorem sum_card_products_le_sqrt {L : Type*} (s : Finset L) (T : L → Finset (ι × κ × τ)) {p : ℕ} {q y : L → ℕ}
    (hp : ∀ ℓ ∈ s, #((T ℓ).image fun x => (x.1, x.2.1)) ≤ p)
    (hq : ∀ ℓ ∈ s, #((T ℓ).image fun x => (x.2.2, x.2.1)) ≤ q ℓ)
    (hy : ∀ ℓ ∈ s, #((T ℓ).image fun x => (x.2.2, x.1)) ≤ y ℓ) :
    (∑ ℓ ∈ s, #(T ℓ) : ℝ) ≤ √(p * (∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, y ℓ) := by
  refine Real.le_sqrt_of_sq_le ?_
  have h : (∑ ℓ ∈ s, #(T ℓ)) ^ 2 ≤ (∑ ℓ ∈ s, q ℓ) * ∑ ℓ ∈ s, p * y ℓ :=
    sum_sq_le_sum_mul_sum_of_sq_le_mul s (fun _ _ => Nat.zero_le _) (fun _ _ => Nat.zero_le _) fun ℓ hℓ =>
      (card_products_sq_le (T ℓ)).trans <| by
        calc _ ≤ p * q ℓ * y ℓ := Nat.mul_le_mul (Nat.mul_le_mul (hp ℓ hℓ) (hq ℓ hℓ)) (hy ℓ hℓ)
          _ = q ℓ * (p * y ℓ) := by ring
  rw [← mul_sum] at h
  have h' : ((∑ ℓ ∈ s, #(T ℓ) : ℕ) : ℝ) ^ 2 ≤ (((∑ ℓ ∈ s, q ℓ) * (p * ∑ ℓ ∈ s, y ℓ) : ℕ) : ℝ) := by exact_mod_cast h
  push_cast at h'
  exact h'.trans_eq (by push_cast; ring)

end Products

end Nci.SecurityProofs
