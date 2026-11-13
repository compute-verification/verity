module

public import Mathlib.Analysis.Real.Sqrt

/-!
# The discrete Loomis–Whitney inequality, on Mathlib alone

A finite set of triples has at most `√(|π₁₂| |π₁₃| |π₂₃|)` elements, where the `π` are its three coordinate-pair images.
The proof slices the set by its first coordinate: a slice lies in the product of its two projections' slices and injects
into the third projection, so its size squared is at most the product of the three, and Cauchy–Schwarz sums the slices.

The min-cut bound applies it (`loomis_whitney`) to a block's MACs of one matmul (`card_inBlock_sq_le`), and imports it
from here, which needs nothing but Mathlib.
-/

@[expose] public section

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

end Nci.SecurityProofs
