module

public import Mathlib.Data.Set.Card
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-!
# Scratch: the counting step under "the leak is at most the sum of the certifiers' `K`"

A scratch module, never landing. A value read back from its parts' views takes at most the product of the views'
counts: with each view's count at most `2 ^ K j`, at most `2 ^ ∑ K` values.
-/

@[expose] public section

namespace PartsCountScratch

/-- **A value read back from its parts takes at most the product of the parts' counts**: where every `z` of `Zs` is
`dec` of its views and view `j` takes at most `2 ^ K j` values over `Zs`, `Zs` holds at most `2 ^ ∑ K` values. -/
theorem encard_le_two_pow_sum {Z ι : Type*} [Fintype ι] {V : ι → Type*} (Zs : Set Z) (view : ∀ j, Z → V j)
    (dec : (∀ j, V j) → Z) (hdec : ∀ z ∈ Zs, dec (fun j => view j z) = z) (K : ι → ℕ)
    (hK : ∀ j, (view j '' Zs).encard ≤ ((2 ^ K j : ℕ) : ℕ∞)) :
    Zs.encard ≤ ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by
  have hsub : Zs ⊆ dec '' (Set.pi Set.univ fun j => view j '' Zs) := by
    intro z hz
    exact ⟨fun j => view j z, fun j _ => ⟨z, hz, rfl⟩, hdec z hz⟩
  calc Zs.encard ≤ (dec '' (Set.pi Set.univ fun j => view j '' Zs)).encard := Set.encard_le_encard hsub
    _ ≤ (Set.pi Set.univ fun j => view j '' Zs).encard := Set.encard_image_le _ _
    _ = ∏ j, (view j '' Zs).encard := Set.encard_pi_eq_prod_encard
    _ ≤ ∏ j, ((2 ^ K j : ℕ) : ℕ∞) := Finset.prod_le_prod fun j _ => hK j
    _ = ((∏ j, 2 ^ K j : ℕ) : ℕ∞) := (Nat.cast_prod _ _).symm
    _ = ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by rw [Finset.prod_pow_eq_pow_sum]

end PartsCountScratch
