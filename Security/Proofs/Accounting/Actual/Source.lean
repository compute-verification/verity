import Definitions.Accounting.Actual.Source
import Proofs.Accounting.Actual.Split

/-!
# Ingress by source

Ingress from a source is part of ingress (`ingressFrom_le_ingress`) and grows with the source (`ingressFrom_mono`). It
adds over disjoint sources (`ingressFrom_union`) and is all of ingress from a source that holds the whole outside
(`ingressFrom_eq_ingress`), so ingress is the sum over a partition of the outside. A cover bounds it
(`ingress_le_ingressFrom_add`).
-/

namespace Accounting.SecurityProofs.Actual

open Verity Finset

variable {L : Type*} (A : Placed L)

theorem ingressFrom_le_ingress (X Y : Finset L) (I : Set ℕ) : A.ingressFrom X Y I ≤ A.ingress X I := by
  classical
  unfold Placed.ingressFrom Placed.ingress
  exact width_mono A fun a ha => by
    rw [mem_filter] at ha ⊢
    exact ⟨ha.1, ha.2.1⟩

theorem ingressFrom_mono (X : Finset L) {Y Y' : Finset L} (h : Y ⊆ Y') (I : Set ℕ) :
    A.ingressFrom X Y I ≤ A.ingressFrom X Y' I := by
  classical
  unfold Placed.ingressFrom
  exact width_mono A fun a ha => by
    rw [mem_filter] at ha ⊢
    exact ⟨ha.1, ha.2.1, h ha.2.2⟩

theorem ingressFrom_eq_ingress (X Y : Finset L) (I : Set ℕ) (h : ∀ l, l ∉ X → l ∈ Y) :
    A.ingressFrom X Y I = A.ingress X I := by
  classical
  unfold Placed.ingressFrom Placed.ingress
  congr 1
  ext a
  simp only [mem_filter]
  exact ⟨fun h' => ⟨h'.1, h'.2.1⟩, fun h' => ⟨h'.1, h'.2, h _ h'.2⟩⟩

theorem ingress_le_ingressFrom_add (X Y Z : Finset L) (I : Set ℕ) (h : ∀ l, l ∉ X → l ∈ Y ∨ l ∈ Z) :
    A.ingress X I ≤ A.ingressFrom X Y I + A.ingressFrom X Z I := by
  classical
  unfold Placed.ingressFrom Placed.ingress
  refine (width_mono A fun a ha => ?_).trans (width_union_le A _ _)
  rw [mem_filter] at ha
  rw [mem_union, mem_filter, mem_filter]
  rcases h _ ha.2 with hy | hz
  · exact Or.inl ⟨ha.1, ha.2, hy⟩
  · exact Or.inr ⟨ha.1, ha.2, hz⟩

open Classical in
theorem ingressFrom_union (X Y Z : Finset L) (I : Set ℕ) (hd : Disjoint Y Z) :
    A.ingressFrom X (Y ∪ Z) I = A.ingressFrom X Y I + A.ingressFrom X Z I := by
  classical
  unfold Placed.ingressFrom Circuit.width
  rw [← Finset.sum_union]
  · congr 1
    ext a
    simp only [mem_filter, mem_union]
    tauto
  · rw [Finset.disjoint_left]
    intro a ha hb
    rw [mem_filter] at ha hb
    exact Finset.disjoint_left.1 hd ha.2.2 hb.2.2

/-- Links by source bound ingress from a cover of the outside by two sources. -/
theorem ingress_le_of_linksFrom {𝒮 : Set (Finset L × Finset L)} {β : Finset L → Finset L → ℕ}
    (hL : A.LinksFrom 𝒮 β) (X Y Z : Finset L) (hY : (X, Y) ∈ 𝒮) (hZ : (X, Z) ∈ 𝒮)
    (h : ∀ l, l ∉ X → l ∈ Y ∨ l ∈ Z) (a b : ℕ) :
    A.ingress X (Set.Ico a b) ≤ (β X Y + β X Z) * (b - a) := by
  have h1 := hL _ hY a b
  have h2 := hL _ hZ a b
  have := ingress_le_ingressFrom_add A X Y Z (Set.Ico a b) h
  simp only at h1 h2
  rw [add_mul]
  omega

end Accounting.SecurityProofs.Actual
