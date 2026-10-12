import Definitions.Accounting.Actual.Source
import Proofs.Accounting.Actual.Physical

/-!
# Ingress by source

Ingress from a source is part of ingress (`ingressFrom_le_ingress`) and grows with the source (`ingressFrom_mono`). It
adds over disjoint sources (`ingressFrom_union`) and is all of ingress from a source that holds the whole outside
(`ingressFrom_eq_ingress`), so ingress is the sum over a partition of the outside. A cover bounds it
(`ingress_le_ingressFrom_add`). Under `Links`, ingress from `Y` is at most the capacity of the links from `Y` a tick
(`ingressFrom_le`), as ingress is at most `β_X` (`ingress_le`).
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

/-- **Ingress from `Y` is at most the capacity of the links from `Y` a tick**:
`in_X([a, b); Y) ≤ linkCapacityFrom ends β X Y · (b − a)`. A gate placed in `Y` that `A[X, I]` reads from outside `X`
reaches it on a link across `X`'s boundary whose outer end is the gate's leaf, in `Y`. -/
theorem ingressFrom_le {E : Type*} [Fintype E] {ends : E → L × L} {β : E → ℕ} {link : Fin A.N → Fin A.N → E}
    (h : A.Links ends β link) (X Y : Finset L) (a b : ℕ) :
    A.ingressFrom X Y (Set.Ico a b) ≤ linkCapacityFrom ends β X Y * (b - a) := by
  classical
  unfold Placed.ingressFrom linkCapacityFrom
  rw [Finset.sum_mul]
  refine (width_mono A ?_).trans ((width_biUnion_le A _ _).trans
    (Finset.sum_le_sum fun e _ => (h.2 e (inner ends X e) a b).1))
  intro c hc
  obtain ⟨hc, hcX, hcY⟩ := mem_filter.1 hc
  obtain ⟨-, g, hg, hcg⟩ := (mem_into A).1 hc
  obtain ⟨-, hgX, hgt⟩ := (mem_region A).1 hg
  have hne : A.x c ≠ A.x g := fun h' => hcX (h' ▸ hgX)
  have he := (h.1 g c hcg hne).2
  refine mem_biUnion.2 ⟨link g c, mem_filter.2 ⟨mem_boundary hcX hgX he, ?_⟩, ?_⟩
  · rcases he with he | he <;> rw [he] <;> simp [hcX, hcY]
  · simp only [Placed.readOn, mem_filter, mem_univ, true_and]
    exact ⟨g, hcg, hne, rfl, (inner_outer hcX hgX he).1.symm, hgt⟩

end Accounting.SecurityProofs.Actual
