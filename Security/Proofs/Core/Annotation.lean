import Definitions.Core.Annotation

/-!
# Proofs about annotations

A `Verity.Partition` is a partition-shaped annotation (`Partition.toAnnotation_isPartition`), and its refinement is the
annotations' (`Partition.toAnnotation_refines`). Relabelling composes (`Annotation.map_id`, `Annotation.map_map`), and
a partition pulls back to a partition along a containment (`Annotation.IsPartition.pullback`).
-/

namespace Verity

open Finset

namespace Annotation

variable {C D : Circuit} {L L' L'' : Type}

theorem ext {α β : Annotation C L} (h : ∀ ℓ, α.holds ℓ = β.holds ℓ) : α = β := by
  cases α; cases β; simp only [mk.injEq]; exact funext h

@[simp] theorem mem_map {f : L → L'} {α : Annotation C L} {ℓ' : L'} {g : Fin C.N} :
    g ∈ (α.map f).holds ℓ' ↔ ∃ ℓ, f ℓ = ℓ' ∧ g ∈ α.holds ℓ := by
  simp [map]

@[simp] theorem mem_pullback {ι : Containment C D} {β : Annotation D L} {ℓ : L} {g : Fin C.N} :
    g ∈ (β.pullback ι).holds ℓ ↔ ι.emb g ∈ β.holds ℓ := by
  simp [pullback]

theorem map_id (α : Annotation C L) : α.map id = α :=
  ext fun _ => Finset.ext fun _ => by simp

theorem map_map (f : L → L') (f' : L' → L'') (α : Annotation C L) : (α.map f).map f' = α.map (f' ∘ f) :=
  ext fun _ => Finset.ext fun _ => by simp

theorem refines_map (f : L → L') (α : Annotation C L) : α.Refines (α.map f) :=
  fun ℓ => ⟨f ℓ, fun _ hg => mem_map.2 ⟨ℓ, rfl, hg⟩⟩

theorem IsPartition.pullback {α : Annotation D L} (h : α.IsPartition) (ι : Containment C D) :
    (α.pullback ι).IsPartition := by
  refine ⟨fun g hg => ?_, fun ℓ g hg => ?_⟩
  · have hD : ι.emb g ∈ D.computed := by
      simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hg ⊢
      rw [ι.isInput_emb, hg]
    simpa using h.1 _ hD
  · rw [← ι.isInput_emb]; exact h.2 ℓ _ (mem_pullback.1 hg)

end Annotation

namespace Partition

variable {C : Circuit} {n : ℕ}

@[simp] theorem mem_toAnnotation {P : Partition C n} {u : Fin n} {g : Fin C.N} :
    g ∈ P.toAnnotation.holds u ↔ P.unit g = some u := by
  simp [toAnnotation, gates]

theorem toAnnotation_isPartition (P : Partition C n) : P.toAnnotation.IsPartition := by
  refine ⟨fun g hg => ?_, fun u g hg => ?_⟩
  · have hin : C.isInput g = false := by simpa [Circuit.computed] using hg
    obtain ⟨u, hu⟩ : ∃ u, P.unit g = some u :=
      Option.ne_none_iff_exists'.1 fun h => by simp [(P.input_iff g).1 h] at hin
    exact ⟨u, mem_toAnnotation.2 hu, fun v hv => Option.some_injective _ ((mem_toAnnotation.1 hv).symm.trans hu)⟩
  · cases h : C.isInput g
    · rfl
    · simp [(P.input_iff g).2 h] at hg

theorem toAnnotation_refines {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (h : Pf.Refines Pc) :
    Pf.toAnnotation.Refines Pc.toAnnotation :=
  fun u => ⟨h.parent u, fun g hg => mem_toAnnotation.2 (by rw [h.sub g, mem_toAnnotation.1 hg]; rfl)⟩

end Partition

end Verity
