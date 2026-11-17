module

public import Definitions.Core.Partition

/-! Annotations, each label of a circuit mapped to the gates it holds, and the predicates on them (labelling,
partition, cover, refinement). -/

@[expose] public section

namespace Verity

open Finset

/-- An annotation of `C` by labels `L`: the gates each label holds. -/
structure Annotation (C : Circuit) (L : Type) where
  holds : L → Finset (Fin C.N)

namespace Annotation

variable {C D : Circuit} {L L' : Type}

/-- `α` labels `S`: every gate of `S` is held by exactly one label. Gates outside `S` are unconstrained. -/
def Labels (α : Annotation C L) (S : Finset (Fin C.N)) : Prop :=
  ∀ g ∈ S, ∃! ℓ, g ∈ α.holds ℓ

def IsPartition (α : Annotation C L) : Prop :=
  α.Labels C.computed ∧ ∀ ℓ, ∀ g ∈ α.holds ℓ, C.isInput g = false

def Covers (α : Annotation C L) (S : Finset (Fin C.N)) : Prop :=
  ∀ g ∈ S, ∃ ℓ, g ∈ α.holds ℓ

def Refines (α : Annotation C L) (β : Annotation C L') : Prop :=
  ∀ ℓ, ∃ ℓ', α.holds ℓ ⊆ β.holds ℓ'

open Classical in
noncomputable def map (f : L → L') (α : Annotation C L) : Annotation C L' :=
  ⟨fun ℓ' => univ.filter fun g => ∃ ℓ, f ℓ = ℓ' ∧ g ∈ α.holds ℓ⟩

def pullback (ι : Containment C D) (β : Annotation D L) : Annotation C L :=
  ⟨fun ℓ => univ.filter fun g => ι.emb g ∈ β.holds ℓ⟩

end Annotation

def Partition.toAnnotation {C : Circuit} {n : ℕ} (P : Partition C n) : Annotation C (Fin n) :=
  ⟨P.gates⟩

end Verity
