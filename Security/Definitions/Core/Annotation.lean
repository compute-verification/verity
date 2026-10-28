import Definitions.Core.Partition

/-!
# Annotations: the gates each label holds

An **annotation** of a circuit `C` with label set `L` maps each label to the set of gates it holds (`holds`). Proof
blocks, inference blocks, a placement, a cover and a selection are all annotations; each kind is a predicate on one,
not a type of its own:

* a **labelling** of a set `S` of gates (`Labels`): every gate of `S` is held by exactly one label, so the annotation
  reads as a function from `S` to `L`. A placement labels computed gates by devices;
* a **partition** (`IsPartition`): a labelling of the computed gates that holds no input gate. Proof blocks and
  inference blocks are partitions, and so is a `Verity.Partition` (`Partition.toAnnotation`);
* a **cover** of `S` (`Covers`): every gate of `S` is held by some label, and labels may overlap, as the blocks of an
  (s, k)-cover recompute;
* a **selection**: an annotation with one label, `Annotation C Unit`.

Annotations compose by maps of labels. `map f α` holds at `ℓ'` the gates of every label `f` sends to `ℓ'`, as a
placement on devices maps to one on hosts. `α` refines `β` (`Refines`) when each label of `α` holds a subset of some
label of `β`: "each inference block on one server" says the inference blocks refine the host placement. An annotation
of a circuit `D` that contains `C` pulls back to one of `C` along the containment (`pullback`).

An annotation is given by a public query over the circuit's description, the registered site and a private table; this
module states only what it denotes. Its **meaning** Φ is a `Prop` on an annotation, stated in its owner's area, and Φ is
what a check establishes: a check reads the annotation's committed table, never the algorithm that chose it. The query
reads no trace and no coins: the sample is a set of labels, not an annotation.
-/

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

/-- `α` is a partition: it labels the computed gates, and no label holds an input gate. -/
def IsPartition (α : Annotation C L) : Prop :=
  α.Labels C.computed ∧ ∀ ℓ, ∀ g ∈ α.holds ℓ, C.isInput g = false

/-- `α` covers `S`: every gate of `S` is held by some label; labels may overlap. -/
def Covers (α : Annotation C L) (S : Finset (Fin C.N)) : Prop :=
  ∀ g ∈ S, ∃ ℓ, g ∈ α.holds ℓ

/-- `α` refines `β`: each label of `α` holds a subset of what some label of `β` holds. -/
def Refines (α : Annotation C L) (β : Annotation C L') : Prop :=
  ∀ ℓ, ∃ ℓ', α.holds ℓ ⊆ β.holds ℓ'

open Classical in
/-- `α` relabelled along `f`: label `ℓ'` holds the gates of every label `f` sends to `ℓ'`. -/
noncomputable def map (f : L → L') (α : Annotation C L) : Annotation C L' :=
  ⟨fun ℓ' => univ.filter fun g => ∃ ℓ, f ℓ = ℓ' ∧ g ∈ α.holds ℓ⟩

/-- `β`, an annotation of `D`, pulled back to `C` along a containment of `C` in `D`: label `ℓ` holds the gates of `C`
whose images `β` puts in `ℓ`. -/
def pullback (ι : Containment C D) (β : Annotation D L) : Annotation C L :=
  ⟨fun ℓ => univ.filter fun g => ι.emb g ∈ β.holds ℓ⟩

end Annotation

/-- A partition as an annotation: unit `u` holds its gates. -/
def Partition.toAnnotation {C : Circuit} {n : ℕ} (P : Partition C n) : Annotation C (Fin n) :=
  ⟨P.gates⟩

end Verity
