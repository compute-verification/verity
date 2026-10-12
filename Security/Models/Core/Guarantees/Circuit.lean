import Definitions.Core.Partition
import Definitions.Core.Declaration

/-! # What correct units establish about a circuit, for gates of any width -/

namespace Verity.Guarantees

/-- **Composition**: when every unit is correct and the committed inputs are the anchors' values, every committed wire
carries its value in the circuit's evaluation. -/
def Compose : Prop :=
  ∀ (C : Circuit) (n : ℕ) (P : Partition C n) (X a : C.Transcript),
    C.InputsAgree X a → P.wrong X = ∅ → ∀ g ∈ P.committed, X g = C.eval a g

/-- **Composition through a cone**: the same for the committed wires of `T`, given only that no unit of `T`'s cone is
wrong. -/
def ComposeCone : Prop :=
  ∀ (C : Circuit) (n : ℕ) (P : Partition C n) (X a : C.Transcript) (T : Finset (Fin C.N)),
    C.InputsAgree X a → Disjoint (P.wrong X) (P.cone T) → ∀ t ∈ T, t ∈ P.committed → X t = C.eval a t

/-- A unit's correctness is decided by its committed inputs and outputs, what an opening shows. -/
def CorrectOfOpening : Prop :=
  ∀ (C : Circuit) (n : ℕ) (P : Partition C n) (X Y : C.Transcript) (u : Fin n),
    (∀ g ∈ P.io u, X g = Y g) → (P.Correct X u ↔ P.Correct Y u)

/-- Under a refinement, a transcript that agrees with another on the coarse committed wires has at least as many wrong
fine units as the other has wrong coarse units. -/
def RefineWrong : Prop :=
  ∀ (C : Circuit) (nf nc : ℕ) (Pf : Partition C nf) (Pc : Partition C nc), Pf.Refines Pc →
    ∀ X Y : C.Transcript, (∀ g ∈ Pc.committed, Y g = X g) → (Pc.wrong X).card ≤ (Pf.wrong Y).card

/-- A declaration that contains another implements it. -/
def ContainsImplements : Prop :=
  ∀ t d : Declaration, t.Contains d → t.Implements d

end Verity.Guarantees
