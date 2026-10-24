import Definitions.Core.Partition
import Definitions.Population.Dataflow

/-!
# A partition as a population

A partition of a circuit (`Verity.Partition`, sampled proofs' units) is a population (`toDataflow`) over its committed
wires: the slots are the committed gates, the inputs are the input gates, and unit `u` writes its committed gates,
out(R) for its gates `R`, and reads the gates outside it that its gates read, in(R). Both sets come from the circuit's
wiring, so they are public and fixed before any value exists.

The partition's correctness of a unit becomes the population's through `fns`: unit `u`'s function recomputes its gates
from the values it reads (`Partition.localEval`), every value carried in `W` bits (`encode`), `W` at least every
gate's width.

`Ordered`: no wire runs from a gate of a later unit into a gate of an earlier one. `InLayoutOrder`: a later gate lies
in the same unit or a later one, so every unit is an interval of gates and the units are numbered in layout order, as
`Q_call` v1 cuts them.
-/

namespace Verity.Partition

open Finset

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- The committed wires, as slots. -/
abbrev Slot : Type := {g : Fin C.N // g ∈ P.committed}

/-- **The partition's population**: the committed wires as slots and the input gates as inputs; unit `u` writes its
committed gates and reads in(R) for its gates `R`. -/
def toDataflow : Population.Dataflow (Fin n) P.Slot where
  input := univ.filter fun s => C.isInput s.1 = true
  reads u := univ.filter fun s => s.1 ∈ C.into (P.gates u)
  writes u := univ.filter fun s => P.unit s.1 = some u

/-- A transcript's committed values, each in `W` bits. -/
def encode (W : ℕ) (X : C.Transcript) : P.Slot → BitVec W := fun s => (X s.1).setWidth W

open Classical in
/-- **The partition's unit functions**: unit `u` recomputes its gates from the values it reads, every other gate
taken as zero, and writes its committed gates' values in `W` bits. -/
noncomputable def fns (W : ℕ) : P.toDataflow.Fns (BitVec W) := fun u r s =>
  (P.localEval (fun g => if h : ∃ hg : g ∈ P.committed, (⟨g, hg⟩ : P.Slot) ∈ P.toDataflow.reads u
      then (r ⟨⟨g, h.1⟩, h.2⟩).setWidth (C.w g) else 0) u s.1.1).setWidth W

/-- **Ordered**: a gate that a gate of unit `v` reads lies in no later unit than `v`. -/
def Ordered : Prop :=
  ∀ g h : Fin C.N, ∀ u v : Fin n, g ∈ C.reads h → P.unit g = some u → P.unit h = some v → u ≤ v

/-- **In layout order**: a later gate lies in the same unit or a later one. -/
def InLayoutOrder : Prop :=
  ∀ g h : Fin C.N, ∀ u v : Fin n, g ≤ h → P.unit g = some u → P.unit h = some v → u ≤ v

end Verity.Partition
