module

public import Definitions.Core.Partition
public import Definitions.Spot.Dataflow

/-! A partition of a circuit as a population over its committed wires, its read and write sets public because the wiring gives them. -/

@[expose] public section

namespace Verity.Partition

open Finset

variable {C : Circuit} {n : ℕ} (P : Partition C n)

abbrev Slot : Type := {g : Fin C.N // g ∈ P.committed}

def toDataflow : Population.Dataflow (Fin n) P.Slot where
  input := univ.filter fun s => C.isInput s.1 = true
  reads u := univ.filter fun s => s.1 ∈ C.into (P.gates u)
  writes u := univ.filter fun s => P.unit s.1 = some u

def encode (W : ℕ) (X : C.Trace) : P.Slot → BitVec W := fun s => (X s.1).setWidth W

open Classical in
/-- **The partition's unit functions**: unit `u` recomputes its gates from the values it reads, every other gate
taken as zero, and writes its committed gates' values in `W` bits. -/
noncomputable def fns (W : ℕ) : P.toDataflow.Fns (BitVec W) := fun u r s =>
  (P.localEval (fun g => if h : ∃ hg : g ∈ P.committed, (⟨g, hg⟩ : P.Slot) ∈ P.toDataflow.reads u
      then (r ⟨⟨g, h.1⟩, h.2⟩).setWidth (C.w g) else 0) u s.1.1).setWidth W

def Ordered : Prop :=
  ∀ g h : Fin C.N, ∀ u v : Fin n, g ∈ C.reads h → P.unit g = some u → P.unit h = some v → u ≤ v

/-- Every unit is then an interval of gates, numbered in layout order, as `Q_call` v1 cuts them. -/
def InLayoutOrder : Prop :=
  ∀ g h : Fin C.N, ∀ u v : Fin n, g ≤ h → P.unit g = some u → P.unit h = some v → u ≤ v

end Verity.Partition
