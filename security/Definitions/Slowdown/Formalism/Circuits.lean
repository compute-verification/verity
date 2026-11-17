module

public import Definitions.Core.Formalism

/-! Section 1 of the training formalism v2, circuits: a `Verity.Circuit` that is `Wellformed`, its gate sets, dependence and reconstruction. -/

@[expose] public section

namespace Slowdown.Formalism

open Finset
open Verity (Circuit)

structure Wellformed (C : Circuit) : Prop where
  boolean : C.Boolean
  fanIn : C.FanIn 2

/-- **`S` is a set of gates**: it holds no input. -/
def GateSet (C : Circuit) (S : Finset (Fin C.N)) : Prop := S ⊆ C.computed

def inputs (C : Circuit) : Finset (Fin C.N) := univ.filter fun v => C.isInput v = true

def AgreesOn (C : Circuit) (x y : C.Trace) (T : Finset (Fin C.N)) : Prop := ∀ v ∈ T, x v = y v

def DependsOn (C : Circuit) (g v : Fin C.N) : Prop := ∃ h, v ∈ C.reads h ∧ C.Feeds h g

inductive FeedsWithin (C : Circuit) (S : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {g : Fin C.N} : g ∈ S → FeedsWithin C S g g
  | step {h h' g : Fin C.N} : h ∈ C.reads h' → h ∈ S → FeedsWithin C S h' g → FeedsWithin C S h g

/-- **The reconstruction of `S` from `x`**: `S`'s gates evaluated once each in increasing order, each on the values its
predecessors left, starting from `x`; every value outside `S` keeps `x`'s. -/
def reconstruct (C : Circuit) (S : Finset (Fin C.N)) (x : C.Trace) : C.Trace :=
  (S.sort (· ≤ ·)).foldl (fun y g => Function.update y g (C.op g y)) x

end Slowdown.Formalism
