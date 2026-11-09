module

public import Definitions.WeightSecurity.Tied
public import Definitions.WeightSecurity.ToyServed

/-!
# Weight security with the releases tied to their traces: the toy instance

The instance `WeightSecurity.Guarantees.TheftTied`'s witnesses are stated at: the toy deployment
(`Definitions.WeightSecurity.ToyServed`) as a population with one unit, which has no input slot, reads nothing and
writes the one egress slot, with the function that writes `0`. Its faulted function releases a fault pattern's value or
the honest `0`, and reads nothing of the weights. The honest run's windows carry the honest assignment as their traces.
The leaking run's window `i` carries the weights' `i`-th value in the slot, and releases it.
-/

@[expose] public section

namespace WeightSecurity.ToyServed

/-- **The toy as a population**: one unit, with no input slot, that reads nothing and writes the one egress slot. -/
def pop : Population.Dataflow Unit Unit where
  input := ∅
  reads _ := ∅
  writes _ := {()}

/-- The unit's function: it writes `0`. -/
def fns : pop.Fns (Fin 16) := fun _ _ _ => 0

/-- The egress slot. -/
def eg : Unit → Unit := id

/-- The population's inputs: it has none. -/
def inp (_ : W) (_ : Unit × Unit) (_ : Unit) : Fin 16 := 0

/-- Every party that queries the population's faulted function: the class of all parties. -/
def anyoneP : Set (Outsider Unit ((Unit × Unit) × pop.Faults (Fin 16)) (Unit → Fin 16) W) := Set.univ

/-- A window accepted on an empty sample, whose trace holds `v` in the slot, which it releases, with `k` incorrect
proof blocks recorded. -/
def window (k : ℕ) (v : Fin 16) : Served.Traced Unit (Fin 16) Unit Unit Unit :=
  ⟨⟨True, False, k, fun _ => v, (), ()⟩, fun _ => v⟩

/-- The honest run with its traces: two windows whose traces are the honest assignment. -/
def honestTraced (_ : W) (_ : Unit) : Verity.Game (List (Served.Traced Unit (Fin 16) Unit Unit Unit)) :=
  .ret [window 0 0, window 0 0]

/-- The leaking run with its traces: window `i`'s trace holds the weights' `i`-th value in the slot. -/
def leakTraced (θ : W) (_ : Unit) : Verity.Game (List (Served.Traced Unit (Fin 16) Unit Unit Unit)) :=
  .ret [window 1 θ.1, window 1 θ.2]

end WeightSecurity.ToyServed
