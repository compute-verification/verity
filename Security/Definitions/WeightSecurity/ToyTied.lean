module

public import Definitions.WeightSecurity.Tied
public import Definitions.WeightSecurity.ToyServed

/-!
# Weight security with the releases tied to their traces: the toy instance

The instance `WeightSecurity.Guarantees.TheftServedTied`'s witnesses are stated at: the toy deployment
(`Definitions.WeightSecurity.ToyServed`) as a population with one unit, which has no input slot, reads nothing and
writes the one egress slot, the one position of the one charge unit, with the function that writes `0`. Its served
function reads nothing of the weights. The honest run's windows carry the honest assignment as their traces. The
leaking run's window `i` carries the weights' `i`-th value in the slot, and releases it.
-/

@[expose] public section

namespace WeightSecurity.ToyTied

open WeightSecurity.ToyServed

/-- **The toy as a population**: one unit, with no input slot, that reads nothing and writes the one egress slot. -/
def pop : Population.Dataflow Unit Unit where
  input := ∅
  reads _ := ∅
  writes _ := {()}

/-- The unit's function: it writes `0`. -/
def fns : pop.Fns (Fin 16) := fun _ _ _ => 0

/-- The egress slot: the one charge unit's one position. -/
def eg : Unit × Unit → Unit := fun _ => ()

/-- The population's inputs: it has none. -/
def inp (_ : W) (_ : Unit × Unit) (_ : Unit) : Fin 16 := 0

/-- Every party that queries the population's served function: the class of all parties. -/
def anyone : Set (Outsider Unit (Unit × Unit) (Unit → Unit → Fin 16) W) := Set.univ

/-- A window accepted on an empty sample, whose trace holds `v` in the slot, which it releases, with `k` incorrect
proof blocks recorded. -/
def window (k : ℕ) (v : Fin 16) : Served.Traced Unit (Unit → Fin 16) Unit Unit Unit (Fin 16) :=
  ⟨⟨True, False, k, fun _ _ => v, (), ()⟩, fun _ => v⟩

/-- The honest run with its traces: two windows whose traces are the honest assignment. -/
def honestRun (_ : W) (_ : Unit) : Verity.Game (List (Served.Traced Unit (Unit → Fin 16) Unit Unit Unit (Fin 16))) :=
  .ret [window 0 0, window 0 0]

/-- The leaking run with its traces at the verifier that accepts everything: window `i`'s trace holds the weights'
`i`-th value in the slot. -/
def leakRun (θ : W) (_ : Unit) : Verity.Game (List (Served.Traced Unit (Unit → Fin 16) Unit Unit Unit (Fin 16))) :=
  .ret [window 1 θ.1, window 1 θ.2]

/-- A user of the honest deployment: it asks twice and outputs `(0, 0)`. -/
def user : Outsider Unit (Unit × Unit) (Option ((Unit → Unit → Fin 16) × Unit)) W where
  query _ _ := ((), ())
  out _ _ := (0, 0)

/-- The thief: it asks twice and reads the weights off the two releases (its view is newest first). -/
def thief : Outsider Unit (Unit × Unit) (Option ((Unit → Unit → Fin 16) × Unit)) W where
  query _ _ := ((), ())
  out _ h := match h with
    | [(_, some (b, _)), (_, some (a, _))] => (a () (), b () ())
    | _ => (0, 0)

end WeightSecurity.ToyTied
