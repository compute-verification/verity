module

public import Definitions.WeightSecurity.Faulted
public import Definitions.WeightSecurity.ToyServed

/-!
# Weight security under SPOT's fault model: the toy instance

The instance `WeightSecurity.Guarantees.TheftFaulted`'s witnesses are stated at: the toy deployment
(`Definitions.WeightSecurity.ToyServed`), whose one proof block writes its one egress slot. A fault pattern is `none`,
of size `0`, or the value the block writes instead, of size `1`, and the faulted function releases that value or the
honest `0`. The faulted function reads nothing of the weights, so faulted use steals them no more often than honest use.
The leaking run's window `i` is the faulted run with the weights' `i`-th value as its one fault.
-/

@[expose] public section

namespace WeightSecurity.ToyServed

/-- A fault pattern: the value the one proof block writes instead, if it is faulted. -/
abbrev Fault := Option (Fin 16)

/-- A pattern's size: `1` if it faults the block. -/
def size : Fault → ℕ
  | none => 0
  | some _ => 1

/-- The faulted served function: the block's fault value, or the honest `0`. -/
def Cf (_ : W) (x : (Unit × Unit) × Fault) (_ : Unit) : Fin 16 := x.2.getD 0

/-- Every party that queries the faulted function: the class of all parties. -/
def anyoneF : Set (Outsider Unit ((Unit × Unit) × Fault) (Unit → Fin 16) W) := Set.univ

end WeightSecurity.ToyServed
