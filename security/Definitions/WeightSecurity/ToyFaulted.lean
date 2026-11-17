module

public import Definitions.WeightSecurity.Faulted
public import Definitions.WeightSecurity.ToyServed

/-! The toy instance `TheftFaulted`'s witnesses are stated at, whose faulted function reads nothing of the weights. -/

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

/-- The patterns no cut covers at the cut that cuts nothing: every pattern. -/
abbrev FaultI := (Cut.none (Unit × Unit) (Unit → Fin 16)).Interiors Cf size none

/-- Every party that queries the faulted function at those patterns: the class of all parties. -/
def anyoneF : Set (Outsider Unit ((Unit × Unit) × FaultI) (Unit → Fin 16) W) := Set.univ

end WeightSecurity.ToyServed
