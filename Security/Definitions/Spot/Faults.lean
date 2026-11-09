module

public import Definitions.Spot.Dataflow

/-!
# A population's runs with faults

SPOT's guarantee bounds the units an assignment gets wrong. The fault model reads those units' writes as faults:
inputs to the population, not its functions' values. A **fault pattern** (`Faults`) names, for each unit, nothing or
the values it writes instead. The **run with the inputs `a` and the pattern `φ`** (`RunOf`) holds `a` on the input
slots and the pattern's values on each faulted unit's writes, and makes every other unit correct; a checked population
has exactly one. `faultSet φ` is the units `φ` names and `faultCount φ` how many there are. `faultedOut f eg a φ` is
the run's values on the egress slots `eg`: the population's **faulted function**, what the circuit serves when its
incorrect proof blocks write the pattern's values.
-/

@[expose] public section

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

/-- **A fault pattern**: for each unit, nothing, or the values it writes in place of its function's. -/
abbrev Faults (V : Type) : Type := (u : U) → Option (P.writes u → V)

variable {V : Type}

/-- **A run with faults**: `σ` holds the inputs `a`, each unit the pattern `φ` names writes `φ`'s values, and every
other unit is correct. -/
def RunOf (f : P.Fns V) (a : S → V) (φ : P.Faults V) (σ : S → V) : Prop :=
  (∀ s ∈ P.input, σ s = a s) ∧ (∀ u w, φ u = some w → ∀ t : P.writes u, σ t = w t) ∧
    ∀ u, φ u = none → P.Correct f σ u

open Classical in
/-- The units a fault pattern names. -/
noncomputable def faultSet [Fintype U] (φ : P.Faults V) : Finset U :=
  univ.filter fun u => (φ u).isSome

/-- **A fault pattern's size**: how many units it names. -/
noncomputable def faultCount [Fintype U] (φ : P.Faults V) : ℕ := (P.faultSet φ).card

open Classical in
/-- **The population's faulted function**: the values on the slots `eg` of its run at the inputs `a` with the fault
pattern `φ`, and `a`'s values there where it has no run. A checked population has exactly one run. -/
noncomputable def faultedOut (f : P.Fns V) {O : Type} (eg : O → S) (a : S → V) (φ : P.Faults V) : O → V :=
  fun o => if h : ∃ σ, P.RunOf f a φ σ then Classical.choose h (eg o) else a (eg o)

end Population.Dataflow
