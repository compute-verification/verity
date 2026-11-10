module

public import Definitions.NetworkCertifier.SampledAudit

/-!
# The layouts, openings and operator of the network certifier's witnesses (trusted definitions; used only by the
satisfiability witnesses)

The nontrivial lemmas of `Guarantees/SampledAudit`, `Guarantees/SampledCharge`, `Guarantees/Checked` and `Period` state
their conclusions false at an audit that accepts everything, over these:

* `trivLay`: one link of one slot with no gate, at which `SampledAuditAccepted`'s conclusion fails;
* `notLay`: step `t`'s ingress is the input, and its egress slot's presence gate the input's negation, at which the
  leak bounds fail;
* `certOps`: the network certifier's openings, the input and its negation both `true`, at which `PeriodLeakAccepted`'s
  bound fails.

The honest lemma of the period's operator premise (`PeriodOperator`) is at `readsBefore`, a period of two windows'
operator whose second window reads the first's observations.
-/

@[expose] public section

namespace NetTiming.Witness

/-- A layout of one link of one slot, with no gate. -/
def trivLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (0, []), fun _ => []⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- The network certifier's openings: the input and its negation, both `true`. -/
def certOps : List (Claims.Sampling.Opening Bool) := [⟨0, true, []⟩, ⟨1, true, []⟩]

/-- A period of two windows' operator that reads the window before: the first window's gives no ingress, and the
second's one bit, whether the first window observed no step. -/
def readsBefore : Fin 2 → (Fin 2 → List (Obs 1 1 1)) → ℕ → List (Obs 1 1 1) → List Bool :=
  fun i o _ _ => if i = 0 then [] else [(o 0).isEmpty]

end NetTiming.Witness
