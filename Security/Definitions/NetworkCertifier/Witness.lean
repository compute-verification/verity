module

public import Definitions.NetworkCertifier.SampledAudit

/-!
# The layouts and openings the network certifier's guarantees fail at (trusted definitions; used only by the
satisfiability witnesses)

The nontrivial lemmas of `Guarantees/SampledAudit`, `Guarantees/SampledCharge`, `Guarantees/Checked` and `Period` state
their conclusions false at an audit that accepts everything, over these:

* `trivLay`: one link of one slot with no gate, at which `SampledAuditAccepted`'s conclusion fails;
* `notLay`: step `t`'s ingress is the input, and its egress slot's presence gate the input's negation, at which the
  leak bounds fail;
* `certOps`: the network certifier's openings, the input and its negation both `true`, at which `PeriodLeakAccepted`'s
  bound fails.
-/

@[expose] public section

namespace NetTiming.Witness

/-- A layout of one link of one slot, with no gate. -/
def trivLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (0, []), fun _ => []⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- The network certifier's openings: the input and its negation, both `true`. -/
def certOps : List (Claims.Sampling.Opening Bool) := [⟨0, true, []⟩, ⟨1, true, []⟩]

end NetTiming.Witness
