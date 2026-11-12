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

`PooledLeakRegistered`'s honest lemma states its premises at a program of two inputs, over these and `certOps`:

* `twoLay`: no egress link, and step `t`'s ingress the input gate `0`;
* `twoObs`: one step's record over no egress link, every status complete.
-/

@[expose] public section

namespace NetTiming.Witness

/-- A layout of one link of one slot, with no gate. -/
def trivLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (0, []), fun _ => []⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- The network certifier's openings: the input and its negation, both `true`. -/
def certOps : List (Claims.Sampling.Opening Bool) := [⟨0, true, []⟩, ⟨1, true, []⟩]

/-- A layout of no egress link: step `t`'s ingress is the input gate `0`. -/
def twoLay : BoundaryGates 0 1 := ⟨fun _ ℓ _ => ℓ.elim0, fun _ => [0]⟩

/-- One step's record over no egress link: jitter class `0`, every status complete. -/
def twoObs : Obs 0 1 1 := ⟨fun ℓ => ℓ.elim0, 0, fun _ => .complete⟩

end NetTiming.Witness
