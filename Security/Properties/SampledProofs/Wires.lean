module

public import SampledProofs.Wires
public import Definitions.SampledProofs.Dataflow

/-!
# Sampled proofs' wires: the dataflow a wiring check accepts is acyclic (`SampledProofs.Wires.check`)

The property of the verifier's wiring check (`SampledProofs.Wires.check`) as the sampled-proofs service runs it, stated
against the function for every wiring its decoder produces. It takes no premise.

A wiring's dataflow (`wiresDataflow`) is a population (`Population.Dataflow`) whose units are its templates' units
`(template, unit)` and whose slots are their output rows `(template, unit, output)`. Each unit writes its template's
outputs and reads the output rows its wired rows hold, one for each entry of a wire (`Wire.entries`). A unit's other
input rows are no unit's writes, so they add no dependency, and the population leaves them out (`input` is empty).

`WiresAcyclic`: the dataflow of a wiring `check` accepts is acyclic (`Population.Dataflow.Acyclic`): no unit depends on
itself through a chain of units each reading what the one before writes.
-/

@[expose] public section

open SampledProofs.Wires

namespace Properties.SampledProofs

/-- A wiring's dataflow: units `(template, unit)` over the output rows `(template, unit, output)`; each unit writes its
template's outputs and reads the writer's rows of its wired rows. -/
def wiresDataflow (c : Wiring) : Population.Dataflow (String × Int) (String × Int × String) where
  input := ∅
  reads y := ((c.wires.flatMap Wire.entries).filterMap fun e =>
    if (e.1.1, (e.1.2.1 : Int)) = y then some e.2 else none).toFinset
  writes x := ((((c.templates.lookup x.1).map (·.outs)).getD []).map fun o => (x.1, x.2, o.1)).toFinset

/-- The dataflow of a wiring `check` accepts is acyclic. -/
def WiresAcyclic : Prop :=
  ∀ (c : Wiring) (S : Summary), check c = .ok S → (wiresDataflow c).Acyclic

end Properties.SampledProofs
