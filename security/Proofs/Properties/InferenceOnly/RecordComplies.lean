module

public import Proofs.Properties.InferenceOnly.Record
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordFormCounts
public import Proofs.Properties.InferenceOnly.RecordServed

/-!
# Accepting a record gives the rate rule on its circuit, for the record check's own form

`form_complies` (`Properties.Record.Complies`): where the record check accepts a record, the record carries the
instance's fields and lays out a circuit, and every circuit it lays out meets the policy block by block at the
instance's `Γ` and `α`, so it complies. It is `complies_of` at the four statements of the check's own form: the
record carries the instance's fields (`form_owns`), it lays out a circuit (`form_laidOut`), the summary counts every
block (`form_counts`), and the rate rule holds on the summary (`form_rates`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record.SecurityProofs

open InferenceOnlyVerifier.Record.SecurityProofs (form form_owns form_laidOut form_laidOut_noEdge form_rates)

/-- **`ServedComplies` of a record that declares shared nodes and has no wires.** Where the record check accepts a
record, the record carries the instance's fields and lays out a circuit, and every circuit it lays out meets the
policy block by block at the instance's `Γ` and `α`, so it complies (`Properties.Record.Complies`), for every record
the check reads of a root no unit of which reads another unit's outputs (`NoEdgeReads`): its units read input nodes the
record declares once (`registered.nodes`) or runs of leaves of input nodes of their own, so its wiring has no wire.
A record with wires is not covered yet: the rows of a unit that reads another unit's outputs are its own input gates
or its writers' outputs only where `OwnRowsEdges` holds, which is open (LEAN-186). `FinishTemplates` is the wiring's
templates being the front's (`finish`). -/
theorem form_complies_noEdge (hNE : InferenceOnlyVerifier.Record.SecurityProofs.NoEdgeReads)
    (hFT : InferenceOnlyVerifier.Record.SecurityProofs.FinishTemplates) : Complies form :=
  complies_of form_owns (form_laidOut_noEdge hNE hFT) (form_counts_noEdge hNE) form_rates

/-- **Accepting a record gives the rate rule on its circuit**, for the record check's own form. -/
theorem form_complies (hNE : InferenceOnlyVerifier.Record.SecurityProofs.EdgeFreeReads) (hFT : InferenceOnlyVerifier.Record.SecurityProofs.FinishTemplates) :
    Complies form :=
  form_complies_noEdge (InferenceOnlyVerifier.Record.SecurityProofs.noEdgeReads_of_edgeFreeReads hNE) hFT

end InferenceOnlyVerifier.Properties.Record.SecurityProofs
