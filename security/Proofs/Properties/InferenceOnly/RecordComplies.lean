module

public import Proofs.Properties.InferenceOnly.Record
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordFormCounts
public import Proofs.Properties.InferenceOnly.RecordServed

/-! Accepting a record gives the rate rule on its circuit, for the record check's own form (`form_complies`). -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open InferenceOnlyCheck.Record.SecurityProofs (form form_owns form_laidOut form_rates)

theorem form_complies : Complies form := complies_of form_owns form_laidOut form_counts form_rates

end InferenceOnlyCheck.Properties.Record.SecurityProofs
