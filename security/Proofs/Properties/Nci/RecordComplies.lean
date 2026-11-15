module

public import Proofs.Properties.Nci.Record
public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordFormCounts
public import Proofs.Properties.Nci.RecordServed

/-!
# Accepting a record gives the rate rule on its circuit, for the record check's own form

`form_complies` (`Properties.Record.Complies`): where the record check accepts a record, the record carries the
instance's fields and lays out a circuit, and every circuit it lays out meets the policy block by block at the
instance's `Γ` and `α`, so it complies. It is `complies_of` at the four statements of the check's own form: the
record carries the instance's fields (`form_owns`), it lays out a circuit (`form_laidOut`), the summary counts every
block (`form_counts`), and the rate rule holds on the summary (`form_rates`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.Record.SecurityProofs (form form_owns form_laidOut form_rates)

/-- **Accepting a record gives the rate rule on its circuit**, for the record check's own form. -/
theorem form_complies : Complies form := complies_of form_owns form_laidOut form_counts form_rates

end NciCheck.Properties.Record.SecurityProofs
