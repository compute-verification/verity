module

public import Proofs.Properties.InferenceOnly.RecordBlockFresh
public import Proofs.Properties.InferenceOnly.RecordBlockGates
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordFresh
public import Proofs.Properties.InferenceOnly.RecordFreshRows

/-! The record check's form counts every block (`form_counts`): a block reads at most the summary's fresh bits and has
its gates or none. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open InferenceOnlyCheck.Record.SecurityProofs (form FreshRows fresh_rows)

theorem counts_of_fresh (hFR : FreshRows) : Counts form := fun _ _ _ h _ hd u =>
  ⟨block_fresh hFR h hd u, block_gates h hd u⟩

theorem form_counts : Counts form := counts_of_fresh fresh_rows

end InferenceOnlyCheck.Properties.Record.SecurityProofs
