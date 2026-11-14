module

public import Proofs.Properties.InferenceOnly.RecordBlockFresh
public import Proofs.Properties.InferenceOnly.RecordBlockGates
public import Proofs.Properties.InferenceOnly.RecordForm
public import Proofs.Properties.InferenceOnly.RecordFresh
public import Proofs.Properties.InferenceOnly.RecordFreshRows

/-!
# The summary counts every block

`form_counts`: the record check's form counts every block (`Properties.Record.Counts`). In every circuit a record
the check reads lays out, a block reads at most the summary's fresh bits and has the summary's gates or none.

* The gates: `block_gates`.
* The fresh bits: a gate the block reads from outside itself that θ does not hold is charged to one row of the
  block's slot, a row the check counts neither as wired inside its slot nor as θ, and to no more bits than the
  row's (`block_fresh`); the bits of such rows of one slot are within the summary's fresh (`fresh_rows`).

`counts_of_fresh` is the statement with the second of those as a hypothesis (`FreshRows`).
-/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open InferenceOnlyCheck.Record.SecurityProofs (form FreshRows fresh_rows)

/-- **The summary counts every block**, where the fresh rows of a slot are within the summary's fresh: a block reads
at most the summary's fresh bits (`block_fresh`), and has the summary's gates or none (`block_gates`). -/
theorem counts_of_fresh (hFR : FreshRows) : Counts form := fun _ _ _ h _ hd u =>
  ⟨block_fresh hFR h hd u, block_gates h hd u⟩

/-- **The summary counts every block**: in every circuit a record the check reads lays out, each block reads at
most the summary's fresh bits, and has the summary's gates or none. -/
theorem form_counts : Counts form := counts_of_fresh fresh_rows

end InferenceOnlyCheck.Properties.Record.SecurityProofs
