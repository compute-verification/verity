module

public import Proofs.Properties.Nci.RecordBlockFresh
public import Proofs.Properties.Nci.RecordBlockGates
public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordFreshRows

/-!
# The summary counts every block

`counts_of_fresh`: the record check's form counts every block (`Properties.Record.Counts`), where the fresh rows of
a slot are within the summary's fresh (`FreshRows`). In every circuit a record the check reads lays out, a block
reads at most the summary's fresh bits (`block_fresh`) and has the summary's gates or none (`block_gates`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.Record.SecurityProofs (form FreshRows)

/-- **The summary counts every block**, where the fresh rows of a slot are within the summary's fresh: a block reads
at most the summary's fresh bits (`block_fresh`), and has the summary's gates or none (`block_gates`). -/
theorem counts_of_fresh (hFR : FreshRows) : Counts form := fun _ _ _ h _ hd u =>
  ⟨block_fresh hFR h hd u, block_gates h hd u⟩

end NciCheck.Properties.Record.SecurityProofs
