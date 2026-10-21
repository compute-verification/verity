import Proofs.Flock.EndToEndRegisteredRows.Statement
import Proofs.Flock.EndToEndRegisteredRows.Reduce

/-! `Flock.SecurityProofs.EndToEndRegisteredRows : Flock.Guarantees.EndToEndRegisteredRows`: the reduction
`EndToEndRegisteredRows_of_rowsLink` at the rows' extraction bound `rowsLinkLe`. -/

/-- **C-Flock's one-stage audit, end to end, with the rows link, at the drawn units**
(`Flock.Guarantees.EndToEndRegisteredRows`). -/
theorem Flock.SecurityProofs.EndToEndRegisteredRows : Flock.Guarantees.EndToEndRegisteredRows :=
  FlockSoundness.Discharge.ZkReg.EndToEndRegisteredRows_of_rowsLink FlockSoundness.Discharge.ZkReg.rowsLinkLe
