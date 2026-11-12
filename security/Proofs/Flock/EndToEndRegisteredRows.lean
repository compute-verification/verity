module

public import Proofs.Flock.EndToEndRegisteredRows.Statement
public import Proofs.Flock.EndToEndRegisteredRows.Reduce
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! `Flock.SecurityProofs.EndToEndRegisteredRows : Flock.Guarantees.EndToEndRegisteredRows`: the reduction
`EndToEndRegisteredRows_of_rowsLink` at the rows' extraction bound `rowsLinkLe`. -/

@[expose] public section

/-- **Flock's one-stage audit, end to end, with the rows link, at the sampled proof blocks**
(`Flock.Guarantees.EndToEndRegisteredRows`). -/
theorem Flock.SecurityProofs.EndToEndRegisteredRows : Flock.Guarantees.EndToEndRegisteredRows :=
  FlockSoundness.Discharge.ZkReg.EndToEndRegisteredRows_of_rowsLink FlockSoundness.Discharge.ZkReg.rowsLinkLe

/-- **`EndToEndRegisteredRows` at a flat circuit file** (`Flock.Guarantees.EndToEndRegisteredRowsFlat`): it at
`Desc.single c`. -/
theorem Flock.SecurityProofs.EndToEndRegisteredRowsFlat : Flock.Guarantees.EndToEndRegisteredRowsFlat := by
  intro n Reg own I ht c hc
  exact Flock.SecurityProofs.EndToEndRegisteredRows n Reg own I ht (.single c) (Flock.Desc.parse_single hc)
