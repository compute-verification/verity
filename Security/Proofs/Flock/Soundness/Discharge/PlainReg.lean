module

public import Proofs.Flock.Soundness.Discharge.PlainReg.Defs
public import Proofs.Flock.Soundness.Discharge.PlainReg.BindH
public import Proofs.Flock.Soundness.Discharge.PlainReg.Checked
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

/-!
# One sampled session with registered reads, without `--zk`

`Defs`: `verify --registered F`'s acceptance without `--zk` (`AcceptsR`), which implies `Exec.Accepts`. `BindH`: the
hidden-output rows without `pub.tables = none` (`rsAtHS`), since the registered check accepts only a shared-row public
file. `Checked`: the registered check at an accepted hidden-output outcome of one table (`opens_of_acceptsR`), at the
unit's layout (`checked_of_acceptsRH`) and the template instance's (`checked_of_acceptsRHI`). The headlines over what
the verifier accepts are `FailClosed.PlainRegistered`'s.
-/

@[expose] public section
