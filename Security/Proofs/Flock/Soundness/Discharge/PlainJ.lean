module

public import Proofs.Flock.Soundness.Discharge.PlainJ.Defs
public import Proofs.Flock.Soundness.Discharge.PlainJ.Decode
public import Proofs.Flock.Soundness.Discharge.PlainJ.Sound
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
# One plain sampled session of `J` tables (`flock-verify verify --plain-tables`)

* `Defs.lean`: the plan, sites, layout and rows at the plain inputs, the coin server's draws and table `j`'s openings.
* `Decode.lean`: `setupOf_tablePJ`, `decodesOnPJ_exec` (R11b at every number of tables).
* `Sound.lean`: `plain_session_soundJ`.
-/

@[expose] public section
