module

public import Proofs.Flock.Soundness.Discharge.PlainHJ.Defs
public import Proofs.Flock.Soundness.Discharge.PlainHJ.Decode
public import Proofs.Flock.Soundness.Discharge.PlainHJ.Bind
public import Proofs.Flock.Soundness.Discharge.PlainHJ.Sound
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
# One plain sampled session of `J` hidden-output tables (`flock-verify verify --plain-tables`, hidden outputs)

* `Defs.lean`: the hidden-output plan at the plain inputs, the coin server's draws and table `j`'s openings.
* `Decode.lean`: `setupOf_tablePHJ`, `accepts_reps_atH`, `decodesOnPHJ_exec` (R11b at every number of tables).
* `Bind.lean`: the sites, the layout and the rows at those tables, at the template instance's program `instProg` and,
  where the wires feed neither the unit nor the outputs, at the unit's key program `keyProg`.
* `Sound.lean`: `plain_session_soundHJ` (`keyProg`) and `plain_session_soundHJI` (`instProg`).
-/

@[expose] public section
