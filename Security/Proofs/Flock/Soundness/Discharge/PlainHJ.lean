import Proofs.Flock.Soundness.Discharge.PlainHJ.Defs
import Proofs.Flock.Soundness.Discharge.PlainHJ.Decode
import Proofs.Flock.Soundness.Discharge.PlainHJ.Bind
import Proofs.Flock.Soundness.Discharge.PlainHJ.Sound

/-!
# One plain sampled session of `J` hidden-output tables (`flock-verify verify --plain-tables`, hidden outputs)

* `Defs.lean`: the hidden-output plan at the plain inputs, the coin server's draws and table `j`'s openings.
* `Decode.lean`: `setupOf_tablePHJ`, `accepts_reps_atH`, `decodesOnPHJ_exec` (R11b at every number of tables).
* `Bind.lean`: the sites, the layout and the rows at those tables, at the template instance's program `instProg` and,
  where the wires feed neither the unit nor the outputs, at the unit's key program `keyProg`.
* `Sound.lean`: `plain_session_soundHJ` (`keyProg`) and `plain_session_soundHJI` (`instProg`).
-/
