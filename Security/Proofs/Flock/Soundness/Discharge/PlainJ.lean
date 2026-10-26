import Proofs.Flock.Soundness.Discharge.PlainJ.Defs
import Proofs.Flock.Soundness.Discharge.PlainJ.Decode
import Proofs.Flock.Soundness.Discharge.PlainJ.Sound

/-!
# One plain sampled session of `J` tables (`flock-verify verify --plain-tables`)

* `Defs.lean`: the plan, sites, layout and rows at the plain inputs, the coin server's draws and table `j`'s openings.
* `Decode.lean`: `setupOf_tablePJ`, `decodesOnPJ_exec` (R11b at every number of tables).
* `Sound.lean`: `plain_session_soundJ`.
-/
