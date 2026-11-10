module

public import Proofs.Flock.Soundness.Discharge.PlainJ.Defs
public import Proofs.Flock.Soundness.Discharge.PlainJ.Decode
public import Proofs.Flock.Soundness.Discharge.PlainJ.Sound

/-!
# One plain sampled session of `J` tables (`flock-verify verify --plain-tables`)

* `Defs.lean`: the plan, sites, layout and rows at the plain inputs, the challenger's live-coin part's samples and table `j`'s openings.
* `Decode.lean`: `setupOf_tablePJ`, `decodesOnPJ_exec` (the decoder premise at every number of tables).
* `Sound.lean`: `plain_session_soundJ`.
-/

@[expose] public section
