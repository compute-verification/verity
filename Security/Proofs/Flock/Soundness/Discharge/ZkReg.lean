import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
import Proofs.Flock.Soundness.Discharge.ZkReg.Bind
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH
import Proofs.Flock.Soundness.Discharge.ZkReg.BindH
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedH
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.NonVacuousHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.Guarded
import Proofs.Flock.Soundness.Discharge.ZkReg.PublicInst
import Proofs.Flock.Soundness.Discharge.Composed.Sound
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundCustody

/-!
# One `--zk` sampled session with registered reads

`Defs`: `verify --zk --registered F`'s acceptance (`AcceptsZKR`) and what its registered check reads at the record's
draw (`OwnReads`, `regPaths`). `Bind`: the `--zk` rows without `pub.tables = none` (`rsZAtS`). `Checked`: the registered
check at an accepted outcome (`checked_of_accepts`). `Sound`: `zk_session_soundR`.

At hidden outputs (`Stmt.setupHidden`): `DefsH`, the registered reads at setupHidden's drawn statement (`OwnReadsH`,
`regPathsH`); `BindH`, the hidden-output `--zk` rows without `pub.tables = none` (`rsZHS`); `CheckedH`,
`checked_of_acceptsH`; `SoundH`: `zk_session_soundHR`; `SoundHCustody`: `zk_session_soundHR_custody`, under
`live-verifier`.

At hidden outputs and `J` tables (the production configuration): `DefsHJ`, the registered reads at the table each unit
is drawn into (`OwnReadsHJ`, `regPathsHJ`); `BindHJ`, the `J`-table rows without `pub.tables = none` (`rsZHJS`);
`CheckedHJ`, `checked_of_acceptsHJ`; `SoundHJ`: `zk_session_soundHJR`; `SoundHJCustody`: `drawOkZKHJ_of_file` and
`zk_session_soundHJR_custody`, under `live-verifier`; `NonVacuousHJ`: their reference setup and draw hypotheses
satisfiable together; `PublicHJ`: `zk_session_soundHJR_custody_reg`, with "wrong" measured against the public file,
and `zk_session_soundHJR_custody_drawn_reg`, at the drawn units; `PublicInst`: those two and `checked_of_acceptsHJ` at
the template instance's program (`instProg`), with no condition on the unit's input groups or on where the outputs are
returned.
-/
