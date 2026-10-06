import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
import Proofs.Flock.Soundness.Discharge.ZkReg.Bind
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Soundness.Discharge.ZkReg.Sound
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH
import Proofs.Flock.Soundness.Discharge.ZkReg.BindH
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedH
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundH
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHCustody
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
import Proofs.Flock.Soundness.Discharge.ZkReg.NonVacuousHJ
import Proofs.Flock.Soundness.Discharge.ZkRegV2.Sound
import Proofs.Flock.Soundness.Discharge.ZkRegV2.SoundH
import Proofs.Flock.Soundness.Discharge.ZkRegV2.SoundHCustody
import Proofs.Flock.Soundness.Discharge.ZkRegV2.SoundHJ
import Proofs.Flock.Soundness.Discharge.ZkRegV2.SoundHJCustody
import Proofs.Flock.Soundness.Discharge.ZkRegV2.NonVacuous
import Proofs.Flock.Soundness.Discharge.ZkRegV2.NonVacuousHJ

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
satisfiable together.

At `sha512/row/v2` rows, the rows the registered check admits (`ZkRegV2/`): `scopeOkV2` and, at hidden outputs,
`scopeOkHV2` in place of `Layout.scopeOk`, which admits `sha512/row/v1` rows only; `zk_session_soundRV2`,
`zk_session_soundHRV2`, `zk_session_soundHR_custodyV2`, `zk_session_soundHJRV2` and `zk_session_soundHJR_custodyV2`;
`ZkRegV2.NonVacuous`: their circuit premises and the registered check's port test hold together, and at
`Layout.scopeOk` the registered check refuses every registered port; `ZkRegV2.NonVacuousHJ`: the `J`-table statements'
reference setup and draw hypotheses satisfiable together at `scopeOkHV2`.
-/
