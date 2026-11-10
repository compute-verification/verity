module

public import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
public import Proofs.Flock.Soundness.Discharge.ZkReg.Bind
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH
public import Proofs.Flock.Soundness.Discharge.ZkReg.BindH
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Init.Data.Array.Basic
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Sound
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundCustody
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJSt
public import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody
public import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJSt
public import Proofs.Flock.Soundness.Discharge.ZkReg.NonVacuousHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.Guarded
public import Proofs.Flock.Soundness.Discharge.ZkReg.PublicInst

/-!
# One `--zk` sampled session with registered reads

`Defs`: `verify --zk --registered F`'s acceptance (`AcceptsZKR`) and what its registered check reads at the record's
sample (`OwnReads`, `regPaths`). `Bind`: the `--zk` rows without `pub.tables = none` (`rsZAtS`). `Checked`: the registered
check at an accepted outcome (`checked_of_accepts`). A session with registered reads is bounded only at hidden outputs
and `J` tables (`zk_session_soundHJR`, below).

At hidden outputs (`Stmt.setupHidden`): `DefsH`, the registered reads at setupHidden's sampled statement (`OwnReadsH`,
`regPathsH`); `BindH`, the hidden-output `--zk` rows without `pub.tables = none` (`rsZHS`).

At hidden outputs and `J` tables (the production configuration): `DefsHJ`, the registered reads at the table each unit
is sampled into (`OwnReadsHJ`, `regPathsHJ`); `BindHJ`, the `J`-table rows without `pub.tables = none` (`rsZHJS`);
`CheckedHJ`, `checked_of_acceptsHJ`; `SoundHJ`: `zk_session_soundHJR`; `SoundHJCustody`: `drawOkZKHJ_of_file` and
`zk_session_soundHJR_custody`, under `live-verifier`; `NonVacuousHJ`: their reference setup and sample hypotheses
satisfiable together; `PublicHJ`: `zk_session_soundHJR_custody_reg`, with "wrong" measured against the public file,
and `zk_session_soundHJR_custody_drawn_reg`, at the sampled proof blocks; `PublicInst`: those two and `checked_of_acceptsHJ` at
the template instance's program (`instProg`), with no condition on the unit's input groups or on where the outputs are
returned. `SoundHJSt` and `PublicHJSt`: `checked_of_acceptsHJ`, `zk_session_soundHJR` and the two `PublicHJ` theorems at
the session's statement circuit (`HmOut.parseStmt`), which a templated statement unrolls to.
-/

@[expose] public section
