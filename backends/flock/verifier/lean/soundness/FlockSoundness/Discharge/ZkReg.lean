import FlockSoundness.Discharge.ZkReg.DrawnRefines
import FlockSoundness.Discharge.ZkReg.RegLink
import FlockSoundness.Discharge.ZkReg.Vals
import FlockSoundness.Discharge.ZkReg.ValsH
import FlockSoundness.Discharge.ZkReg.Joint
import FlockSoundness.Discharge.ZkReg.Off
import FlockSoundness.Discharge.ZkReg.Defs
import FlockSoundness.Discharge.ZkReg.Bind
import FlockSoundness.Discharge.ZkReg.Checked
import FlockSoundness.Discharge.ZkReg.Sound
import FlockSoundness.Discharge.ZkReg.DefsH
import FlockSoundness.Discharge.ZkReg.BindH
import FlockSoundness.Discharge.ZkReg.CheckedH
import FlockSoundness.Discharge.ZkReg.SoundH
import FlockSoundness.Discharge.ZkReg.SoundHCustody

/-!
# One `--zk` sampled session with registered reads

`Defs`: `verify --zk --registered F`'s acceptance (`AcceptsZKR`) and what its registered check reads at the record's
draw (`OwnReads`, `regPaths`). `Bind`: the `--zk` rows without `pub.tables = none` (`rsZAtS`). `Checked`: the registered
check at an accepted outcome (`checked_of_accepts`). `Sound`: `zk_session_soundR`.

At hidden outputs (`Stmt.setupHidden`): `DefsH`, the registered reads at setupHidden's drawn statement (`OwnReadsH`,
`regPathsH`); `BindH`, the hidden-output `--zk` rows without `pub.tables = none` (`rsZHS`); `CheckedH`,
`checked_of_acceptsH`; `SoundH`: `zk_session_soundHR`; `SoundHCustody`: `zk_session_soundHR_custody`, under
`live-verifier`.
-/
