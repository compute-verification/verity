import FlockSoundness.Discharge.ZkReg.DrawnRefines
import FlockSoundness.Discharge.ZkReg.RegLink
import FlockSoundness.Discharge.ZkReg.Vals
import FlockSoundness.Discharge.ZkReg.Joint
import FlockSoundness.Discharge.ZkReg.Off
import FlockSoundness.Discharge.ZkReg.Defs
import FlockSoundness.Discharge.ZkReg.Bind
import FlockSoundness.Discharge.ZkReg.Checked
import FlockSoundness.Discharge.ZkReg.Sound

/-!
# One `--zk` sampled session with registered reads

`Defs`: `verify --zk --registered F`'s acceptance (`AcceptsZKR`) and what its registered check reads at the record's
draw (`OwnReads`, `regPaths`). `Bind`: the `--zk` rows without `pub.tables = none` (`rsZAtS`). `Checked`: the registered
check at an accepted outcome (`checked_of_accepts`). `Sound`: `zk_session_soundR`.
-/
