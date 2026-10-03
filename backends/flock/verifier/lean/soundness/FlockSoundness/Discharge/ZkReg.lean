import FlockSoundness.Discharge.ZkReg.Defs
import FlockSoundness.Discharge.ZkReg.Bind
import FlockSoundness.Discharge.ZkReg.Sound

/-!
# One `--zk` sampled session with registered reads

`Defs`: `verify --zk --registered F`'s acceptance (`AcceptsZKR`) and what its registered check reads at the record's
draw (`OwnReads`, `regPaths`). `Bind`: the `--zk` rows without `pub.tables = none` (`rsZAtS`). `Sound`:
`zk_session_soundR`.
-/
