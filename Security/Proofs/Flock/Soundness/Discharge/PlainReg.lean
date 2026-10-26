import Proofs.Flock.Soundness.Discharge.PlainReg.Defs
import Proofs.Flock.Soundness.Discharge.PlainReg.BindH
import Proofs.Flock.Soundness.Discharge.PlainReg.Checked

/-!
# One sampled session with registered reads, without `--zk`

`Defs`: `verify --registered F`'s acceptance without `--zk` (`AcceptsR`), which implies `Exec.Accepts`. `BindH`: the
hidden-output rows without `pub.tables = none` (`rsAtHS`), since the registered check accepts only a shared-row public
file. `Checked`: the registered check at an accepted hidden-output outcome of one table (`opens_of_acceptsR`), at the
unit's layout (`checked_of_acceptsRH`) and the template instance's (`checked_of_acceptsRHI`). The headlines over what
the verifier accepts are `FailClosed.PlainRegistered`'s.
-/
