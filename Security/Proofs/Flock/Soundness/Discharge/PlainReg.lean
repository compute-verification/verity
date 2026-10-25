import Proofs.Flock.Soundness.Discharge.PlainReg.Defs
import Proofs.Flock.Soundness.Discharge.PlainReg.BindH
import Proofs.Flock.Soundness.Discharge.PlainReg.Checked
import Proofs.Flock.Soundness.Discharge.PlainReg.CheckedHJ

/-!
# One sampled session with registered reads, without `--zk`

`Defs`: `verify --registered F`'s acceptance without `--zk` (`AcceptsR`), which implies `Exec.Accepts`. `BindH`: the
hidden-output rows without `pub.tables = none` (`rsAtHS`), since the registered check accepts only a shared-row public
file. `Checked`: the registered check at an accepted hidden-output outcome of one table (`opens_of_acceptsR`), at the
unit's layout (`checked_of_acceptsRH`) and the template instance's (`checked_of_acceptsRHI`). `CheckedHJ`: the same at
`J` tables under `--plain-tables` (`checked_of_acceptsRHJ`, `checked_of_acceptsRHJI`), at `PlainHJ`'s layouts, whose rows
read no `pub.tables = none`. The headlines over what the verifier accepts are `FailClosed.PlainRegistered`'s, and at `J`
tables `FailClosed.SoundExecHJR`.
-/
