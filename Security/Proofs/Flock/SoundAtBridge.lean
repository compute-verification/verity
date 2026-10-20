import Proofs.Flock.SoundAtBridge.Statement
import Proofs.Flock.SoundAtBridge.Reduce
import Proofs.Flock.SoundAtBridge.Served
import Proofs.Flock.SoundAtBridge.ServedProof
import Proofs.Flock.SoundAtBridge.NonVacuous
import Proofs.Flock.EndToEndRegisteredRows

/-! `Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge`: the reduction
`Flock.SoundAtBridge.soundAtBridge_of_rows` at C-Flock's rows link, `Flock.SecurityProofs.EndToEndRegisteredRows`.
`Flock.SecurityProofs.ServedRowsLinked : Flock.Guarantees.ServedRowsLinked`: `Flock.SoundAtBridge.servedRowsLinked`.
Its witness is `Flock.SoundAtBridge.soundAtBridge_nonvacuous`. -/

/-- **PoUW's proof premise from C-Flock's rows link** (`Flock.Guarantees.SoundAtBridge`). -/
theorem Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge :=
  Flock.SoundAtBridge.soundAtBridge_of_rows Flock.SecurityProofs.EndToEndRegisteredRows

/-- **C-Flock's rows link at served roots** (`Flock.Guarantees.ServedRowsLinked`). -/
theorem Flock.SecurityProofs.ServedRowsLinked : Flock.Guarantees.ServedRowsLinked :=
  Flock.SoundAtBridge.servedRowsLinked
