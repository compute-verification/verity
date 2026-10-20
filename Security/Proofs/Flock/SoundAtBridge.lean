import Proofs.Flock.SoundAtBridge.Statement
import Proofs.Flock.SoundAtBridge.Reduce
import Proofs.Flock.EndToEndRegisteredRows

/-! `Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge`: the reduction `soundAtBridge_of_rows` at
C-Flock's proved rows link, `Flock.SecurityProofs.EndToEndRegisteredRows`. -/

/-- **PoUW's proof premise from C-Flock's rows link** (`Flock.Guarantees.SoundAtBridge`). -/
theorem Flock.SecurityProofs.SoundAtBridge : Flock.Guarantees.SoundAtBridge :=
  Flock.SoundAtBridge.soundAtBridge_of_rows Flock.SecurityProofs.EndToEndRegisteredRows
