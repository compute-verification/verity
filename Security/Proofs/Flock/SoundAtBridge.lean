import Proofs.Flock.SoundAtBridge.Statement
import Proofs.Flock.SoundAtBridge.Reduce

/-! `Flock.SecurityProofs.SoundAtBridgeStated : Flock.Guarantees.SoundAtBridgeStated`, by reflexivity: the guarantee is
there for its record, which holds `Flock.Guarantees.SoundAtBridge`'s text until its proof lands here. That proof is
`Flock.SoundAtBridge.soundAtBridge_of_rows Flock.SecurityProofs.EndToEndRegisteredRows`, once C-Flock's rows link
(`Proofs.Flock.EndToEndRegisteredRows`, PRF-98) is proved. -/

/-- **PoUW's proof premise from C-Flock's rows link is stated** (`Flock.Guarantees.SoundAtBridgeStated`). -/
theorem Flock.SecurityProofs.SoundAtBridgeStated : Flock.Guarantees.SoundAtBridgeStated := rfl
