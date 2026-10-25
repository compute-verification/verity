import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Lowers

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded; it is not proved. `Lowers`: the claims program `progOf C` lowers every unit of every
circuit (`Flock.Bridge.lowers`). -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
