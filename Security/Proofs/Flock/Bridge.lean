import Proofs.Flock.Bridge.PortRoots
import Proofs.Flock.Bridge.Targets
import Proofs.Flock.Bridge.Canon
import Proofs.Flock.Bridge.OpeningsCanon

/-! `Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated`, by reflexivity: the statement
of `SampledKSLive` is recorded; it is not proved. -/

/-- **`SampledKSLive` is stated** (`Flock.Guarantees.SampledKSLiveStated`). -/
theorem Flock.SecurityProofs.SampledKSLiveStated : Flock.Guarantees.SampledKSLiveStated := rfl
