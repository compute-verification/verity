module

public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Flock.SpotFlock.RunsAs.Parsed
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The form's decoded parameters are two-stage

`ruleIsTwo`: `RuleIsTwo` (`OfRun.lean`) at every reading. A case's parameters decode (`Decoded`) through the form's
parameter parser, which returns only a two-stage rule (`RunsAs.paramsOfJson_twoStage`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Properties.Spot.SecurityProofs (bind_eq_ok)

/-- **Every case's decoded parameters carry a two-stage rule** (`RuleIsTwo`). -/
theorem ruleIsTwo (C : CaseOf) : RuleIsTwo C := by
  intro o k p c e _ hdec
  obtain ⟨j, -, hj⟩ := bind_eq_ok.1 hdec.1
  exact SpotFlock.RunsAs.paramsOfJson_twoStage hj

end Verity.SampledTwo
