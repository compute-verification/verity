module

public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Flock.SpotFlock.RunsAs.Parsed
import Proofs.Flock.Claims.Sampling.Select.Take

/-! `RuleIsTwo` at every reading: the form's parameter parser returns only a two-stage rule. -/

@[expose] public section

namespace Verity.SampledTwo

open Properties.Spot.SecurityProofs (bind_eq_ok)

theorem ruleIsTwo (C : CaseOf) : RuleIsTwo C := by
  intro o k p c e _ hdec
  obtain ⟨j, -, hj⟩ := bind_eq_ok.1 hdec.1
  exact SpotFlock.RunsAs.paramsOfJson_twoStage hj

end Verity.SampledTwo
