module

public import Spot.PlanCheck
public import Definitions.Flock.Audit.Partitioning

/-! What a passing check of the planner's design (`Spot.PlanCheck`) establishes; none says the design is the
cheapest. -/

@[expose] public section

namespace Properties.Spot

open FlockSoundness.Audit FlockSoundness.Audit.Partitioning
open _root_.Spot.PlanCheck (Leak Rate Work leakBound checkWork)

/-- **The leak check is sound**: a design `leakBound` certifies at `v` bits has `den = k + 1` for some `k`, and its
Bernoulli sampling certifies at most `v` bits at `delta`. -/
def LeakCheckSound : Prop :=
  ∀ (d : Leak) (v : ℚ), leakBound d = some v →
    ∃ k : ℕ, d.den = k + 1 ∧ CertifiesLeak (SamplingDesign.bernoulli d.units d.num (k + 1)) (fun _ => d.width) d.delta v

/-- **The leak check certifies its count**: a design `leakBound` answers has `den = k + 1` for some `k`, and more than
`count` incorrect proof blocks escape its Bernoulli sampling with probability below `delta`, the count a write bound of
`width · count` bits reads. -/
def LeakCountSound : Prop :=
  ∀ (d : Leak) (v : ℚ), leakBound d = some v →
    ∃ k : ℕ, d.den = k + 1 ∧ (SamplingDesign.bernoulli d.units d.num (k + 1)).miss (d.count + 1) < ENNReal.ofReal d.delta

/-- **The work check is sound**: a design `checkWork` accepts has `den = k + 1` for some `k`, and however `n` units are
assigned to its strata, sampling each with its stratum's probability meets the target `target` at `delta`. -/
def WorkCheckSound : Prop :=
  ∀ d : Work, checkWork d = true →
    ∃ k : ℕ, d.den = k + 1 ∧ ∀ (n : ℕ) (σ : Fin n → Rate), (∀ u, σ u ∈ d.strata) →
      MeetsTarget (SamplingDesign.bernoulliOf (fun u => (σ u).num) (k + 1)) (fun u => ((σ u).work : ℝ)) d.target d.delta

end Properties.Spot
