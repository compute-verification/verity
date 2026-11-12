module

public import Spot.PlanCheck
public import Definitions.Flock.Audit.Partitioning

/-!
# The planner's checker: what its acceptance means (`Spot.PlanCheck`)

The planner (`optimize.py`) picks a one-stage design in floating point; `Spot.PlanCheck` checks it in exact
arithmetic. These properties say what a check that passes establishes, against the partitioning targets
(`Definitions.Flock.Audit.Partitioning`) that `audit_exfiltration` and the free-compute guarantees read:

* `LeakCheckSound`: when `leakBound d` returns `v`, sampling the design's `units` units independently with probability
  `num / den` (`Law.bernoulli`) certifies at most `v` bits at `delta`, every unit exporting at most `width` bits
  (`CertifiesLeak`).
* `LeakCountSound`: there, more than `count` incorrect units escape that sampling with probability below `delta`.
* `WorkCheckSound`: when `checkWork d` accepts, every assignment of units to the design's strata, each unit sampled
  independently with its stratum's probability (`Law.bernoulliOf`), meets the target: every set of incorrect units doing
  work at least `target` escapes with probability at most `delta` (`MeetsTarget`).

Neither says the design is the cheapest: that is the planner's claim, which nothing here checks.

Each is about a Bernoulli design, whose rate is each unit's own probability whatever population is registered. The
audit and the stream's run refuse a registration under one (`Spot.Draw.bernoulli`), so none of these reads a served
window's rule; the count a served window's rule names over its population is `RuleAtRate`'s (`Properties.Spot.Rate`).
-/

@[expose] public section

namespace Properties.Spot

open FlockSoundness.Audit FlockSoundness.Audit.Partitioning
open _root_.Spot.PlanCheck (Leak Rate Work leakBound checkWork)

/-- **The leak check is sound**: a design `leakBound` certifies at `v` bits has `den = k + 1` for some `k`, and its
Bernoulli sampling certifies at most `v` bits at `delta`. -/
def LeakCheckSound : Prop :=
  ∀ (d : Leak) (v : ℚ), leakBound d = some v →
    ∃ k : ℕ, d.den = k + 1 ∧ CertifiesLeak (Law.bernoulli d.units d.num (k + 1)) (fun _ => d.width) d.delta v

/-- **The leak check certifies its count**: a design `leakBound` answers has `den = k + 1` for some `k`, and more than
`count` incorrect units escape its Bernoulli sampling with probability below `delta`, the count a write bound of
`width · count` bits reads. -/
def LeakCountSound : Prop :=
  ∀ (d : Leak) (v : ℚ), leakBound d = some v →
    ∃ k : ℕ, d.den = k + 1 ∧ (Law.bernoulli d.units d.num (k + 1)).miss (d.count + 1) < ENNReal.ofReal d.delta

/-- **The work check is sound**: a design `checkWork` accepts has `den = k + 1` for some `k`, and however `n` units are
assigned to its strata, sampling each with its stratum's probability meets the target `target` at `delta`. -/
def WorkCheckSound : Prop :=
  ∀ d : Work, checkWork d = true →
    ∃ k : ℕ, d.den = k + 1 ∧ ∀ (n : ℕ) (σ : Fin n → Rate), (∀ u, σ u ∈ d.strata) →
      MeetsTarget (Law.bernoulliOf (fun u => (σ u).num) (k + 1)) (fun u => ((σ u).work : ℝ)) d.target d.delta

end Properties.Spot
