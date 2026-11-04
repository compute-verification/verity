import NetworkCheck.Period
import Definitions.NetworkCertifier.PeriodConditions

/-!
# A design that draws every proof block at one rate (`BernoulliDrawn`)

The sampled audit reads a Bernoulli law at the rate `p` over `population` units as no strata
(`SampledProofs.Bound.strataOf`), and publishes `bernoulliBound population p δ` as its bound on wrong units
(`SampledProofs.Bound.boundOf`): the largest `M ≤ population` with `(1 − p)^M ≥ δ`. `BernoulliDrawn d P population p`
is that law on the claims' side: the design `d` is well formed on the partition `P`, each of its rules draws its
class's proof blocks independently at `p` (`OneStage.bernoulli p.num p.den`), and `P` has at most `population` proof
blocks. At every such design `WrongBounds` holds at the published bound (`wrongBounds_of_bernoulliBound`, in
`Proofs.Properties.NetworkPeriod.Bernoulli`).
-/

namespace Properties.NetworkPeriod

open Claims.Sampling

/-- **The design draws every proof block of `P` independently at the rate `p`**, as the sampled audit's Bernoulli law
over `population` units does: `d` is well formed on `P`, each of its rules is `bernoulli p.num p.den`, and `P` has at
most `population` proof blocks. -/
def BernoulliDrawn (d : Design) (P : Partition) (population : ℕ) (p : Core.Ratio) : Prop :=
  d.WellFormed P ∧ (∀ r ∈ d.rules, r = .bernoulli p.num p.den) ∧ P.blocks.length ≤ population

end Properties.NetworkPeriod
