import NetworkCheck.Period
import Definitions.NetworkCertifier.PeriodConditions

/-!
# A design that draws its population at one rate (`BernoulliDrawn`)

The sampled audit reads a Bernoulli law at the rate `p` over `population` units as no strata
(`SampledProofs.Bound.strataOf`), and publishes `bernoulliBound population p δ` as its bound on wrong units
(`SampledProofs.Bound.boundOf`): the largest `M ≤ population` with `(1 − p)^M ≥ δ`. `BernoulliDrawn d P population p`
is that law on the claims' side, at the shape the deployed draw runs (`Properties.SampledProofs.slotDesign`, the
population's law and then the input class drawn `all`): the design `d` is well formed on the partition `P`, each of
its rules draws its class's proof blocks independently at `p` (`OneStage.bernoulli a b`, `a/b` the rate as written,
`p` in any terms) or draws every one (`all`), and the proof blocks outside the classes drawn `all` number at most
`population`. At every such design `WrongBounds` holds at the published bound (`wrongBounds_of_bernoulliBound`, in
`Proofs.Properties.NetworkPeriod.Bernoulli`).
-/

namespace Properties.NetworkPeriod

open Claims.Sampling

/-- **The design draws `P`'s population independently at the rate `p`**, as the sampled audit's Bernoulli law over
`population` units does, and every other class whole: `d` is well formed on `P`, `p`'s denominator is positive, each
of `d`'s rules is `all` or `bernoulli a b` with `a/b = p`, and at most `population` of `P`'s proof blocks lie outside
the classes drawn `all` (`NetTiming.allSlots`). -/
def BernoulliDrawn (d : Design) (P : Partition) (population : ℕ) (p : Core.Ratio) : Prop :=
  d.WellFormed P ∧ 0 < p.den ∧ (∀ r ∈ d.rules, r = .all ∨ ∃ a b, r = .bernoulli a b ∧ a * p.den = p.num * b) ∧
    (Finset.range P.blocks.length \ NetTiming.allSlots d P).card ≤ population

end Properties.NetworkPeriod
