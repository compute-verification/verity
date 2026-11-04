import Properties.SampledProofs.Draw
import Properties.NetworkPeriod.Bernoulli

/-!
# The Bernoulli draw the audit publishes is the one the draw ran (`BernoulliDerived`)

`BernoulliDerived` is `Properties.SampledProofs.StrataDerived`'s Bernoulli case: where the profile's draw of the law
`Draw.derive` expands is a Bernoulli draw at the rate `p` (`Record.profileDraw`, as `OneStage.check` reads it for its
bound), the design the draw runs (`Properties.SampledProofs.slotDesign`) draws the layout's population independently at
`p` and the input class whole (`BernoulliDrawn`). It takes no premise. It is stated here, not beside `StrataDerived`,
because `Properties.SampledProofs.Draw`'s layer imports nothing of the network period.
-/

open Lean (Json)

namespace Properties.NetworkPeriod

/-- **The Bernoulli draw the audit publishes is the one the draw ran**: wherever `Slots.layout` lays out `d` under
`query`, `slotDesign` gives the design `D`, and the profile's draw of the law `Draw.derive` expands from the rule's law
over the population `n` is a Bernoulli draw at the rate `p` (`Record.profileDraw`, as `OneStage.check` reads it for its
bound), `D` draws the layout's partition's population independently at `p`, its at most `n` proof blocks, and every other
class whole (`BernoulliDrawn`). -/
def BernoulliDerived : Prop :=
  ∀ (rule : SampledProofs.Select.Rule) (d : SampledProofs.Program) (query : Json) (table : Option Json)
    (l : SampledProofs.Slots.SlotOrder) (D : Claims.Sampling.Design) (n : ℕ) (named law : Json) (p : Core.Ratio),
    SampledProofs.Slots.layout d query = .ok l → Properties.SampledProofs.slotDesign rule d query table = .ok D →
    SampledProofs.Partition.units d query = .ok n → rule.lawOver n = .ok named →
    SampledProofs.Draw.derive d query named table = .ok law →
    SampledProofs.Record.profileDraw law = some (.bernoulli p) →
    BernoulliDrawn D (Properties.SampledProofs.slotPartition l) n p

end Properties.NetworkPeriod
