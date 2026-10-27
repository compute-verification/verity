import Definitions.Nci.Assumptions.CoveringTwoCosts

/-!
# Parameter-hardness at a price per weight

`ParameterHardAt` (`Definitions.Nci.Assumptions.CoveringTwoCosts`) charges every updated weight one cost `c_w`. A block may
complete whichever weights it likes, so that `c_w` is what the cheapest weights cost to rebuild, and on real models the
cheapest weights cost several times less than the mean. Here each weight `i` has its own price `p i`
(`ParameterHardPricedAt`) and a block pays the sum of the prices of
the weights it completes, which recovers about the mean. At `p = fun _ => c_w` it is `ParameterHardAt`. Activation-chaining
is unchanged: `ActivationChainedAt` at an activation cost `c_a`, and `𝓜*` is `chainedBlocksAt`.
-/

namespace Nci.Assumptions

open Verity Nci.Covering

/-- **Parameter-hardness at prices `p`** `(s, p)`: every subcircuit `S` of `d` with `|S| ≤ s` reads, in gates outside
the registered parameters, at least the summed price of the updated weights whose multiplications it completes,
`∑_{i ∈ θ'_S} p i ≤ |in(S) \ θ|`. `ParameterHardAt d B s c_w` is this at `p = fun _ => c_w`.

What it assumes: a block of at most `s` gates cannot rebuild a set of updated weights from fewer bits than their prices
add up to. A weight's price is what it carries beyond `θ`: read as a parameter slot counted from a step `t0` after the
commitment, its share of `H(θ'_{t0} | θ)` (the module doc of `Definitions.Nci.Assumptions.CoveringTwoCosts`). Prices have to
add up over every set of weights, not only one at a time: weights rebuilt together from one shared description, such as
the entries of a low-rank update or tied copies of one weight, are priced at their shares of that description, not each
at its own code length.

Where it fails: a weight priced above what a block can rebuild it from. Cheap weights the prover can rebuild — a weight
barely moved since `θ`, a copy of a registered weight or of another updated one, an update small enough for `s` gates to
compute — break it at any price above that cost. Price them at what they cost, zero if need be: the covering bound
(`Nci.Guarantees.CoveringBoundPriced`) asks only that any `D` weights together cost more than `k`, not that every price
be large. -/
def ParameterHardPricedAt (d : Declaration) (B : MulBlocks d) (s : ℕ) (p : Fin B.nw → ℝ) : Prop :=
  ∀ S : Subcircuit d.D, S.gates.card ≤ s → ∑ i ∈ B.weightsCompleted S, p i ≤ (S.ins \ d.θ).card

end Nci.Assumptions
