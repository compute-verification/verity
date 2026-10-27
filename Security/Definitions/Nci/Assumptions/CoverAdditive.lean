import Definitions.Nci.Assumptions.CoveringTwoCosts
import Mathlib.Data.Finset.Union

/-!
# The additive covering bound's premise: parameter-hardness outside the token computations

`Nci.Assumptions.ParameterHardAt` charges a block `c_w` bits outside the registered parameters for each updated weight
whose multiplications it completes, and activation-chaining charges it `c_a` of its inputs inside each token's
computation `T_j`. Both may be the same inputs, so `CoveringBoundTwoCosts` multiplies two bounds of `k` each.
`ConditionalParameterHardAt` pays the weights' bits outside every `T_j`, so the two charges add up to at most `k`, and
AM–GM gives a quarter of the product (`Nci.Cover.Guarantees.CoveringBoundAdditive`).

It is the cover theorem's `ConditionalUpdateHardness` (`Definitions.Nci.Assumptions.Cover`) in the covering
vocabulary (`Nci.Covering`), with the token computations as its regions: on a declaration's gates it implies it
(`Proofs.Nci.Cover.Gates`).
-/

namespace Nci.Cover.Assumptions

open Verity

/-- **Conditional parameter-hardness at weight cost `c_w`** `(s, c_w)` ([CONDITIONAL-UPDATE-HARDNESS]): every
subcircuit `S` of `d` with `|S| ≤ s` reads at least `c_w` bits outside the registered parameters and outside every token
computation for each updated weight whose multiplications it completes, `c_w · |θ'_S| ≤ |in(S) \ (θ ∪ ⋃_j T_j)|`. It
implies `Nci.Assumptions.ParameterHardAt d B s c_w`.

Why it's realistic: `T_j` is token `j`'s forward computation, its activations and the multiplications that make them.
An updated weight is computed from gradients, optimizer state and the previous weights, and a block that rebuilds it
reads those; the forward activations of a token carry what the weights did to that token, not the weights. It fails
wherever `ParameterHardAt` does, and also when the gates a block reads inside the token computations tell it about an
updated weight that it would otherwise have to read: a `T_j` drawn to include the backward pass's per-token gradients
would, since an updated weight is a sum of them. -/
def ConditionalParameterHardAt (d : Declaration) (B : Nci.Covering.MulBlocks d) (s : ℕ) (cw : ℝ) : Prop :=
  ∀ S : Nci.Covering.Subcircuit d.D, S.gates.card ≤ s →
    cw * (B.weightsCompleted S).card ≤ (S.ins \ (d.θ ∪ Finset.univ.biUnion B.T)).card

end Nci.Cover.Assumptions
