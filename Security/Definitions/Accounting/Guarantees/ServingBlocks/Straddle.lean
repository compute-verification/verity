import Definitions.Accounting.ServingBlocks.Straddle

/-!
# The uncertified cap across the certified boundary

Beside `UncertifiedElementary`, which caps what the inference blocks leave of a unit by the operations split between
them: four caps on what blocks `B i` leave of a set of gates `R` with uncertified gates `U = uncertified R B`
(`Accounting.ServingBlocks.Straddle`). At the dark remainder there are no blocks and `U` is all of it; at a serving
block holding uncertified gates beside its inference blocks, `U` is those gates.

* `UncertifiedStructural`: the uncertified cap `ω_max μ |U| / s_×` under `NoStraddle`, the premise the cap was read
  under, made exact. An operation whose product a block computes and whose accumulation `U` does refutes it
  (`UncertifiedStructural_nontrivial`).
* `StraddleBound`: with no premise on straddling, the straddling operations weigh at most `ω_max μ` times the bits
  crossing the certified boundary (`crossing`). Of a straddling operation with its representative in `U`, take the last
  gate outside `U`: another gate of the operation reads it, a later one, which is in `U`, so it is an unregistered input
  of `U`. With its representative in a block, it has an unregistered input of that block the same way.
* `UncertifiedCrossing`: the cap that replaces the premise, `ω_max μ (|U| / s_× + crossing)`. With no blocks it is the
  most an uncertified region `R` can be assigned: `ω_max μ (|R| / s_× + |in(R) \ θ|)`.
* `UncertifiedLocal`: under `UncertifiedBodySize`, the weakest premise that keeps the cap at `ω_max μ |U| / s_×`, plus
  `ω_max μ Σ_i |in(B i) \ θ|` for the operations whose representatives are in blocks.

Each has an honest and nontrivial lemmas (`Security/Definitions/Accounting/Guarantees/ServingBlocks/Straddle/Witnesses.lean`).
The proofs are in `Security/Proofs/Accounting/ServingBlocks/Straddle.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The uncertified cap under no straddling**: with weights at most `ω_max`, every gate in at most `μ` operations'
bodies and every body of at least `s > 0` gates, when no operation assigned to `R` and computed wholly inside none of
the blocks `B i` has a gate outside `R`'s uncertified gates `U`, what the blocks leave of `R` weighs at most
`ω_max μ |U| / s`. -/
def UncertifiedStructural : Prop :=
  ∀ (C : Circuit) (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N))
    (ωmax : ℝ) (μ s : ℕ), 0 < s → 0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) → Overlap J μ → BodySize J s →
    NoStraddle J R B → UncertifiedCap J R B (ωmax * μ * (uncertified R B).card / s)

/-- **The straddle bound**: when the operations' bodies lead to their representatives with each gate in at most `μ` of
them and the weights are at most `ω_max`, the operations assigned to `R`, computed wholly inside none of the blocks
`B i` and not wholly inside `R`'s uncertified gates weigh at most `ω_max μ` times the bits crossing the certified
boundary, `|in(U) \ θ| + Σ_i |in(B i) \ θ|`. -/
def StraddleBound : Prop :=
  ∀ (d : Declaration) (J : Counted d.D) (R : Finset (Fin d.D.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin d.D.N)) (ωmax : ℝ) (μ : ℕ),
    OperationBodies d J → Overlap J μ → 0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) →
      J.straddle R B ≤ ωmax * μ * crossing d R B

/-- **The uncertified cap across the certified boundary**: when the operations' bodies lead to their representatives,
each gate is in at most `μ` of them, every body has at least `s > 0` gates and the weights are at most `ω_max`, what the
blocks `B i` leave of `R` weighs at most `ω_max μ (|U| / s + |in(U) \ θ| + Σ_i |in(B i) \ θ|)`, `U` being `R`'s
uncertified gates. -/
def UncertifiedCrossing : Prop :=
  ∀ (d : Declaration) (J : Counted d.D) (R : Finset (Fin d.D.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin d.D.N)) (ωmax : ℝ) (μ s : ℕ), 0 < s →
    OperationBodies d J → Overlap J μ → BodySize J s → 0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) →
      UncertifiedCap J R B (ωmax * μ * ((uncertified R B).card / s + crossing d R B))

/-- **The uncertified cap under uncertified body size**: when the operations' bodies lead to their representatives, each
gate is in at most `μ` of them, the weights are at most `ω_max`, and every operation whose representative is among
`R`'s uncertified gates `U` has at least `s > 0` of its gates in `U`, what the blocks `B i` leave of `R` weighs at most
`ω_max μ (|U| / s + Σ_i |in(B i) \ θ|)`. -/
def UncertifiedLocal : Prop :=
  ∀ (d : Declaration) (J : Counted d.D) (R : Finset (Fin d.D.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin d.D.N)) (ωmax : ℝ) (μ s : ℕ), 0 < s →
    OperationBodies d J → Overlap J μ → UncertifiedBodySize J R B s → 0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) →
      UncertifiedCap J R B (ωmax * μ * ((uncertified R B).card / s + ∑ i, (d.unregistered (B i)).card))

end Accounting.Guarantees.ServingBlocks
