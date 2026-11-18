module

public import Definitions.Spot.Limits
public import Definitions.Core.LevelsCount

/-!
# The limits at a levels design

`Spot.Limits.LevelsCount`: at a `Verity.Levels` design whose last level's blocks (`Levels.blocks`) hold `P`'s proof
blocks under `blk`, the limits' count premise holds at the count `Levels.countAt L δ`. So `ExfiltrationLimit` and
`InfiltrationLimit` at `U = Finset G`, `L.design` and `k = L.countAt δ` take only their other premises, and their
figure `2^H · ∑_{j ≤ k} C(n, j)` counts over the last level's blocks. `InferenceOnlyLimit` takes no count.
-/

@[expose] public section

namespace Spot.Limits

open scoped ENNReal

/-- **The limits' count premise at a levels design** (stated): when `blk` places `P`'s proof blocks among the last
level's blocks, every set of proof blocks the levels' sample misses with probability at least `δ` has at most
`Levels.countAt L δ` members. Proved by `Spot.Limits.levelsCount`. -/
def LevelsCount : Prop :=
  ∀ {G : Type} [Fintype G] [DecidableEq G] (L : Verity.Levels G) {n : ℕ} (blk : Fin n ↪ Finset G),
    (∀ u, blk u ∈ L.blocks) → ∀ (δ : ℝ≥0∞) (B : Finset (Fin n)),
      δ ≤ L.design.missProb (B.map blk) → B.card ≤ L.countAt δ

end Spot.Limits
