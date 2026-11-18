module

public import Definitions.Core.Levels

/-!
# The count at δ over levels

A sampled guarantee bounds the incorrect blocks by a count at `δ`: the largest set of blocks that the sample misses with
probability at least `δ` (`Spot.Sampling.IsHarmBound` at weight one, which `Spot.Bound.boundOf` computes for one
level). Over levels the blocks are the last level's (`Levels.blocks`), the units `Levels.design` draws; a set holding
anything else is never drawn from, so the count curve `SamplingDesign.miss` over every set says nothing there.

`Levels.IsCountBound L δ K`: every set of the last level's blocks missed with probability at least `δ` has at most `K`
blocks. `Levels.countAt L δ` is the least such `K`, and `Levels.shareAt L δ` the same as a share of the last level's
blocks. `Proofs/Core/LevelsCount.lean` proves that one level gives its design's count, that two levels are bounded
through each outer block's pass probability, and that both are antitone in `δ`.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- The blocks of the last level, the units the level-by-level sample draws; `{univ}` with no level. -/
def blocks (L : Levels G) : Finset (Finset G) :=
  match L.levels.getLast? with
  | some l => l.blocks.parts
  | none => {Finset.univ}

/-- **A count bound at `δ`**: every set of the last level's blocks that the sample misses with probability at least
`δ` has at most `K` blocks. -/
def IsCountBound (L : Levels G) (δ : ℝ≥0∞) (K : ℕ) : Prop :=
  ∀ T ⊆ L.blocks, δ ≤ L.design.missProb T → T.card ≤ K

open Classical in
/-- **The count at `δ`**: the largest set of the last level's blocks that the sample misses with probability at least
`δ`. -/
noncomputable def countAt (L : Levels G) (δ : ℝ≥0∞) : ℕ :=
  (L.blocks.powerset.filter fun T => δ ≤ L.design.missProb T).sup Finset.card

/-- **The share at `δ`**: the count at `δ` over the number of the last level's blocks. -/
noncomputable def shareAt (L : Levels G) (δ : ℝ≥0∞) : ℝ≥0∞ :=
  L.countAt δ / L.blocks.card

end Levels

end Verity
