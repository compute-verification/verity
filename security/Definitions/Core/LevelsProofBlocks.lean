module

public import Definitions.Core.LevelsCount
public import Definitions.Claims.Sampling

/-!
# A claims partition's proof blocks as the last level's blocks

A claims partition (`Claims.Sampling.Partition`) numbers its proof blocks by slot, `0` to `|P.blocks| - 1`, class by
class and stratum by stratum (`Partition.blocks`), and the sampling claims read sets of slots: the incorrect set
(`Claims.Sampling.faulty`), its acceptance probability under the statement's sampling design (`Design.Pa`, read by the
confidence set `Design.M` and the miss `Design.miss`) and its size (`RCount`). Levels (`Verity.Levels`) draw blocks as
gate sets: the last level's blocks (`Levels.blocks`) are the units `Levels.design` samples, and the count at `δ`
(`Levels.countAt`) is over them. Two predicates join the two:

* `Levels.SlotsAreBlocks L P blk`: `blk` places `P`'s proof blocks as the blocks of `L`'s last level: it is injective
  on the slots, each slot's image is a block of the last level, and every block of the last level is a slot's image.
  So `blk` is a bijection from the slots onto `L.blocks`, nothing else. Its last clause is the form of the levels'
  accept bound's `hparts` (`Levels.prCoin_accept_le_late`, whose `blk` places a core partition's units the same way).
* `Levels.MissesAsDesign L d P blk`: `L`'s sample misses the blocks `blk` gives a set of slots with the acceptance
  probability the claims design `d` gives that set on `P` (`Claims.Sampling.Design.Pa d P`).

Under both, `Proofs/Core/LevelsProofBlocks.lean` carries the claims' slot sets across: an incorrect set's image is a
set of the last level's blocks of its size, missed with its `Design.Pa`, and the count at `δ` over slot sets is
`Levels.countAt`.
-/

@[expose] public section

namespace Verity

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **`blk` places the claims partition `P`'s proof blocks as `L`'s last level's blocks**: injective on `P`'s slots,
each slot's image a block of the last level, and every block of the last level a slot's image. -/
def SlotsAreBlocks (L : Levels G) (P : Claims.Sampling.Partition) (blk : ℕ → Finset G) : Prop :=
  Set.InjOn blk (Finset.range P.blocks.length : Set ℕ) ∧ (∀ v < P.blocks.length, blk v ∈ L.blocks) ∧
    ∀ X ∈ L.blocks, ∃ v < P.blocks.length, blk v = X

/-- **`L`'s sample misses the slots as the claims design `d` on `P` does**: every set `B` of `P`'s slots has its
blocks missed with the acceptance probability `Design.Pa d P B`. -/
def MissesAsDesign (L : Levels G) (d : Claims.Sampling.Design) (P : Claims.Sampling.Partition)
    (blk : ℕ → Finset G) : Prop :=
  ∀ B ⊆ Finset.range P.blocks.length, (L.design.missProb (B.image blk)).toReal = d.Pa P B

end Levels

end Verity
