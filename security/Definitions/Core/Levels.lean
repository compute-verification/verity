module

public import Definitions.Core.Draw
public import Mathlib.Order.Partition.Finpartition

/-!
# Levels: a sample drawn level by level

A level (`Level`) is a partition of the gates `G` into blocks (`blocks`) with its rule for sampling them (`design`):
at each block `X` of the level above, a sampling design over blocks. Levels `(B₁, D₁), …, (B_L, D_L)` (`Levels`),
coarsest first, are nested (`nested`: every block of `B_ℓ` lies inside a block of `B_{ℓ-1}`, `B₀ = {univ}`), and at
each block `X` of `B_{ℓ-1}` the design `D_ℓ^X` draws only blocks of `B_ℓ` inside `X` (`inside`). A design is a function
of its block alone, so it is fixed in advance, and each is drawn on coins of its own, independent of every other.

The sample (`Levels.design`) is drawn level by level: `Q₀ = {univ}`, and `Q_ℓ` is the union over `X ∈ Q_{ℓ-1}` of a
draw from `D_ℓ^X` (`SamplingDesign.bind`). It is a sampling design over the blocks of `B_L`, so its miss probability
is `SamplingDesign.missProb` and its count curve `SamplingDesign.miss`. With two levels the blocks of `B₁` are the
outer blocks and those of `B₂` the inner blocks. One level's sample is its design `D₁` itself (`Levels.design_one`), and
two give `missProb D T = E_{Q₁ ∼ D₁}[∏_{X ∈ Q₁} missProb D₂^X (T_X)]` (`Levels.missProb_two`), in
`Proofs/Core/Levels.lean`.
-/

@[expose] public section

namespace Verity

namespace SamplingDesign

/-- The design that samples `S` on every coin. -/
def const {U : Type} (S : Finset U) : SamplingDesign U where
  Ω := Unit
  draw _ := S

/-- Draw from `D`, then at each sampled `u` from `F u` on coins of its own, and take the union. -/
def bind {U V : Type} [Fintype U] [DecidableEq U] [DecidableEq V] (D : SamplingDesign U)
    (F : U → SamplingDesign V) : SamplingDesign V where
  Ω := D.Ω × ((u : U) → (F u).Ω)
  draw ω := (D.draw ω.1).biUnion fun u => (F u).draw (ω.2 u)

end SamplingDesign

/-- **A level**: a partition of the gates into blocks, and at each block `X` of the level above, the design that
samples this level's blocks inside `X`. -/
structure Level (G : Type) [Fintype G] [DecidableEq G] where
  blocks : Finpartition (Finset.univ : Finset G)
  design : Finset G → SamplingDesign (Finset G)

namespace Level

variable {G : Type} [Fintype G] [DecidableEq G]

/-- Below the blocks `P`, every block of each level lies inside a block of the level above. -/
def NestedBelow : Finset (Finset G) → List (Level G) → Prop
  | _, [] => True
  | P, l :: ls => (∀ b ∈ l.blocks.parts, ∃ X ∈ P, b ⊆ X) ∧ NestedBelow l.blocks.parts ls

/-- Below the blocks `P`, at each block `X` of the level above, each level's design draws only its blocks inside
`X`. -/
def InsideBelow : Finset (Finset G) → List (Level G) → Prop
  | _, [] => True
  | P, l :: ls => (∀ X ∈ P, ∀ ω, ∀ b ∈ (l.design X).draw ω, b ∈ l.blocks.parts ∧ b ⊆ X) ∧
      InsideBelow l.blocks.parts ls

/-- The sample of the levels `ls` drawn level by level inside the block `X`. -/
def sampleBelow : List (Level G) → Finset G → SamplingDesign (Finset G)
  | [], X => SamplingDesign.const {X}
  | [l], X => l.design X
  | l :: ls, X => (l.design X).bind fun Y => sampleBelow ls Y

end Level

/-- **Levels** `(B₁, D₁), …, (B_L, D_L)`, coarsest first, below `B₀ = {univ}`. -/
structure Levels (G : Type) [Fintype G] [DecidableEq G] where
  levels : List (Level G)
  nested : Level.NestedBelow {Finset.univ} levels
  inside : Level.InsideBelow {Finset.univ} levels

/-- **The level-by-level sample** `Q_L`, a sampling design over the blocks of the last level. -/
def Levels.design {G : Type} [Fintype G] [DecidableEq G] (L : Levels G) : SamplingDesign (Finset G) :=
  Level.sampleBelow L.levels Finset.univ

end Verity
