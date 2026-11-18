module

public import Definitions.Core.Scheme
public import CoreProtocol.Profile

/-!
# A profile's credited blocks are unsplit

`Core.Integrity.creditedProblem` (`verity/protocols/interface/CoreProtocol/Profile.lean`) is the code's refusal of a
profile's credited blocks, by which SPOT's check (`Spot.OneStage.check`) and every reader of a profile
(`Core.Integrity.ofJson`) refuse. `Unsplit` says that when a profile's credited numbers are those of a placed
scheme's credited blocks at the first level (`Verity.Scheme.Placed.credited`), what it doesn't refuse is
`Verity.Scheme.Placed.Unsplit` under the levels the profile records (`Records`).
-/

@[expose] public section

namespace Core.Integrity

/-- The blocks above level `i` of `L` (from 0): the whole gate set above the first level, else the level above's
blocks. -/
def above {G : Type} [Fintype G] [DecidableEq G] (L : Verity.Levels G) : ℕ → Finset (Finset G)
  | 0 => {Finset.univ}
  | i + 1 => match L.levels[i]? with
    | some l => l.blocks.parts
    | none => ∅

/-- `p` records the levels `L`: as many levels, and inside each block of the level above, as many blocks of each
level as `p`'s size of it. -/
def Records {G : Type} [Fintype G] [DecidableEq G] (p : Integrity) (L : Verity.Levels G) : Prop :=
  L.levels.length = p.levels.size ∧
    ∀ i (l : Verity.Level G) (q : Core.Level), L.levels[i]? = some l → p.levels[i]? = some q →
      ∀ X ∈ above L i, (l.blocks.parts.filter (· ⊆ X)).card = q.size

/-- **What the credited refusal accepts is credited unsplit**: for levels `L` that a profile `p` of at least one
level records, and a placed scheme `P` whose credited blocks are blocks of the first level whose numbers under `num`
are `p.credited`, a profile whose credited blocks `creditedProblem` doesn't refuse has `P.Unsplit L`. -/
def Unsplit : Prop :=
  ∀ (p : Integrity) (S : Verity.Scheme) (G : Type) [Fintype G] [DecidableEq G] (P : S.Placed G)
    (L : Verity.Levels G) (num : Finset G → ℕ), p.levels ≠ #[] → Records p L →
    (∀ b ∈ P.credited, b ∈ above L 1 ∧ num b ∈ p.credited) → (∀ c ∈ p.credited, ∃ b ∈ P.credited, num b = c) →
    p.creditedProblem = none → P.Unsplit L

end Core.Integrity
