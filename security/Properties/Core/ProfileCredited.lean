module

public import Definitions.Core.LevelsCredited
public import CoreProtocol.Profile

/-!
# A profile's credited blocks are unsplit

`Core.Integrity.creditedProblem` (`verity/protocols/interface/CoreProtocol/Profile.lean`) is the code's refusal of a
profile's credited blocks, by which SPOT's check (`Spot.OneStage.check`) and every reader of a profile
(`Core.Integrity.ofJson`) refuse. `Unsplit` says that what it doesn't refuse is `Verity.Levels.CreditedUnsplit` of the
levels the profile records (`Records`): each credited block is a block of the first level and of every later one.
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

/-- **What the credited refusal accepts is `CreditedUnsplit`**: for levels `L` that a profile `p` of at least one
level records, and credited blocks `Cr` of the first level that `num` numbers among `p.credited`, a profile whose
credited blocks `creditedProblem` doesn't refuse has `L.CreditedUnsplit Cr`. -/
def Unsplit : Prop :=
  ∀ (p : Integrity) (G : Type) [Fintype G] [DecidableEq G] (L : Verity.Levels G) (Cr : Finset (Finset G))
    (num : Finset G → ℕ), p.levels ≠ #[] → Records p L → (∀ b ∈ Cr, b ∈ above L 1 ∧ num b ∈ p.credited) →
    p.creditedProblem = none → L.CreditedUnsplit Cr

end Core.Integrity
