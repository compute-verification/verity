module

public import Definitions.Core.Levels

/-!
# The credited blocks of levels

A resource scheme credits blocks of the circuit, and their values must be committed before the sample: the sampling
bound holds only for values committed before the coins it draws, and a later level's values are committed after the
draw above it. So a credited block sits at the first level and no later level splits it (`Levels.CreditedUnsplit`): it
is a block of the first level (`Levels.CreditedFirst`) and of every later one. Its reads and writes are then boundary
values of the first level, and no level commits a value inside it. The levels setting takes the credited blocks `Cr`
as an input and these predicates hold of them.

Since each level's blocks lie inside the blocks of the level above, being a block of every later level is the same as
no later level having a block strictly inside it (`Levels.creditedUnsplit_iff_none_inside`, in
`Proofs/Core/LevelsCredited.lean`). One level constrains nothing (`Levels.creditedUnsplit_one`), and the sample
draws a credited block only whole (`Levels.CreditedUnsplit.drawn_eq`).
-/

@[expose] public section

namespace Verity

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **The credited blocks sit at the first level**: each block of `Cr` is a block of the first level. -/
def CreditedFirst (L : Levels G) (Cr : Finset (Finset G)) : Prop :=
  ∃ l ls, L.levels = l :: ls ∧ ∀ b ∈ Cr, b ∈ l.blocks.parts

/-- **The credited blocks are never split**: each block of `Cr` is a block of the first level and of every later
level, so the values it reads and writes are boundary values of the first level, committed before the sample. -/
def CreditedUnsplit (L : Levels G) (Cr : Finset (Finset G)) : Prop :=
  L.CreditedFirst Cr ∧ ∀ l ∈ L.levels.tail, ∀ b ∈ Cr, b ∈ l.blocks.parts

end Levels

end Verity
