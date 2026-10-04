import FlockSoundness.ExecTemplatePlace
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Place.Pos

set_option Elab.async false

/-!
# P1 piece: a lookup slot's table reads, as `lookupCols`' columns

**To prove** (`lookup_rowB`), in a flat statement (`Layout st`, `Aliased.UnitFacts tags st`), at net `k`'s slot `i` of
VU `g` for `k ∈ vuNets`, at any row `x` below the slot's constant: the slot's lookup share of the block's `B`
(`lookupAt`: `FlockLevel3.lookupEntry` read in the slot) is the count of `lookupCols net.lookup x` moved to the slot.
`lookupCols` (`Vu.lean`) lists the low minterms `loTop + l`, `l < 2 ^ lo`, whose `lookupEntry` is 1; `slotRow` puts them
on a lookup product row's `B` side. A net with no table (`lookup = none`) has both sides 0.

**Twin of** the `lookupEntry` half of `Layout.placedB_comp` (`ComposePlace.lean`), where the unit has no table
(`Layout.unit_lookup`) so the term is 0; nothing proves it for a net with a table yet.

**Route.** Unfold `lookupAt`, `lookupEntry` and `lookupCols`: `lookupEntry t x j'` is 0 unless
`t.loTop ≤ j' < t.loTop + 2 ^ t.lo`, and there it is 0 or 1. So both sides agree once the low minterms' window lies
inside the slot, `t.loTop + 2 ^ t.lo ≤ 2 ^ slotLog` (from `Lookup.build`, `ParseFacts.nets`, and `Layout.slotLog` /
`Layout.fit`). Then `count_slot` (`ExecTemplatePlace.lean`) moves the slot-local list into the slot.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

/-- **A lookup slot's table reads are `lookupCols`' columns**, moved to the slot. -/
theorem lookup_rowB {x : ℕ} (hx : x < (st.c.nets.getD k default).2.constPos) (j : ℕ) :
    lookupAt st.c k g i x j =
      ((((lookupCols (st.c.nets.getD k default).2.lookup x).map (st.c.slotCol k g i + ·)).count j : ℕ) : ZMod 2) :=
  sorry

end FlockSoundness.Discharge.ScopeG
