import FlockSoundness.ComposePlace
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Vu

set_option Elab.async false

/-!
# P1 piece: a VU's columns are in the block, and its slots' constants read the pin

**To prove**, in a flat statement (`Layout st`, `Aliased.UnitFacts tags st`), at net `k`'s slot `i` of VU `g` for
`k ∈ vuNets` (every net but the row circuits):

* `vuSlot_lt`: every column of the slot up to its constant is in the block (`< 2 ^ kLog`);
* `const_rowA` / `const_rowB`: the slot's constant's row of `placedA` / `placedB` (Δ included) reads the pin alone.

**Twin of** `Layout.unit_slot_lt` (`ExecPlacement.lean`) and `Layout.placedA_const` / `Layout.placedB_const`
(`ComposePlace.lean`), which prove these for the unit's slots only.

**Route.** For `vuSlot_lt`: net `k`'s range is in `c.ranges` with kind `.net k` (`CheckFacts.one_range`, as
`Circuit.rangeOf` finds it), the slot index `g · perVu k + i` is below its count (`CheckFacts.count`), the column is
inside the slot (`Layout.slotLog`, `Layout.fit`), and the range inside the block (`Layout.bound`), as `unit_slot_lt`.
For the constants, as `placedA_const`: `rangesEntry_of_covers` with `Layout.const_covers`, the net's `ConstRow` (the
unit's is `Layout.unit_const`; each `vuNets` net's from `ParseFacts.nets`: `Net.ofRows` and `Lookup.build` both write
it, with every slot within `kLog ≤ 32`), and Δ at the constant: `constPairs`' `(const, const)` and `(const, pin)`
(`Layout.delta`; none when the constant is the pin), every other pair at an input bit (`InBit`), so not here
(`Layout.delta_unit_const`'s argument, at net `k`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

/-- **A VU slot's columns are in the block** (`Layout.unit_slot_lt` at every net of `vuNets`). -/
theorem vuSlot_lt {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) : st.c.slotCol k g i + x < 2 ^ st.c.kLog :=
  sorry

/-- **A VU slot's constant reads the pin in `A`** (`Layout.placedA_const` at every net of `vuNets`). -/
theorem const_rowA (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + (st.c.nets.getD k default).2.constPos) :
    placedA st r j = if j.val = st.pin then 1 else 0 := sorry

/-- **A VU slot's constant reads the pin in `B`** (`Layout.placedB_const` at every net of `vuNets`). -/
theorem const_rowB (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + (st.c.nets.getD k default).2.constPos) :
    placedB st r j = if j.val = st.pin then 1 else 0 := sorry

end FlockSoundness.Discharge.ScopeG
