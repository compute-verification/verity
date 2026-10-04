import FlockSoundness.ExecTemplatePlace
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Place.Pos

set_option Elab.async false

/-!
# P1 piece: every VU net's rows at its slots

**To prove**, in a flat statement (`Layout st`, `Aliased.UnitFacts tags st`), at net `k`'s slot `i` of VU `g` for
`k ∈ vuNets` (every net but the row circuits: the unit, the tail's stages, the lookup slots), at any row `x` below the
slot's constant: the ranges' share of the block (`rangesEntry`, Δ apart) is the net's own row `x` moved to the slot,

* `stage_rowA`: in `A`, `(net.a.row x).map (slotCol k g i + ·)`;
* `stage_rowB`: in `B`, `(net.b.row x).map (slotCol k g i + ·)` plus the slot's lookup share `lookupAt` (`Lookup.lean`
  turns that into `lookupCols`' columns).

**Twin of** `Layout.placedA_comp` / `Layout.placedB_comp` (`ComposePlace.lean`), which prove this for the unit's slots
and computed rows only (and add Δ, which `Delta.lean` handles here).

**Route.** `rangesEntry_at` (`ExecTemplatePlace.lean`) reads `rangesEntry` in net `k`'s slot: it needs `CheckFacts.disj`,
net `k`'s range in `c.ranges` with kind `.net k` (`CheckFacts.one_range`, as `Circuit.rangeOf` finds it), and `AtSlot`
(slot `g · perVu k + i < count`: `CheckFacts.count`; `x < 2 ^ slotLog`: `Layout.slotLog`, `Layout.fit`). Then
`slotEntryA (.net k) x` is `sparseEntry net.a x`, which is the row's count: `sparseEntry_built_a` for a net
`Net.ofRows` built, and its twin for a `Lookup.build` net (`ParseFacts.nets` says each net is one of the two). Last,
`count_slot` (`ExecTemplatePlace.lean`) moves the row into the slot, given the row's columns are below `2 ^ slotLog`
(as `Layout.unit_row_lt` shows for the unit).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

/-- **The block's `A` (Δ apart) at a VU net's row**: the net's row, moved to the slot (`Layout.placedA_comp` at every
net of `vuNets`). -/
theorem stage_rowA {x : ℕ} (hx : x < (st.c.nets.getD k default).2.constPos) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + x) :
    rangesEntry st (slotEntryA st) r.val j.val =
      ((((st.c.nets.getD k default).2.a.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) := sorry

/-- **The block's `B` (Δ apart) at a VU net's row**: the net's row, moved to the slot, plus the slot's lookup share
(`Layout.placedB_comp` at every net of `vuNets`). -/
theorem stage_rowB {x : ℕ} (hx : x < (st.c.nets.getD k default).2.constPos) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + x) :
    rangesEntry st (slotEntryB st) r.val j.val =
      ((((st.c.nets.getD k default).2.b.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        lookupAt st.c k g i x j.val := sorry

end FlockSoundness.Discharge.ScopeG
