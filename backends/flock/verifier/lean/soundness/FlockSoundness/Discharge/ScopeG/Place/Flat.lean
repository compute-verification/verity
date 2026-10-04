import FlockSoundness.Discharge.ScopeW.OutPairs
import FlockSoundness.Discharge.ScopeG.Vu

set_option Elab.async false

/-!
# P1 piece: a hidden statement reads its flat twin's rows at every row of a VU's slots

**To prove** (`vuRowsFlat`): an untyped statement `Stmt.setupHidden` accepted reads its flat twin's (`Hidden.flatTwin`,
Δ = `HmRow.delta`'s) `placedA` and `placedB` at every row of every slot of every net a VU computes (`vuNets`: all but
the row circuits `sha512x3` and `hm96`), the constant's row included.

**Twin of** `ScopeW.unitRowsFlat_any` (`Discharge/ScopeW/OutPairs.lean`), which proves this at the unit's slots only
(`HiddenExec.UnitRowsFlat`, `Discharge/HiddenExec/Flat.lean`).

**Route.** As `unitRowsFlat_any`: the hidden Δ is the flat Δ followed by `Hidden.outPairs` (`Hidden.hidden_delta'`), so
by `Hidden.deltaEntry_append_out` it is enough that no output pair's row is a row of net `k`'s slot. An output pair's row
is a row's message bit (`HiddenExec.outPairs_fst`, `Hidden.deltaMsgCol`), an input bit of a `sha512x3` or `hm96` slot
(`ScopeW.deltaMsgCol_ne_unitW` reads it so). That replaces `Layout.ne_unit_at` by its twin at net `k`: a column inside a
slot of the row circuits' ranges is in no slot of net `k ∉ {shaNet, hmNet}`'s range (`CheckFacts.disj`,
`CheckFacts.one_range`, `covers_slot`, the ranges' `Apart`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **A statement reads its flat twin's rows at every row of a VU's slots**: every net of `vuNets`, every VU, every
slot of the VU, every column up to the slot's constant. -/
def VuRowsFlat (st : Stmt) : Prop :=
  ∀ k ∈ vuNets st.c, ∀ g < st.c.g, ∀ i < st.c.perVu.getD k 0, ∀ x ≤ (st.c.nets.getD k default).2.constPos,
    ∀ r j : Fin (2 ^ st.c.kLog), r.val = st.c.slotCol k g i + x →
      placedA st r j = placedA (Hidden.flatTwin st) r j ∧ placedB st r j = placedB (Hidden.flatTwin st) r j

variable {tags : Tags} {st : Stmt} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray}

/-- **An untyped hidden statement reads its flat twin's rows at every row of a VU's slots** (`ScopeW.unitRowsFlat_any`
at every net of `vuNets`). -/
theorem vuRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : VuRowsFlat st := sorry

end FlockSoundness.Discharge.ScopeG
