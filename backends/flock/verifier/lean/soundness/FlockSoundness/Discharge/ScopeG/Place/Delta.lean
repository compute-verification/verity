import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Place.Pos

set_option Elab.async false

/-!
# P1 piece: Δ at a VU's input and computed rows, in a flat statement

**To prove**, in a flat statement (`Layout st`, `Aliased.UnitFacts tags st`: Δ is `HmRow.delta`'s, the circuit
untyped), at net `k`'s slot `i` of VU `g` for `k ∈ vuNets`:

* `delta_inA` / `delta_inB`: at a wired input row (`x < netIn`, `wireSrcG` names `s`), the net's own row there plus Δ's
  pairs read exactly `s`: the own row is `[x]` (a wired column is a port bit), Δ's `(x, x)` cancels it, Δ's `(x, s)`
  stays. `B` also carries the slot's lookup share (`lookupAt`), which is 0 at input rows.
* `delta_own`: at every other row below the constant (a computed row, or an input row no wire copies of a net other
  than the unit), Δ writes nothing, so the row reads its net's rows (`slotRowG`).

**Twin of** `Layout.delta_unit_comp` (`ExecPlacement.lean`) for `delta_own`, at every net instead of the unit; and, for
the wired input rows, of the unit-only copy facts `Layout.delta_port` / `Aliased.copyAt` read as matrix entries instead of
values.

**Route.** Δ exactly: `delta_flat` (`FlatDelta`: the slot constants' `constPairs`, the rows' copies, `Flat.leafPairs`,
`Flat.cutPairs`, `Flat.wirePairs`). Count `(r, j)` in each part: `constPairs` are at constants, the row copies at
`sha512x3` / `hm96` slots, leaf and cut pairs at the unit's leaf and cut bits (never a wired port, nor another net's
column: `CheckFacts.wire_leaf`, `CheckFacts.cut_leaf`, `CheckFacts.disj`), and `wirePairs` gives `(dst, dst)` and
`(dst, src)` once per wired bit (`CheckFacts.wired`: every non-leaf input port wired exactly once, the wire `find?`
returns). The net's own row at a port bit is `[x]` (`Net.ofRows`'s `inputsOk`, or `Lookup.build`'s input rows;
`ParseFacts.nets`). At a computed row, `Layout.delta`'s pairs past `constPairs` are input bits (`InBit`); at an unwired
input row of a net other than the unit no wire pair lands (a wire's destination is a port bit `wireSrcG` finds).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

/-- **`A` at a wired VU input row**: the net's own row plus Δ's pairs read exactly the wire's source column. -/
theorem delta_inA {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.a.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        deltaEntry st.da r.val j.val = (([s].count j.val : ℕ) : ZMod 2) := sorry

/-- **`B` at a wired VU input row**: the net's own row, the slot's lookup share and Δ's pairs read exactly the wire's
source column. -/
theorem delta_inB {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.b.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        lookupAt st.c k g i x j.val + deltaEntry st.db r.val j.val = (([s].count j.val : ℕ) : ZMod 2) := sorry

/-- **Δ writes nothing at a VU's other rows**: a computed row, or an unwired input row of a net other than the unit
(`Layout.delta_unit_comp` at every net of `vuNets`). -/
theorem delta_own {x : ℕ} (hx : x < (st.c.nets.getD k default).2.constPos)
    (hno : x < netIn (st.c.nets.getD k default).2 → wireSrcG st.c g k i x = none ∧ k ≠ st.c.unitNet)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    deltaEntry st.da r.val j.val = 0 ∧ deltaEntry st.db r.val j.val = 0 := sorry

end FlockSoundness.Discharge.ScopeG
