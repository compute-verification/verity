import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Place.Pos

set_option Elab.async false

/-!
# P1 piece: Δ at a VU's input and computed rows, in a flat statement

**To prove**, in a flat statement (`Layout st`, `Aliased.UnitFacts tags st`: Δ is `HmRow.delta`'s, the circuit
untyped), at net `k`'s slot `i` of VU `g` for `k ∈ vuNets`:

* `delta_inA` / `delta_inB`: at an input row `x < netIn`, the net's own row there plus Δ's pairs read exactly the column
  `wireSrcG` names (a wired input: its own `[x]` and Δ's `(x, x)` cancel, Δ's `(x, src)` stays), or nothing (a padding
  input: no row, no pair). A unit input must be a wired one (`hwire`): the unit's other inputs are the VU program's input
  gates, which no row computes. `B` also carries the slot's lookup share (`lookupAt`), which is 0 at input rows.
* `delta_comp`: at a computed row (`netIn ≤ x < constPos`), Δ writes nothing.

**Twin of** `Layout.delta_unit_comp` (`ExecPlacement.lean`) for `delta_comp`, at every net instead of the unit; and, for
the input rows, of the unit-only copy facts `Aliased.copyAt` / `HiddenExec.copyAt_blockH` read as matrix entries
instead of values.

**Route.** Δ exactly: `ScopeW.delta_flat_eqW` (`HmRow.delta`'s pairs are `ScopeW.flatPairsW`: the slot constants'
`constPairs`, the rows' copies, `Flat.leafPairs`, `Flat.cutPairs`, `Flat.wirePairs`). Count `(r, j)` in each part:
`constPairs` are at constants, the row copies at `sha512x3` / `hm96` slots, leaf and cut pairs at the unit's leaf and
cut bits (never a wired port: `CheckFacts.wire_leaf`, `CheckFacts.cut_leaf`), and `wirePairs` gives `(dst, dst)` and
`(dst, src)` once per wired bit (`CheckFacts.wired`: every non-row input port wired exactly once). The net's own row at
an input bit is `[x]` at a port bit and empty at padding (`Net.ofRows`'s `inputsOk`, or `Lookup.build`'s rows;
`ParseFacts.nets`). Rows of other slots are no row here (`CheckFacts.disj`). For `delta_comp`, `Layout.delta`'s pairs
past `constPairs` are input bits (`InBit`), and a column at or past its net's input rows is none.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

/-- **`A` at a VU input row**: the net's own row plus Δ's pairs read exactly `wireSrcG`'s column. -/
theorem delta_inA {x : ℕ} (hx : x < netIn (st.c.nets.getD k default).2)
    (hwire : k = st.c.unitNet → (wireSrcG st.c g k i x).isSome) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.a.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        deltaEntry st.da r.val j.val =
      (((wireSrcG st.c g k i x).toList.count j.val : ℕ) : ZMod 2) := sorry

/-- **`B` at a VU input row**: the net's own row, the slot's lookup share and Δ's pairs read exactly `wireSrcG`'s
column. -/
theorem delta_inB {x : ℕ} (hx : x < netIn (st.c.nets.getD k default).2)
    (hwire : k = st.c.unitNet → (wireSrcG st.c g k i x).isSome) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.b.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        lookupAt st.c k g i x j.val + deltaEntry st.db r.val j.val =
      (((wireSrcG st.c g k i x).toList.count j.val : ℕ) : ZMod 2) := sorry

/-- **Δ writes nothing at a VU's computed row** (`Layout.delta_unit_comp` at every net of `vuNets`). -/
theorem delta_comp {x : ℕ} (hlo : netIn (st.c.nets.getD k default).2 ≤ x)
    (hhi : x < (st.c.nets.getD k default).2.constPos) (r j : Fin (2 ^ st.c.kLog))
    (hr : r.val = st.c.slotCol k g i + x) :
    deltaEntry st.da r.val j.val = 0 ∧ deltaEntry st.db r.val j.val = 0 := sorry

end FlockSoundness.Discharge.ScopeG
