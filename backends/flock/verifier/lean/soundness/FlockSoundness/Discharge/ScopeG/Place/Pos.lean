import FlockSoundness.Discharge.ScopeG.Vu

set_option Elab.async false

/-!
# A VU's columns at VU `g`

`vuLogical`'s columns are block columns at VU 0: net `k`'s slot `i`, column `x` is `c.slotCol k 0 i + x` (`vuRowList`).
VU `g` holds the same slots at `c.slotCol k g i + x` (`Circuit.slotCol`: slot `g · perVu k + i` of net `k`'s range).
These are the definitions the placement of `vuLogical` at VU `g` (`Place.lean`) and its pieces (`Place/*.lean`) share:

* `vuPos c g`: a VU-0 column moved to VU `g`, the `pos` at which `Compose.placement_of_realizes` places `vuLogical`;
* `wireSrcG`, `slotRowG`: `wireSrc` and `slotRow` at VU `g` (`wireSrcG_zero`, `slotRowG_zero`: at VU 0 they are those);
* `lookupAt`: a lookup slot's table share of the block's `B` at a row of net `k`'s slot `i` of VU `g`
  (`rangesEntry`'s `lookupEntry` term, `FlockLevel3.slotEntryB`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

/-- Every column of the VU's slots, at VU 0 and at VU `g`, slot by slot as `vuRowList` lists them. -/
def vuColMap (c : Circuit) (g : ℕ) : List (ℕ × ℕ) :=
  (vuNets c).flatMap fun k => (List.range (c.perVu.getD k 0)).flatMap fun i =>
    (List.range ((c.nets.getD k default).2.constPos + 1)).map fun x => (c.slotCol k 0 i + x, c.slotCol k g i + x)

/-- **A VU-0 column at VU `g`** (`vuColMap`); a column of no VU slot stays where it is. -/
def vuPos (c : Circuit) (g p : ℕ) : ℕ := ((vuColMap c g).lookup p).getD p

/-- **`wireSrc` at VU `g`**: the column Δ copies into input column `x` of net `k`'s slot `i` of VU `g`. -/
def wireSrcG (c : Circuit) (g k i x : ℕ) : Option ℕ :=
  let pc := portCols (c.nets.getD k default).2.inGroups
  match (List.range pc.size).find? (fun p => decide (pc[p]!.1 ≤ x ∧ x < pc[p]!.1 + pc[p]!.2)) with
  | none => none
  | some p => match c.wires.toList.find? (fun w => w.dst == (k, i, p)) with
    | none => none
    | some w => some (c.slotCol w.src.1 g w.src.2.1 +
        ((portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1 + (x - pc[p]!.1))

theorem wireSrcG_zero (c : Circuit) : wireSrcG c 0 = wireSrc c := rfl

/-- **`slotRow` at VU `g`**: net `k`'s slot `i` of VU `g`, column `x`, as the block holds it. -/
def slotRowG (c : Circuit) (g k i x : ℕ) : Option (List ℕ × List ℕ) :=
  let net := (c.nets.getD k default).2
  let base := c.slotCol k g i
  if x = net.constPos then none
  else match (if x < netIn net then wireSrcG c g k i x else none) with
    | some s => some ([s], [s])
    | none => some ((net.a.row x).toList.map (base + ·), ((net.b.row x).toList ++ lookupCols net.lookup x).map (base + ·))

theorem slotRowG_zero (c : Circuit) : slotRowG c 0 = slotRow c := rfl

/-- **A lookup slot's table share of `B`** at row `x` of net `k`'s slot `i` of VU `g`, column `j`: `rangesEntry`'s
`lookupEntry` term (`FlockLevel3.slotEntryB`), read in the slot. -/
def lookupAt (c : Circuit) (k g i x j : ℕ) : ZMod 2 :=
  if j / 2 ^ (c.rangeOf k).slotLog = (c.rangeOf k).pos0 + (g * c.perVu.getD k 0 + i) then
    lookupEntry (c.nets.getD k default).2.lookup x (j % 2 ^ (c.rangeOf k).slotLog)
  else 0

end FlockSoundness.Discharge.ScopeG
