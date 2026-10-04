import FlockLevel3.Placed
import FlockSoundness.Compose
import FlockSoundness.Discharge.Aliased.Keys
import FlockSoundness.Discharge.Const.Inst
import FlockSoundness.Discharge.Placed.Wiring

set_option Elab.async false

/-!
# A VU's program: its units, its tail's stages and its lookup slots, in one order

`Placed.keyProg` runs `units_per_vu` copies of the unit net (`Placed.circuitRows`), each unit input on the input gate of
its Δ source (`Aliased.srcKey`). A unit cut input that a wire copies from a stage's output (`Aliased.Src.wire`) is then an
input gate no row of the VU computes, so a unit is correct for every value of it: an rmsnorm whose scale `s` is free.
And no order of whole nets fits a VU whose unit both feeds its tail and reads it back (rmsnorm: the unit's warp
aggregate, then `stage0`, `lookup:rsq`, `stage1`, then the unit's `s`), while `Prog.snoc` reads only earlier gates.

The block has an order of single columns, though: Δ makes a wired input row `dst` read `src` alone (its own `[dst]` and
Δ's `(dst, dst)` cancel, `HmRow.delta`), and a slot's constant read the pin. This file is that order:

* `vuRowList`: every column of the VU's slots of every net but the row circuits (`vuNets`), at VU 0, with its row
  (`slotRow`): the constant copies `one`, a wired input copies its wire's output column (`wireSrc`), every other row
  reads its net's `A` and `B` rows in its slot (a lookup slot's `B` also its table's entries, `lookupCols`), so an
  unwired input row outside `vuIns` must read nothing for `orderedB` to hold;
* `vuIns`: the VU's inputs, the VU-stacked unit inputs no wire copies (row words, leaf cuts, forced zeros, padding);
* `vuOrder`: the inputs, then Kahn's order of the other columns (`kahn`); `vuLogical` is the `Compose.Logical`;
* `orderedB`: a check of `Compose.Ordered` (`ordered_of_orderedB`);
* `vuProg`: `n` units of `Rows.compose vuLogical` on source-indexed input gates (`vuKeyIdx`), as `Placed.circuitProg`
  at the composed rows; `vuOutsOf`: the gates on the returned outputs' columns (`HmOut.outCol`: the unit's, or the
  tail's last stage's).

The VU's `wrong` is then a VU whose returned outputs are not its rows' evaluation on its row words.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Compose

/-! ## The VU's columns and rows, at VU 0 -/

/-- The nets a VU computes: every circuit of the block but the row circuits (`sha512x3`, `hm96`). -/
def vuNets (c : Circuit) : List ℕ :=
  (List.range c.nets.size).filter fun k => k != c.shaNet && k != c.hmNet

/-- A net's input columns per slot (`Rows.ofNet`'s `nIn`). -/
def netIn (net : Net) : ℕ := min (net.inWords * WORD) net.constPos

/-- **The column Δ copies into input column `x` of net `k`'s slot `i`**, at VU 0: the wire into its port's output
column, at the bit's offset (`HmRow.delta`'s wires); none for a column of no port or of no wire. -/
def wireSrc (c : Circuit) (k i x : ℕ) : Option ℕ :=
  let pc := portCols (c.nets.getD k default).2.inGroups
  match (List.range pc.size).find? (fun p => decide (pc[p]!.1 ≤ x ∧ x < pc[p]!.1 + pc[p]!.2)) with
  | none => none
  | some p => match c.wires.toList.find? (fun w => w.dst == (k, i, p)) with
    | none => none
    | some w => some (c.slotCol w.src.1 0 w.src.2.1 +
        ((portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1 + (x - pc[p]!.1))

/-- A lookup slot's row `x`'s `B` entries from its table (`FlockLevel3.lookupEntry`), as slot columns. -/
def lookupCols : Option LookupTable → ℕ → List ℕ
  | none, _ => []
  | some t, x =>
    ((List.range (2 ^ t.lo)).filter fun l => decide (lookupEntry (some t) x (t.loTop + l) = 1)).map (t.loTop + ·)

/-- **Net `k`'s slot `i`, column `x`, as the block holds it at VU 0**: the constant copies `one` (`none`, Δ's copy of
the pin); a wired input copies its wire's output column; every other row reads the net's `A` and `B` rows in the slot.
An unwired input row is then empty (a padding row), or reads itself (a port bit, or a padding row `Net.ofRows` let read
itself), which `orderedB` refuses unless the column is an input the VU's program takes (`vuIns`). -/
def slotRow (c : Circuit) (k i x : ℕ) : Option (List ℕ × List ℕ) :=
  let net := (c.nets.getD k default).2
  let base := c.slotCol k 0 i
  if x = net.constPos then none
  else match (if x < netIn net then wireSrc c k i x else none) with
    | some s => some ([s], [s])
    | none => some ((net.a.row x).toList.map (base + ·), ((net.b.row x).toList ++ lookupCols net.lookup x).map (base + ·))

/-- **The VU's columns and their rows**: every slot the VU holds of every net of `vuNets`, each slot's columns up to its
constant. -/
def vuRowList (c : Circuit) : List (ℕ × Option (List ℕ × List ℕ)) :=
  (vuNets c).flatMap fun k => (List.range (c.perVu.getD k 0)).flatMap fun i =>
    (List.range ((c.nets.getD k default).2.constPos + 1)).map fun x => (c.slotCol k 0 i + x, slotRow c k i x)

/-- The VU's columns, at VU 0. -/
def vuCols (c : Circuit) : List ℕ := (vuRowList c).map Prod.fst

/-- Column `p`'s row. -/
def vuRow (c : Circuit) (p : ℕ) : Option (List ℕ × List ℕ) := (List.lookup p (vuRowList c)).getD (some ([], []))

/-! ## The VU's inputs and order -/

/-- A unit copy's input columns, VU-stacked (`Aliased.srcKey`'s `W`). -/
def unitW (c : Circuit) : ℕ := c.unit.inWords * WORD

/-- A Δ source that is a circuit's output column. -/
def wiredKey : Aliased.Src → Bool
  | .wire _ _ _ => true
  | _ => false

/-- **The VU's inputs**: the VU-stacked unit inputs `k · W + x` no wire copies (a row word, a leaf cut, a forced zero,
padding), in stacked order. -/
def vuIns (c : Circuit) : List ℕ :=
  (List.range c.upv).flatMap fun k =>
    ((List.range (netIn c.unit)).filter fun x => !wiredKey (Aliased.srcKey c (k * unitW c + x))).map (k * unitW c + ·)

/-- A VU-stacked unit input's column, at VU 0. -/
def vuInCol (c : Circuit) (s : ℕ) : ℕ := c.slotCol c.unitNet 0 (s / unitW c) + s % unitW c

/-- The VU's input columns, at VU 0. -/
def vuInCols (c : Circuit) : List ℕ := (vuIns c).map (vuInCol c)

/-- The columns a row reads (none for a copy of `one`). -/
def readsOf (row : ℕ → Option (List ℕ × List ℕ)) (p : ℕ) : List ℕ :=
  match row p with
  | none => []
  | some r => r.1 ++ r.2

/-- **Kahn's order**: from `done`, append every column of `todo` whose reads are all done, in `todo`'s order, for at
most `fuel` rounds; what is left when no column is ready (a cycle) is appended as it is, which `orderedB` refuses. -/
def kahn (row : ℕ → Option (List ℕ × List ℕ)) : ℕ → List ℕ → List ℕ → List ℕ
  | 0, done, todo => done ++ todo
  | fuel + 1, done, todo =>
    match todo.partition fun p => (readsOf row p).all done.contains with
    | ([], rest) => done ++ rest
    | (ready, rest) => kahn row fuel (done ++ ready) rest

/-- **The VU's order**: its inputs, then Kahn's order of its other columns. -/
def vuOrder (c : Circuit) : List ℕ :=
  kahn (vuRow c) ((vuCols c).length + 1) (vuInCols c) ((vuCols c).filter fun p => !(vuInCols c).contains p)

/-- **The VU's rows in logical order.** -/
def vuLogical (c : Circuit) : Logical where
  nIn := (vuInCols c).length
  order := vuOrder c
  row := vuRow c

/-- **`Compose.Ordered`, checked**: at least `nIn` columns; each computed row reads earlier logical columns or `one`,
and only columns of the order. -/
def orderedB (L : Logical) : Bool :=
  decide (L.nIn ≤ L.order.length) &&
    (List.range (L.order.length - L.nIn)).all fun k =>
      ((lrow L k).1 ++ (lrow L k).2).all (fun x => decide (x < L.nIn + k ∨ x = L.order.length)) &&
        match L.row (L.order.getD (L.nIn + k) 0) with
        | none => true
        | some p => (p.1 ++ p.2).all fun x => L.order.contains x

theorem ordered_of_orderedB {L : Logical} (h : orderedB L = true) : Ordered L := by
  simp only [orderedB, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨hle, hk⟩ := h
  refine ⟨hle, fun k hk' x hx => (hk k hk').1 x hx, fun k hk' p hp x hx => ?_⟩
  have h2 := (hk k hk').2
  rw [hp] at h2
  simp only [List.all_eq_true, List.contains_iff_mem] at h2
  exact h2 x hx

/-- **Every VU input a row reads is a forced zero or a row's message bit** (`Layout.portReadsB` at the VU): a padding
input no row reads, and no input a fixed column. -/
def vuReadsB (c : Circuit) : Bool :=
  (vuIns c).all fun s => match Aliased.srcKey c s with
    | .zero => true
    | .msg _ _ _ => true
    | .pad _ _ => !(vuCols c).any fun p => (readsOf (vuRow c) p).contains (vuInCol c s)
    | _ => false

end FlockSoundness.Discharge.ScopeG
