import PouwWindow.Record
import PouwWindow.Shape
import PouwWindow.Side

/-!
# The window statement (`window.py`, `window_problem`)

L4 on the committed entry rows and the side rows' index fields: `bound` rows, each an entry row of the layout whose
public fields are the shape's (`shapeEntries`); every call's -h3 root is the pass's; in the fields (per call and then the
filler, each side's `[index, ok, admit, blank]` per row, `side_field`) each row's index is its leaf, its ok, admit and
blank are bits, the filler's rows are not admitted, and a row that is not admitted is the canonical empty row (blank, ok
0: the emptiness check); and the registered work is `credit` of the masked sum Σ over the calls of
(Σ ok_A·admit_A)·(Σ ok_B·admit_B) (`selected`): tile (i, j) is credited iff admit_i ∧ ok_i ∧ admit_j ∧ ok_j. Given the
calls' records it holds each to its entry (`recordProblem`). A problem is a value (`some why`); a refusal (`.error`) is
where Python raises `ValueError`, or would raise another error on input off its types.
-/

namespace PouwWindow

open Lean (Json)

/-- One side's rows' fields, `[index, ok, admit, blank]` each, and both sides of an entry. -/
abbrev Fields := List (List Int)

private def fieldsBad : String :=
  "the side rows' fields are, per call and then the filler, each side's [index, ok, admit, blank] per row"

private def sideFields? (f : Json) (side : Side) : Except String Fields := do
  let some (.arr rows) := pyGet? f side.name | throw fieldsBad
  rows.toList.mapM fun p => match p with
    | .arr xs => match xs.toList.mapM strictInt? with
      | some ints => if ints.length == 4 then pure ints else throw fieldsBad
      | none => throw fieldsBad
    | _ => throw fieldsBad

/-- `_fields`: the side rows' fields as `window_problem` reads them, `count` entries of both sides' fields. -/
def fieldsOf? (j : Json) (count : Nat) : Except String (List (Fields × Fields)) := do
  let .arr fs := j | throw fieldsBad
  if fs.size != count then throw fieldsBad
  fs.toList.mapM fun f => match f with
    | .obj _ => do pure (← sideFields? f .a, ← sideFields? f .b)
    | _ => throw fieldsBad

/-- `selected`: a side's selected rows, Σ ok·admit over its rows' fields. -/
def selected (rows : Fields) : Int := (rows.map fun p => p[1]! * p[2]!).sum

/-- `window_problem`: the window statement on the entry rows and the side rows' fields, or `none`. -/
def windowProblem (rows : List ByteArray) (work : Work) (bound : Int) (rootH3 : ByteArray) (shape : List Call)
    (fields : Json) (bits : Nat := gridBits) (records : Option (List Json) := none) : Except String (Option String) := do
  if rootH3.size != 64 then throw "the pass's -h3 tile root is 64 bytes"
  if let some why := shapeProblem shape bound then throw s!"the shape is not a window's: {why}"
  let fields ← fieldsOf? fields (shape.length + 1)
  if (rows.length : Int) != bound then return some s!"{rows.length} entries, not the bound {bound}"
  let want := shapeEntries shape rows.length
  let mut entries : Array Entry := #[]
  for (row, e) in rows.zipIdx do
    match Entry.fromRow? row with
    | .error x => return some s!"entry {e}: not an entry row: {x}"
    | .ok entry =>
      if let some why := entry.layoutProblem then return some s!"entry {e}: not the layout: {why}"
      if entry.publicFields != (want.getD e { kind := .none }).publicFields then
        return some s!"entry {e}: its kind, range, rows, tile0, cols and m are not the shape's"
      if entry.kind == .call && entry.rootH3 != rootH3 then
        return some s!"entry {e}: its -h3 root is not the pass's tile commitment"
      entries := entries.push entry
  let mut real : Int := 0
  for ((entry, (fa, fb)), e) in (entries.toList.zip fields).zipIdx do
    for (side, got) in [(Side.a, fa), (Side.b, fb)] do
      let count := entry.nSide side
      if got.map (·[0]!) != (List.range count).map Int.ofNat then
        return some s!"entry {e}: its side-{side.name} rows' indices are not their leaves 0 to {(count : Int) - 1}"
      if got.any fun p => (p.drop 1).any (!isBit ·) then
        return some s!"entry {e}: a side-{side.name} row's ok, admit or blank is not 0 or 1"
      if entry.kind == .filler && got.any (·[2]! != 0) then
        return some s!"entry {e}: the filler's side-{side.name} row is admitted"
      for (p, i) in got.zipIdx do
        if p[2]! == 0 && (p[1]! != 0 || p[3]! == 0) then
          return some s!"entry {e}: side-{side.name} row {i} is not admitted and not the canonical empty row"
    if entry.kind == .call then real := real + selected fa * selected fb
  let n := shapeTiles shape
  if real < 1 then
    return some "no tile is credited: no call has an admitted slot and weight row that both passed their forming"
  let got ← credit? bits real n
  if got != work then
    return some s!"{real} credited tiles of {n} are level {got.level}, not the registered level {work.level} of {work.n}"
  if let some records := records then
    if (records.length : Int) != bound || (records.drop shape.length).any (· != .null) then
      return some s!"{records.length} records, not one per call and None past the calls"
    for e in [0:shape.length] do
      if let some why := (← recordProblem entries[e]! records[e]!) then return some s!"entry {e}: {why}"
  return none

end PouwWindow
