import PouwWindow.Record

/-!
# The window statement (`window.py`, `window_problem`)

L4 on the committed entry rows: `bound` rows, each an entry row of the layout; the calls, then one filler, then empty
entries; the ranges tile [0, N) in order; a call's range is its tiles and the filler's holds the rest within its rows;
every call's -h3 root is the pass's; the calls' pass tiles are in order and disjoint; and the registered level is
`bucket` of the calls' tiles. Given the calls' records it holds each to its entry (`recordProblem`), and given the side
rows' indices it holds a call's to its record's keep and the filler's to zero. A problem is a value (`some why`); a
refusal (`.error`) is where Python raises `ValueError`, or would raise another error on input off its types.
-/

namespace PouwWindow

open Lean (Json)

/-- Python's `int(i)` of a side row's index as JSON holds it: an integer or a boolean. -/
private def indexInt? (j : Json) : Except String Int :=
  match pyInt? j with
  | some i => .ok i
  | none => .error s!"int() of {pyRepr j}"

/-- A call's record's keep on `side`, as `record_problem` has accepted it: its strictly increasing integers. -/
private def keepOf? (record : Json) (side : Side) : Except String (List Int) :=
  match (pyGet? record "keep").bind (pyGet? · side.name) |>.bind pyList? |>.bind (·.mapM strictInt?) with
  | some ks => .ok ks
  | none => .error s!"the record's keep[{PouwCommit.pyRepr side.name}] is not a list of integers"

/-- `window_problem`: the window statement on the entry rows, or `none`. -/
def windowProblem (rows : List ByteArray) (work : Work) (bound : Int) (rootH3 : ByteArray) (bits : Nat := gridBits)
    (records : Option (List Json) := none) (indices : Option (List Json) := none) : Except String (Option String) := do
  if rootH3.size != 64 then throw "the pass's -h3 tile root is 64 bytes"
  if (rows.length : Int) != bound then return some s!"{rows.length} entries, not the bound {bound}"
  let mut entries : Array Entry := #[]
  for (row, e) in rows.zipIdx do
    match Entry.fromRow? row with
    | .error x => return some s!"entry {e}: not an entry row: {x}"
    | .ok entry =>
      if let some why := entry.layoutProblem then return some s!"entry {e}: not the layout: {why}"
      entries := entries.push entry
  let kinds := entries.toList.map (·.kind)
  let shape := "the entries are the calls, then one filler, then empty entries"
  let some calls := kinds.findIdx? (· == .filler) | return some shape
  if calls < 1 || kinds.take calls != List.replicate calls .call || kinds.drop (calls + 1) != List.replicate
      (rows.length - calls - 1) .none then
    return some shape
  let mut pos := 0
  let mut inPass := 0
  for (entry, e) in (entries.toList.take (calls + 1)).zipIdx do
    if entry.lo != pos || entry.hi ≤ pos then
      return some s!"entry {e}: its range does not start where the last ended, or is empty"
    let size := entry.hi - entry.lo
    if entry.kind == .call && size != entry.tiles then
      return some s!"entry {e}: its range is {size} tiles, not its {entry.tiles} tiles"
    if entry.kind == .call && entry.rootH3 != rootH3 then
      return some s!"entry {e}: its -h3 root is not the pass's tile commitment"
    if entry.kind == .call && entry.tile0 < inPass then
      return some s!"entry {e}: its pass tiles start at {entry.tile0}, before the last call's end {inPass}"
    if entry.kind == .filler && size > entry.tiles then
      return some s!"entry {e}: the filler's range is {size} tiles, beyond its {entry.tiles}"
    pos := entry.hi
    if entry.kind == .call then inPass := entry.tile0 + entry.passTiles
  let real := entries[calls]!.lo
  if pos != work.n then return some s!"the ranges end at {pos}, not the padded count {work.n}"
  let got ← bucket? bits real
  if got != work then
    return some s!"{real} real tiles are level {got.level}, not the registered level {work.level}"
  if let some records := records then
    if (records.length : Int) != bound || (records.drop calls).any (· != .null) then
      return some s!"{records.length} records, not one per call and None past the calls"
    for e in [0:calls] do
      if let some why := (← recordProblem entries[e]! records[e]!) then return some s!"entry {e}: {why}"
  if let some indices := indices then
    let some records := records | throw "the side rows' indices are held to the calls' records, so give both"
    for (entry, e) in (entries.toList.take (calls + 1)).zipIdx do
      let got := indices.getD e .null
      for side in sides do
        let want ← if entry.kind == .call then keepOf? records[e]! side
          else pure (List.replicate (entry.nSide side) (0 : Int))
        let bad := some (s!"entry {e}: its side-{side.name} rows' indices are not " ++
          (if entry.kind == .call then "its record's keep" else "zero"))
        if got == .null then return bad
        let some (.arr is) := pyGet? got side.name | throw s!"entry {e}'s indices have no side {side.name}"
        let is ← is.toList.mapM indexInt?
        if is != want then return bad
  return none

end PouwWindow
