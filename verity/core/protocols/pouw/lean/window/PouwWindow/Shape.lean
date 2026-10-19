import PouwWindow.Entry
import PouwWindow.Json

/-!
# The window's shape (`window.py`, "the shape")

A window's structure is a function of its public shape alone: its calls (`Call`), each its batch slots `m`, its weight
rows `n` and its first tile `tile0` in its pass. Call c's entry holds every slot and every weight row (n_a = m, n_b = n)
and the m·n tiles after the calls before it; the filler's one tile is the last, N − 1 (`shapeTiles`, `shapeEntries`); and
`tileAt?` maps a drawn index to its entry and rows from the shape and no value. `shapeProblem` holds a shape to what a
window of a given entry bound can have.

Beside the shape, the window's fixed length (`Length`, CMP-43): its rows ran ignoring end-of-sequence at a public
`max_tokens` per call, so its useful work is public; `fixedLength` and `lengthProblem` hold a window of record to it.
-/

namespace PouwWindow

open Lean (Json)

/-- One call of a window's shape: its batch slots, its weight rows and its first tile in its pass. -/
structure Call where
  m : Nat
  n : Nat
  tile0 : Nat
  deriving DecidableEq, Repr

def Call.toJson (c : Call) : Json := Json.mkObj [("m", c.m), ("n", c.n), ("tile0", c.tile0)]

/-- `shape_json`. -/
def shapeJson (shape : List Call) : Json := Json.arr (shape.map Call.toJson).toArray

private def callOf? : Json → Option Call
  | c@(.obj kvs) =>
    if kvs.toArray.toList.map (·.1) == ["m", "n", "tile0"] then do
      pure ⟨← (field? c "m").bind natOf?, ← (field? c "n").bind natOf?, ← (field? c "tile0").bind natOf?⟩
    else none
  | _ => none

/-- `shape_of`: the shape a JSON value holds, a non-empty list of calls `{m, n, tile0}` of natural numbers, or `none`
where Python raises. -/
def shapeOf? : Json → Option (List Call)
  | .arr a => if a.isEmpty then none else a.toList.mapM callOf?
  | _ => none

/-- `shape_tiles`: N, the calls' m·n tiles each and the filler's one. -/
def shapeTiles (shape : List Call) : Nat := (shape.map fun c => c.m * c.n).sum + 1

/-- `shape_problem`: why `shape` is not the shape of a window of `bound` entries, or `none`. -/
def shapeProblem (shape : List Call) (bound : Int) : Option String := Id.run do
  if !(1 ≤ shape.length && (shape.length : Int) ≤ bound - 1) then
    return some s!"{shape.length} calls, not 1 to the bound {bound} less the filler"
  let mut inPass := 0
  for (call, c) in shape.zipIdx do
    if !(rowStep ≤ call.m && call.m ≤ 2 ^ Side.a.depth && call.m % rowStep == 0) then
      return some s!"call {c}: its m {call.m} is not a multiple of {rowStep} from {rowStep} to 2^{Side.a.depth}"
    if !(1 ≤ call.n && call.n ≤ 2 ^ Side.b.depth) then
      return some s!"call {c}: its n {call.n} is not 1 to 2^{Side.b.depth}"
    if call.tile0 < inPass then
      return some s!"call {c}: its pass tiles start at {call.tile0}, before the last call's end {inPass}"
    inPass := call.tile0 + (call.m + tile - 1) / tile * ((call.n + tile - 1) / tile)
    if inPass > 2 ^ tBits then
      return some s!"call {c}: its pass tiles end at {inPass}, beyond 2^{tBits}"
  if shapeTiles shape ≥ 2 ^ tBits then
    return some s!"its {shapeTiles shape} tiles are not of {tBits}-bit indices"
  return none

/-- A window's fixed length (`window.Length`): whether its rows ran ignoring end-of-sequence (vLLM's `ignore_eos`), and
each call's `max_tokens` L, in the shape's order. -/
structure Length where
  ignoreEos : Bool
  maxTokens : List Nat
  deriving DecidableEq, Repr

def Length.toJson (l : Length) : Json :=
  Json.mkObj [("ignore_eos", Json.bool l.ignoreEos),
    ("max_tokens", Json.arr (l.maxTokens.map fun (n : Nat) => Json.num n).toArray)]

/-- `length_of`: the length a JSON value holds, `{ignore_eos, max_tokens}`, a boolean and a non-empty list of natural
numbers, or `none` where Python raises. -/
def lengthOf? : Json → Option Length
  | j@(.obj kvs) =>
    if kvs.toArray.toList.map (·.1) == ["ignore_eos", "max_tokens"] then
      match field? j "ignore_eos", field? j "max_tokens" with
      | some (.bool b), some (.arr a) => if a.isEmpty then none else do pure ⟨b, ← a.toList.mapM natOf?⟩
      | _, _ => none
    else none
  | _ => none

/-- `fixed_length`: whether a JSON value says its rows ran at a fixed length, an object whose `ignore_eos` is true;
anything else is an EOS-keyed window's, which PoUW's check refuses as `window-form: eos-keyed`. -/
def fixedLength (j : Json) : Bool := field? j "ignore_eos" == some (.bool true)

/-- `length_problem`: why `length` is not a fixed length of `shape`'s calls, or `none`: its rows ignored
end-of-sequence, and it holds one max_tokens per call, each at least 1. -/
def lengthProblem (length : Length) (shape : List Call) : Option String := Id.run do
  if !length.ignoreEos then
    return some "its rows stop at end-of-sequence (ignore_eos false), not at a fixed length"
  if length.maxTokens.length != shape.length then
    return some s!"{length.maxTokens.length} lengths, not one per call of the shape's {shape.length}"
  for (n, c) in length.maxTokens.zipIdx do
    if n < 1 then
      return some s!"call {c}: its max_tokens {n} is not a positive length"
  return none

/-- `shape_entries`: the entries' public fields, each a function of the shape (their roots and records zero), then
empty entries up to `bound`. -/
def shapeEntries (shape : List Call) (bound : Nat) : List Entry := Id.run do
  let mut out : Array Entry := #[]
  let mut pos := 0
  for c in shape do
    out := out.push { kind := .call, lo := pos, hi := pos + c.m * c.n, nA := c.m, nB := c.n, tile0 := c.tile0,
                      cols := (c.n + tile - 1) / tile, m := c.m }
    pos := pos + c.m * c.n
  out := out.push { kind := .filler, lo := pos, hi := pos + 1, nA := 1, nB := 1 }
  return out.toList ++ List.replicate (bound - out.size) { kind := .none }

/-- `public_fields`: an entry's fields that the shape fixes, its kind, range, side rows, tile0, cols and m. -/
def Entry.publicFields (e : Entry) : Kind × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (e.kind, e.lo, e.hi, e.nA, e.nB, e.tile0, e.cols, e.m)

/-- `tile_at`: (entry, A row, B row) of drawn index `t`, from the shape alone: in call c's range, `t − lo_c = i·n_c + j`;
N − 1 is the filler's (C, 0, 0). -/
def tileAt? (shape : List Call) (t : Int) : Except String (Nat × Nat × Nat) := do
  if t < 0 then throw s!"index {t} is not one of the window's tiles"
  let t := t.toNat
  let mut pos := 0
  for (call, c) in shape.zipIdx do
    if t < pos + call.m * call.n then
      return (c, (t - pos) / call.n, (t - pos) % call.n)
    pos := pos + call.m * call.n
  if t == pos then return (shape.length, 0, 0)
  throw s!"index {t} is not one of the window's {pos + 1} tiles"

end PouwWindow
