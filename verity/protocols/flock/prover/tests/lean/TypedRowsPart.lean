import Flock.Lookup
import Flock.Typed

/-!
`typed-rows` (`Typed.rows`) cut where its work splits, so that `test_lean_typed_reads.py` runs the parts side by side: the
typed attention template takes the verifier minutes to derive and check, most of them in the unit's `checkLayout` and
`orderChecked`. Each part runs the verifier's own functions:

    typed-rows-part PART FILE [--tables DIR] [--out DIR]

Every part reads META as `classOf` does (`metaChecked`) and derives the layouts with `deriveAll`, as `deriveChecked` does. Then
* `rows`: every layout's rows pass `checkLayout`, and the rows are written to `--out` as `typed-rows` writes them (`files`);
* `order0`, `order1`: the unit's logical order up to `cut` and from it pass `orderOk`, and with `order1`, `outsideOk`;
* `parts`: `partsChecked`.

The theorems hold the parts to `typed-rows`: `classOf` is `metaChecked` and then `deriveChecked`; `deriveChecked` is `deriveAll`
and then `checks`, the parts' checks; `orderChecked` is `orderOk` up to and from any cut, and `outsideOk`; and `Typed.rows`
writes `files`. So every part passes exactly when `typed-rows` passes, and a change to one of those definitions that the parts
don't follow stops this file from building.
-/

open Lean (Json)
open Flock Flock.Typed Flock.DeriveAll

namespace TypedRowsPart

/-- META's types, layouts, held tables, unit and the unit's type, checked as `classOf` checks them, given to `k`. -/
def metaChecked {α : Type} (mj : Json) (tables : List (String × ByteArray))
    (k : Array (String × CircuitType) × Array Layout × Array Held × Nat × CircuitType → V α) : V α := do
  let kvs ← match mj.getObjVal? "types" with
    | .ok (Json.obj kvs) => pure kvs.toArray
    | _ => throw "META: types is not an object"
  let mut types : Array (String × CircuitType) := #[]
  for (d, c) in kvs do
    let t ← CircuitType.ofJson c
    if (← t.digest) != d then throw s!"META: type {d.take 16}… is not its content's digest"
    types := types.push (d, t)
  for (d, t) in types do
    if let .error e := CircuitType.check (typeIn types) t then throw s!"META: type {d.take 16}… breaks the rule '{e}'"
  let held ← hold types tables
  let ls ← (← J.arr mj "layouts" "typed statement META").mapM fun r => do
    let l ← Layout.ofJson r
    if (← J.str r "sha512" "typed statement META") != (← l.digest) then
      throw s!"{"typed statement META"}: a layout's sha512 is not its record's digest"
    pure l
  let unit ← J.nat mj "unit_layout" "typed statement META"
  if let .error e := Layout.check (typeIn types) ls unit (infoIn held) then throw s!"META: the layouts break the rule '{e}'"
  let some ul := ls[unit]? | throw "META: unit_layout names no layout"
  let some ut := ul.type | throw "META: the unit's layout names no type"
  if (← J.str mj "unit_type" "typed statement META") != ut then throw "META: unit_type is not the unit's layout's type"
  if ul.own != .derive then throw "META: the unit's layout is not derived"
  if (← J.str mj "policy" "typed statement META") != POLICY then throw s!"META: policy is not {POLICY}"
  if !(← J.arr mj "unit_in" "typed statement META").isEmpty then
    throw "META: unit_in binds the unit's inputs to constants, which this verifier does not derive yet"
  let some t := typeIn types ut | throw "META: unit_type names no type"
  k (types, ls, held, unit, t)

/-- What `classOf` does with META once it is checked: `deriveChecked`. -/
def derived (x : Array (String × CircuitType) × Array Layout × Array Held × Nat × CircuitType) : V (Class × CircuitType) :=
  let (types, ls, held, unit, t) := x
  match h : deriveChecked (typeIn types) ls (infoIn held) (wordsIn held) unit with
  | .ok done => return (⟨types, ls, held, unit, done, h⟩, t)
  | .error e => throw s!"META: the unit's rows: {e}"

theorem classOf_eq (mj : Json) (tables : List (String × ByteArray)) : classOf mj tables = metaChecked mj tables derived := rfl

/-- `deriveChecked`'s checks on `deriveAll`'s layouts. -/
def checks (types : String → Option CircuitType) (ls : Array Layout) (words : String → Option (Array Nat)) (unit : Nat)
    (done : Array D) : Except String (Array D) :=
  match (List.range ls.size).find? fun j => !checkLayout types words ls done unit j with
  | some j => throw s!"derive: layout {j}'s rows fail the check"
  | none =>
    if !orderChecked words done unit then throw "derive: the unit's rows are not in logical order"
    else if !partsChecked done unit then throw "derive: the unit's parts are not placed as its template reads them"
    else return done

theorem deriveChecked_eq (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo)
    (words : String → Option (Array Nat)) (unit : Nat) :
    deriveChecked types ls info words unit = (deriveAll types ls info words unit >>= checks types ls words unit) := rfl

/-- `orderOk` from position `ℓ` with `seen` on a list cut in two: the first part's, then the second's from where it ends. -/
theorem orderOk_append (words : String → Option (Array Nat)) (u : D) (nIn : Nat) (l₁ l₂ : List Nat) (ℓ : Nat)
    (seen : Std.HashSet Nat) :
    orderOk words u nIn (l₁ ++ l₂) ℓ seen =
      (orderOk words u nIn l₁ ℓ seen && orderOk words u nIn l₂ (ℓ + l₁.length) (l₁.foldl (·.insert ·) seen)) := by
  induction l₁ generalizing ℓ seen with
  | nil => simp [orderOk]
  | cons c rest ih =>
    simp only [List.cons_append, orderOk, ih, List.foldl_cons, List.length_cons, Bool.and_assoc]
    rw [show ℓ + 1 + rest.length = ℓ + (rest.length + 1) by omega]

/-- Where `order0` and `order1` cut the unit's logical order `o`. Nearly all of `orderOk`'s time on typed attention is in
the positions from 1/4 to 1/2 of its order, and this cuts them in half. -/
def cut (o : List Nat) : Nat := o.length * 3 / 8

/-- The unit's logical order `o` up to position `k`, from its start. -/
def before (words : String → Option (Array Nat)) (u : D) (o : List Nat) (k : Nat) : Bool :=
  orderOk words u u.inCols.size (o.take k) 0 {}

/-- `o` from position `k`, with what the positions before it have seen. -/
def after (words : String → Option (Array Nat)) (u : D) (o : List Nat) (k : Nat) : Bool :=
  orderOk words u u.inCols.size (o.drop k) (0 + (o.take k).length) ((o.take k).foldl (·.insert ·) {})

theorem orderChecked_eq (words : String → Option (Array Nat)) (done : Array D) (unit k : Nat) :
    orderChecked words done unit =
      (before words (done.getD unit default) (order done (done.getD unit default)) k &&
        after words (done.getD unit default) (order done (done.getD unit default)) k &&
        outsideOk (done.getD unit default) (order done (done.getD unit default))) := by
  simp only [orderChecked, before, after, ← orderOk_append, List.take_append_drop]

/-- The files `typed-rows` writes for the unit named `name`, from `deriveChecked`'s layouts. -/
def files (name : String) (done : Array D) (ls : Array Layout) (unit : Nat) : V (Array (String × String)) := do
  let u := done.getD unit default
  let digests ← ls.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let fs := placed.toArray.map fun j =>
    let d := digests.getD j ""
    (s!"{d}.txt", netlist (done.getD j default) (d.take 16).toString)
  return #[("root.txt", netlist u name)] ++ fs ++ #[("delta.txt", deltaText ls digests u)]

theorem rows_eq (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    Typed.rows tags bytes tables = (do
      let (_, name, mj) ← split tags bytes
      let (k, _) ← classOf mj tables
      files name k.done k.ls k.unit) := rfl

/-- Part `part` of `typed-rows` on a typed file: the files it writes (`rows`), or none. -/
def run (part : String) (bytes : ByteArray) (tables : List (String × ByteArray)) : V (Array (String × String)) := do
  let (_, name, mj) ← split Tags.circuitTypes bytes
  metaChecked mj tables fun (types, ls, held, unit, _) => do
    let done ← deriveAll (typeIn types) ls (infoIn held) (wordsIn held) unit
    let u := done.getD unit default
    match part with
    | "rows" =>
      if let some j := (List.range ls.size).find? fun j => !checkLayout (typeIn types) (wordsIn held) ls done unit j then
        throw s!"derive: layout {j}'s rows fail the check"
      files name done ls unit
    | "order0" =>
      let o := order done u
      if !before (wordsIn held) u o (cut o) then throw "derive: the unit's rows are not in logical order (before the cut)"
      return #[]
    | "order1" =>
      let o := order done u
      if !(after (wordsIn held) u o (cut o) && outsideOk u o) then
        throw "derive: the unit's rows are not in logical order (from the cut)"
      return #[]
    | "parts" =>
      if !partsChecked done unit then throw "derive: the unit's parts are not placed as its template reads them"
      return #[]
    | _ => throw s!"no part {part}: rows, order0, order1 or parts"

def argOf (args : List String) (flag : String) : Option String :=
  match args.dropWhile (· != flag) with
  | _ :: v :: _ => some v
  | _ => none

end TypedRowsPart

open TypedRowsPart in
def main (args : List String) : IO UInt32 := do
  let (part :: path :: _) := args | IO.eprintln "typed-rows-part PART FILE [--tables DIR] [--out DIR]"; return 2
  let mut tables := []
  if let some dir := argOf args "--tables" then
    for t in MUFU_TABLES do
      let p : System.FilePath := dir ++ s!"/{t.name}.u32"
      if ← p.pathExists then tables := tables ++ [(t.name, ← IO.FS.readBinFile p)]
  match run part (← IO.FS.readBinFile path) tables with
  | .error e => IO.eprintln s!"typed-rows-part {part}: {e}"; return 1
  | .ok files =>
    if let some out := argOf args "--out" then
      IO.FS.createDirAll out
      for (n, text) in files do IO.FS.writeFile (out ++ "/" ++ n) text
    IO.println s!"TYPED-ROWS-PART {part} {files.size}"
    return 0
