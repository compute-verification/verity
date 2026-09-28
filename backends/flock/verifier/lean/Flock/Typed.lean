/-
Item 1e (Project doc `verified-lowering-design.md`): the typed statement `verity/flock-circuit/types`
(`verity_flock.typed_statement`, #225), whose unit is a circuit type and a layout instead of rows. No statement carries the
unit's rows: the verifier derives them and places them as the old statement's `unit` slots. It folds rows only through
`DeriveAll.deriveChecked` (#247): `deriveAll`'s rows, used once every layout's pass the item-by-item check and the unit's
are in logical order. `flock-rows --part` prints unchecked rows, so nothing here reads them.

`expand` checks META's new keys and returns the file with the derived rows as its `CIRCUIT unit` section, first, where the
old statement had it. `HmRow.parseTyped` reads that as it reads any statement, and keeps the typed file's own digest and
class. The keys:
* `types`: `{digest: content}`, each the digest of its content and checked by `CircuitType.check` (#194);
* `layouts`: #200's records, each its `sha512`'s record, checked by `Layout.check`;
* `unit_layout`: the unit's layout, derived (`own` is `derive`);
* `unit_type`: that layout's type;
* `policy`: `calls/v1`;
* `unit_in`: the unit's constant-bound inputs, none in this version.

This version covers units without placed parts, Δ or reads (the flat classes), whose rows are one `CIRCUIT unit` section.
-/
import Flock.DeriveCheck
import Flock.Layout
import Flock.Net
import Flock.Tags

namespace Flock.Typed

open Lean (Json)

private def ctx : String := "typed statement META"

def POLICY : String := "calls/v1"

/-- The typed file's lines, its circuit's name and its META. Refuses a file that carries a unit's rows itself. -/
def split (tags : Tags) (bytes : ByteArray) : V (Array String × String × Json) := do
  let text ← match String.fromUTF8? bytes with
    | some s => pure s
    | none => throw "circuit: not UTF-8"
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let head := (lines.getD 0 "").trimAsciiEnd.toString
  if !head.startsWith (tags.fileFormat ++ " ") then throw s!"circuit: not a {tags.fileFormat} file"
  let name := (head.drop (tags.fileFormat.length + 1)).toString
  let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
  if !metaLine.startsWith "META " then throw "circuit: no META line"
  let mj ← match Json.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  let mut at_ := 2
  while at_ < lines.size do
    let h := tokens lines[at_]!
    if h.size != 3 || h[0]! != tags.blockKeyword then throw s!"circuit: expected a {tags.blockKeyword} line"
    if h[1]! == "unit" || h[1]! == "root" then
      throw "circuit: a typed statement carries no unit rows; the verifier derives them"
    let some n := h[2]!.toNat? | throw s!"circuit: {tags.blockKeyword} line count"
    at_ := at_ + 1 + n
  return (lines, name, mj)

/-- META's types and layouts, checked, and their rows through `DeriveAll.deriveChecked` only: every layout's rows, the
layouts, the unit's layout and its type. -/
def derivation (mj : Json) : V (Array DeriveAll.D × Array Layout × Nat × CircuitType) := do
  let kvs ← match mj.getObjVal? "types" with
    | .ok (Json.obj kvs) => pure kvs.toArray
    | _ => throw "META: types is not an object"
  let mut types : Array (String × CircuitType) := #[]
  for (d, c) in kvs do
    let t ← CircuitType.ofJson c
    if (← t.digest) != d then throw s!"META: type {d.take 16}… is not its content's digest"
    types := types.push (d, t)
  let typeOf := fun (x : String) => (types.find? (·.1 == x)).map (·.2)
  for (d, t) in types do
    if let .error e := CircuitType.check typeOf t then throw s!"META: type {d.take 16}… breaks the rule '{e}'"
  let ls ← (← J.arr mj "layouts" ctx).mapM fun r => do
    let l ← Layout.ofJson r
    if (← J.str r "sha512" ctx) != (← l.digest) then throw s!"{ctx}: a layout's sha512 is not its record's digest"
    pure l
  let unit ← J.nat mj "unit_layout" ctx
  if let .error e := Layout.check typeOf ls unit (fun _ => none) then throw s!"META: the layouts break the rule '{e}'"
  let some ul := ls[unit]? | throw "META: unit_layout names no layout"
  let some ut := ul.type | throw "META: the unit's layout names no type"
  if (← J.str mj "unit_type" ctx) != ut then throw "META: unit_type is not the unit's layout's type"
  if ul.own != .derive then throw "META: the unit's layout is not derived"
  if (← J.str mj "policy" ctx) != POLICY then throw s!"META: policy is not {POLICY}"
  if !(← J.arr mj "unit_in" ctx).isEmpty then
    throw "META: unit_in binds the unit's inputs to constants, which this verifier does not derive yet"
  let some t := typeOf ut | throw "META: unit_type names no type"
  let done ← match DeriveAll.deriveChecked typeOf ls (fun _ => none) (fun _ => none) unit with
    | .ok done => pure done
    | .error e => throw s!"META: the unit's rows: {e}"
  return (done, ls, unit, t)

/-- A flat class's unit: no placed parts, no reads, no Δ but its constant's copy of the block's `one` (which `HmRow.delta`
pins for every unit slot). -/
def isFlat (u : DeriveAll.D) : Bool :=
  u.parts.isEmpty && u.reads.isEmpty && u.delta.all (fun e => e.1 == u.const && e.2.isNone)

/-- A flat class's file with its unit's rows `u` inserted as its `CIRCUIT unit` section, first, where the old statement had
it; META and the other sections unchanged. -/
def expandWith (tags : Tags) (lines : Array String) (name : String) (u : DeriveAll.D) : ByteArray :=
  let rows := ((DeriveAll.netlist u name).splitOn "\n").toArray.pop
  let out := #[lines[0]!, lines[1]!, s!"{tags.blockKeyword} unit {rows.size}"] ++ rows ++ lines.extract 2 lines.size
  ("\n".intercalate out.toList ++ "\n").toUTF8

/-- A template's block as its types derive it (#273's `Instance`), for `HmRow.parse`: the root net (the unit's own region,
its entries at part columns left out, which are Δ's), each placed layout's net by digest, and the block's data in unit
coordinates. -/
structure Template where
  root : Net
  /-- each placed layout, in layout order: its digest, its net, its range's log size -/
  layouts : Array (String × Net × Nat)
  /-- per part, in item order: its layout's digest and its base -/
  parts : Array (String × Nat)
  own : Nat
  ins : Array (Nat × Nat)
  outs : Array (Nat × Nat)
  inCols : Array Nat
  bind : Array (Nat × Array Nat)
  cross : Array (Nat × Array Nat × Array Nat)
  deriving Inhabited

def netOf (text : String) : V Net := Net.parse ((text.splitOn "\n").toArray.pop)

/-- A template's block from its derivation. -/
def templateOf (name : String) (done : Array DeriveAll.D) (ls : Array Layout) (unit : Nat) (t : CircuitType) : V Template := do
  let u := done.getD unit default
  if !u.reads.isEmpty then throw "META: the unit reads tables, which this verifier's template reading does not place yet"
  let own := u.size
  let inOwn := fun (xs : List Nat) => xs.filter (· < own)
  let root ← netOf (DeriveAll.netlist { u with rows := u.rows.map fun (a, b) => (inOwn a, inOwn b) } name)
  let digests ← ls.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let layouts ← placed.toArray.mapM fun j => do
    let d := digests.getD j ""
    pure (d, ← netOf (DeriveAll.netlist (done.getD j default) (d.take 16).toString), (ls.getD j default).rangeLog.getD 0)
  let cross := (Array.range own).filterMap fun r =>
    let (a, b) := u.rows.getD r ([], [])
    let (xa, xb) := (a.filter (· ≥ own), b.filter (· ≥ own))
    if xa.isEmpty && xb.isEmpty then none else some (r, xa.toArray, xb.toArray)
  return { root, layouts, parts := u.parts.map fun (j, b) => (digests.getD j "", b), own, ins := t.ins, outs := t.outs,
           inCols := u.inCols, bind := u.delta.filterMap fun (r, src) => src.map (r, ·.toArray), cross }

/-- **A typed file, read** (item 1e): its rows derived once, through `deriveChecked`. A flat class is its file with the
unit's rows as its `CIRCUIT unit` section (`.inl`); a template is its block (`.inr`), which `HmRow.parse` places. -/
def read (tags : Tags) (bytes : ByteArray) : V (ByteArray ⊕ Template) := do
  let (lines, name, mj) ← split tags bytes
  let (done, ls, unit, t) ← derivation mj
  let u := done.getD unit default
  if isFlat u then return .inl (expandWith tags lines name u)
  return .inr (← templateOf name done ls unit t)

/-- A flat class's file with its unit's rows derived (`typed-expand`). -/
def expand (tags : Tags) (bytes : ByteArray) : V ByteArray := do
  match ← read tags bytes with
  | .inl b => pure b
  | .inr _ => throw "META: the unit has placed parts: a template, whose block `HmRow.parse` places, not a flat class"

end Flock.Typed
